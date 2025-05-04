#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <bitset>
#include <mutex>
#include <array>
#include <cmath>
#include <random>
#include <filesystem>              // for checking/creating the model file
#include "ooo_cpu.h"
#include "gen_config.h"

namespace fs = std::filesystem;

// ──────────────────────────────────────────────────────────────────────────
// PERSISTENCE CONFIGURATION
// ──────────────────────────────────────────────────────────────────────────
// (Persistence features removed for now)

// ──────────────────────────────────────────────────────────────────────────
// GLOBAL DETERMINISM SETUP
// ──────────────────────────────────────────────────────────────────────────
struct DeterministicSetup {
    DeterministicSetup() {
        torch::manual_seed(42);
        std::srand(42);
        at::globalContext().setDeterministicCuDNN(true);
        at::globalContext().setBenchmarkCuDNN(false);
        at::set_num_threads(1);
    }
};
static DeterministicSetup _deterministic_setup;

// -------------------------------------------------------------------------
// Model & History Config (parameters provided via gen_config.h)
// -------------------------------------------------------------------------
constexpr size_t GLOBAL_HISTORY_REGISTER_BITS = 4096;
constexpr int    MAX_HISTORY   = 4096;

constexpr int ceil_mul(int current, double alpha) {
    double prod = current * alpha;
    int rounded = static_cast<int>(prod);
    if (prod > static_cast<double>(rounded)) ++rounded;
    return (rounded > current) ? rounded : (current + 1);
}

template <int N>
constexpr std::array<int, N> compute_fixed_history_lengths() {
    std::array<int, N> arr = {};
    arr[0] = START_HISTORY;
    for (int i = 1; i < N; i++) {
        arr[i] = ceil_mul(arr[i - 1], ALPHA);
        if (arr[i] > MAX_HISTORY) arr[i] = MAX_HISTORY;
    }
    return arr;
}
constexpr auto fixedHistoryLengths = compute_fixed_history_lengths<NUM_LOCAL_HISTORY_SCALES>();

// Global history register per CPU
std::vector<std::bitset<GLOBAL_HISTORY_REGISTER_BITS>> Global_History(NUM_CPUS);

// ──────────────────────────────────────────────────────────────────────────
// PATH HISTORY REGISTER
// ──────────────────────────────────────────────────────────────────────────
// For each CPU, maintain a register (vector) of the most recent branch addresses.
std::vector<std::vector<uint64_t>> Path_History(NUM_CPUS, std::vector<uint64_t>(PATH_HISTORY_REG_SIZE, 0));

// ──────────────────────────────────────────────────────────────────────────
// Hashing helper for Local History
// ──────────────────────────────────────────────────────────────────────────
inline size_t compute_branch_index(uint64_t ip,
    const std::bitset<GLOBAL_HISTORY_REGISTER_BITS>& ghist,
    int history_length,
    int target_bits,
    int scale,
    size_t table_size)
{
    uint32_t folded = 0;
    for (int i = 0; i < history_length; i++) {
        if (ghist[i])
            folded ^= ((i + 1) * 0x9e3779b9U) >> (i % target_bits);
    }
    folded &= (1U << target_bits) - 1;

    uint32_t ip_part = static_cast<uint32_t>((ip ^ (ip >> 2)) & 0xFFFF);
    uint32_t mixed = ip_part ^ folded ^ (folded << (scale % 13));
    mixed ^= (mixed >> (scale + 1));

    return mixed % table_size;
}

// ──────────────────────────────────────────────────────────────────────────
// Helper: Compute Path History Token via Hashing & Folding
// ──────────────────────────────────────────────────────────────────────────
inline uint16_t compute_path_history_token(const std::vector<uint64_t>& pathHistory, int token_bits) {
    uint64_t hash = 0;
    for (size_t i = 0; i < pathHistory.size(); i++) {
        uint64_t val = pathHistory[i];
        hash ^= (val >> (i % 32)) ^ (val << (i % 16));
    }
    uint16_t folded = 0;
    while (hash) {
        folded ^= static_cast<uint16_t>(hash & ((1 << token_bits) - 1));
        hash >>= token_bits;
    }
    return folded;
}

// -------------------------------------------------------------------------
// Multi-head Self-Attention
// -------------------------------------------------------------------------
class MultiHeadSelfAttention : public torch::nn::Module {
public:
    MultiHeadSelfAttention()
      : W_q(register_parameter("W_q", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
        W_k(register_parameter("W_k", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
        W_v(register_parameter("W_v", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
        W_o(register_parameter("W_o", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
        forward_count(0)
    {
        torch::nn::init::xavier_uniform_(W_q);
        torch::nn::init::xavier_uniform_(W_k);
        torch::nn::init::xavier_uniform_(W_v);
        torch::nn::init::xavier_uniform_(W_o);
        dropout = register_module("dropout", torch::nn::Dropout(DROPOUT_RATE));
    }

    torch::Tensor forward(torch::Tensor x, int /*step*/) {
        // Get batch size and sequence length.
        const auto bs = x.size(0);
        const auto seq = x.size(1);
        const auto d = HIDDEN_SIZE / NUM_HEADS;  // dimension per head
    
        // Compute linear projections and immediately reshape & permute to separate heads.
        auto Q = torch::matmul(x, W_q)
                      .view({bs, seq, NUM_HEADS, d})
                      .permute({0, 2, 1, 3});
        auto K = torch::matmul(x, W_k)
                      .view({bs, seq, NUM_HEADS, d})
                      .permute({0, 2, 1, 3});
        auto V = torch::matmul(x, W_v)
                      .view({bs, seq, NUM_HEADS, d})
                      .permute({0, 2, 1, 3});
    
        // Apply the nonlinearity (ELU + 1) as in FAVOR+ style attention.
        auto phi_Q = torch::elu(Q) + 1;
        auto phi_K = torch::elu(K) + 1;
    
        // Compute an intermediate product for values: KV = phi_K^T * V.
        auto KV = torch::matmul(phi_K.transpose(-2, -1), V);
    
        // Numerator: phi_Q * (phi_K^T * V)
        auto numerator = torch::matmul(phi_Q, KV);
    
        // Denominator: Rather than multiplying by an all-ones vector, we sum phi_K along the sequence dimension.
        // phi_K.sum(2, true) has shape {bs, NUM_HEADS, 1, d}, so transpose last dims and perform matmul with phi_Q.
        auto denominator = torch::matmul(phi_Q, phi_K.sum(2, /*keepdim=*/true).transpose(-2, -1))
                               .clamp_min(1e-6);
    
        // Compute attention output by element-wise division.
        auto attn = numerator / denominator;
        
        // Apply dropout in training mode.
        if (this->is_training()) {
            attn = dropout->forward(attn);
        }
    
        // Recombine the heads by permuting and reshaping.
        auto out = attn.permute({0, 2, 1, 3}).contiguous().view({bs, seq, HIDDEN_SIZE});
        ++forward_count;
    
        // Final linear projection.
        return torch::matmul(out, W_o);
    }
    
private:
    torch::Tensor W_q, W_k, W_v, W_o;
    torch::nn::Dropout dropout{nullptr};
    int forward_count;
};

// -------------------------------------------------------------------------
// MultiScale Attention Predictor
// -------------------------------------------------------------------------
class MultiScaleAttentionPredictor : public torch::nn::Module {
public:
    MultiScaleAttentionPredictor() : update_count(0), load_count(0) {
        for (size_t scale = 0; scale < NUM_LOCAL_HISTORY_SCALES; scale++) {
            uint8_t bit_count = LOCAL_HISTORY_TABLE_BITS[scale];
            multiLocalHistoryTables.emplace_back(LOCAL_HISTORY_TABLE_SIZES[scale], 0);
            // Removed validMultiLocalHistoryTables usage entirely.
            auto proj = torch::nn::Linear(bit_count + 1, HIDDEN_SIZE);
            fc_local_combined.push_back(
              register_module("fc_local_combined_" + std::to_string(scale), proj));
        }
        fc_global_history = register_module("fc_global_history", torch::nn::Linear(GLOBAL_HISTORY_TOKEN_BITS, HIDDEN_SIZE));
        fc_ip_lsb = register_module("fc_ip_lsb", torch::nn::Linear(IP_LSB_TOKEN_BITS, HIDDEN_SIZE));
        fc_path_history = register_module("fc_path_history", torch::nn::Linear(PATH_HISTORY_TOKEN_BITS, HIDDEN_SIZE));

        attention  = register_module("attention", std::make_shared<MultiHeadSelfAttention>());
        fc_out     = register_module("fc_out", torch::nn::Linear(HIDDEN_SIZE, 1));
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(std::vector<int64_t>{HIDDEN_SIZE}));
        optimizer  = std::make_unique<torch::optim::Adam>(parameters(), LEARNING_RATE);
    }

    torch::Tensor build_tokens(uint64_t ip, int cpu) {
        // Reserve space for all tokens to avoid dynamic reallocations.
        std::vector<torch::Tensor> tokens;
        tokens.reserve(NUM_LOCAL_HISTORY_SCALES + 3);
    
        // Build tokens for local history scales.
        for (size_t scale = 0; scale < NUM_LOCAL_HISTORY_SCALES; scale++) {
            int historyLength = fixedHistoryLengths[scale];
            int target_bits   = static_cast<int>(std::log2(LOCAL_HISTORY_TABLE_SIZES[scale]));
            uint8_t bit_count = LOCAL_HISTORY_TABLE_BITS[scale];
            size_t table_size = LOCAL_HISTORY_TABLE_SIZES[scale];
    
            size_t idx = compute_branch_index(ip, Global_History[cpu],
                                              historyLength, target_bits, scale, table_size);
    
            // Use torch::empty instead of zeros to avoid unnecessary initialization.
            auto raw = torch::empty({1, bit_count + 1}, torch::kFloat32);
            // Use a raw pointer to directly access memory.
            float* raw_data = raw.data_ptr<float>();
    
            uint16_t old = multiLocalHistoryTables[scale][idx];
            for (uint8_t j = 0; j < bit_count; j++) {
                raw_data[j] = ((old >> j) & 1) ? 1.0f : 0.0f;
            }
            // Set the validity bit explicitly.
            raw_data[bit_count] = 1.0f;
    
            // Process the token via the corresponding fully-connected (fc) layer.
            tokens.push_back(fc_local_combined[scale](raw));
        }
    
        // Global history token
        {
            auto global_token = torch::empty({1, GLOBAL_HISTORY_TOKEN_BITS}, torch::kFloat32);
            float* global_data = global_token.data_ptr<float>();
    
            for (int i = 0; i < GLOBAL_HISTORY_TOKEN_BITS; i++) {
                global_data[i] = Global_History[cpu].test(i) ? 1.0f : 0.0f;
            }
            tokens.push_back(fc_global_history(global_token));
        }
    
        // IP LSB token
        {
            auto ip_lsb_token = torch::empty({1, IP_LSB_TOKEN_BITS}, torch::kFloat32);
            float* ip_lsb_data = ip_lsb_token.data_ptr<float>();
    
            for (int i = 0; i < IP_LSB_TOKEN_BITS; i++) {
                ip_lsb_data[i] = ((ip >> i) & 1) ? 1.0f : 0.0f;
            }
            tokens.push_back(fc_ip_lsb(ip_lsb_token));
        }
    
        // Path history token
        {
            uint16_t path_token_int = compute_path_history_token(Path_History[cpu], PATH_HISTORY_TOKEN_BITS);
            auto path_token = torch::empty({1, PATH_HISTORY_TOKEN_BITS}, torch::kFloat32);
            float* path_data = path_token.data_ptr<float>();
    
            for (int i = 0; i < PATH_HISTORY_TOKEN_BITS; i++) {
                path_data[i] = ((path_token_int >> i) & 1) ? 1.0f : 0.0f;
            }
            tokens.push_back(fc_path_history(path_token));
        }
    
        // Concatenate all tokens and add a batch dimension.
        return torch::cat(tokens, 0).unsqueeze(0);
    }
    

    uint8_t predict(uint64_t ip, int cpu) {
        torch::NoGradGuard no_grad;
        eval();
        auto tokens = build_tokens(ip, cpu);
        auto norm   = layer_norm(tokens);
        auto attn   = attention->forward(norm, 0);
        auto agg    = attn.mean(1);
        auto p      = torch::sigmoid(fc_out(agg)).clamp(1e-6, 1-1e-6);
        return (p.item<double>() > 0.5) ? 1 : 0;
    }

    void update(uint64_t ip, uint8_t taken, int cpu) {
        train();
        auto tokens = build_tokens(ip, cpu);
        auto norm   = layer_norm(tokens);
        auto attn   = attention->forward(norm, 0);
        auto agg    = attn.mean(1);
        auto pred   = torch::sigmoid(fc_out(agg)).clamp(1e-6, 1-1e-6);
        auto target = torch::full({1,1}, static_cast<float>(taken));
        auto loss   = torch::binary_cross_entropy(pred, target);

        optimizer->zero_grad();
        loss.backward();
        optimizer->step();

        for (size_t scale = 0; scale < NUM_LOCAL_HISTORY_SCALES; scale++) {
            int historyLength = fixedHistoryLengths[scale];
            int target_bits   = static_cast<int>(std::log2(LOCAL_HISTORY_TABLE_SIZES[scale]));
            uint8_t bit_count = LOCAL_HISTORY_TABLE_BITS[scale];
            size_t table_size = LOCAL_HISTORY_TABLE_SIZES[scale];

            size_t idx = compute_branch_index(ip, Global_History[cpu],
                                              historyLength, target_bits, scale, table_size);
            uint8_t &entry = multiLocalHistoryTables[scale][idx];
            entry = ((entry << 1) | taken) & ((1 << bit_count) - 1);
        }
        Global_History[cpu] <<= 1;
        Global_History[cpu].set(0, taken);

        {
            auto &path_reg = Path_History[cpu];
            for (int i = PATH_HISTORY_REG_SIZE - 1; i > 0; i--) {
                path_reg[i] = path_reg[i - 1];
            }
            path_reg[0] = ip;
        }
        // Persistence code removed.
    }

    void save(torch::serialize::OutputArchive &archive) const override {
        torch::nn::Module::save(archive);
    }
    void load(torch::serialize::InputArchive &archive) override {
        torch::nn::Module::load(archive);
    }

private:
    std::vector<torch::nn::Linear>        fc_local_combined;
    std::vector<std::vector<uint8_t>>     multiLocalHistoryTables;
    std::vector<std::vector<bool>>        validMultiLocalHistoryTables;
    std::shared_ptr<MultiHeadSelfAttention> attention;
    torch::nn::LayerNorm                  layer_norm{nullptr};
    torch::nn::Linear                     fc_out{nullptr};
    torch::nn::Linear                     fc_global_history{nullptr};
    torch::nn::Linear                     fc_ip_lsb{nullptr};
    torch::nn::Linear                     fc_path_history{nullptr};
    std::unique_ptr<torch::optim::Adam>   optimizer;
    std::mutex                            save_mutex;
    int                                   update_count, load_count;
};

MultiScaleAttentionPredictor global_predictor;

// ──────────────────────────────────────────────────────────────────────────
// O3_CPU interface
// ──────────────────────────────────────────────────────────────────────────
void O3_CPU::initialize_branch_predictor() {
    for (size_t i = 0; i < NUM_LOCAL_HISTORY_SCALES; i++)
        std::cout << "Fixed history length[" << i << "] = " << fixedHistoryLengths[i] << "\n";
}

uint8_t O3_CPU::predict_branch(uint64_t ip) {
    return global_predictor.predict(ip, cpu);
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t, uint8_t taken, uint8_t) {
    global_predictor.update(ip, taken, cpu);
}
