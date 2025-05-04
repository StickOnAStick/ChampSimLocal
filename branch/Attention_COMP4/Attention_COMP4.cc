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
        auto bs = x.size(0), seq = x.size(1);
        auto Q = torch::matmul(x, W_q)
                     .view({bs, seq, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS})
                     .permute({0, 2, 1, 3});
        auto K = torch::matmul(x, W_k)
                     .view({bs, seq, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS})
                     .permute({0, 2, 1, 3});
        auto V = torch::matmul(x, W_v)
                     .view({bs, seq, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS})
                     .permute({0, 2, 1, 3});

        auto phi_Q = torch::elu(Q) + 1;
        auto phi_K = torch::elu(K) + 1;
        auto KV    = torch::matmul(phi_K.transpose(-2, -1), V);
        auto num   = torch::matmul(phi_Q, KV);

        auto ones  = torch::ones({bs, NUM_HEADS, phi_K.size(2), 1}, phi_K.options());
        auto den   = torch::matmul(phi_Q, torch::matmul(phi_K.transpose(-2, -1), ones))
                       .clamp_min(1e-6);

        auto attn  = num / den;
        if (is_training()) attn = dropout->forward(attn);

        auto out = attn.permute({0, 2, 1, 3})
                       .contiguous()
                       .view({bs, seq, HIDDEN_SIZE});
        forward_count++;
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
        std::vector<torch::Tensor> tokens;
        for (size_t scale = 0; scale < NUM_LOCAL_HISTORY_SCALES; scale++) {
            int historyLength = fixedHistoryLengths[scale];
            int target_bits   = static_cast<int>(std::log2(LOCAL_HISTORY_TABLE_SIZES[scale]));
            uint8_t bit_count = LOCAL_HISTORY_TABLE_BITS[scale];
            size_t table_size = LOCAL_HISTORY_TABLE_SIZES[scale];

            size_t idx = compute_branch_index(ip, Global_History[cpu],
                                              historyLength, target_bits, scale, table_size);

            auto raw = torch::zeros({1, bit_count + 1}, torch::kFloat32);
            uint16_t old = multiLocalHistoryTables[scale][idx];
            for (uint8_t j = 0; j < bit_count; j++) {
                raw[0][j] = ((old >> j) & 1) ? 1.0f : 0.0f;
            }
            // Always set the validity bit to 1.0f.
            raw[0][bit_count] = 1.0f;
            tokens.push_back(fc_local_combined[scale](raw));
        }
        {
            auto global_token = torch::zeros({1, GLOBAL_HISTORY_TOKEN_BITS}, torch::kFloat32);
            for (int i = 0; i < GLOBAL_HISTORY_TOKEN_BITS; i++) {
                global_token[0][i] = Global_History[cpu].test(i) ? 1.0f : 0.0f;
            }
            tokens.push_back(fc_global_history(global_token));
        }
        {
            auto ip_lsb_token = torch::zeros({1, IP_LSB_TOKEN_BITS}, torch::kFloat32);
            for (int i = 0; i < IP_LSB_TOKEN_BITS; i++) {
                ip_lsb_token[0][i] = ((ip >> i) & 1) ? 1.0f : 0.0f;
            }
            tokens.push_back(fc_ip_lsb(ip_lsb_token));
        }
        {
            uint16_t path_token_int = compute_path_history_token(Path_History[cpu], PATH_HISTORY_TOKEN_BITS);
            auto path_token = torch::zeros({1, PATH_HISTORY_TOKEN_BITS}, torch::kFloat32);
            for (int i = 0; i < PATH_HISTORY_TOKEN_BITS; i++) {
                path_token[0][i] = ((path_token_int >> i) & 1) ? 1.0f : 0.0f;
            }
            tokens.push_back(fc_path_history(path_token));
        }
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
