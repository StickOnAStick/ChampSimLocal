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
#include "gen_config.h"            // must define START_HISTORY, ALPHA, NUM_LOCAL_HISTORY_SCALES, etc.

namespace fs = std::filesystem;

// ──────────────────────────────────────────────────────────────────────────
// Persistence & Save/Load Configuration
// ──────────────────────────────────────────────────────────────────────────
constexpr bool ENABLE_LOAD        = false;   // turn on loading
constexpr bool LOAD_INITIAL       = true; 
constexpr bool ENABLE_SAVE        = false;   // turn on saving
constexpr int  SAVE_INTERVAL      = 1000;    // save every 1000 updates
constexpr int  LOAD_INTERVAL      = SAVE_INTERVAL * 1.618; // reload every ~1618 updates
const std::string MODEL_FILE_PATH = "multiscale_model.pt";  // adjust path as needed

// ──────────────────────────────────────────────────────────────────────────
// Prediction Threshold Calibration
// ──────────────────────────────────────────────────────────────────────────
constexpr float PRED_THRESHOLD = 0.5f;    // decision threshold (tune on a warm-up set)

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

// ──────────────────────────────────────────────────────────────────────────
// Model & History Config (parameters provided via gen_config.h)
// ──────────────────────────────────────────────────────────────────────────
constexpr size_t GLOBAL_HISTORY_REGISTER_BITS = 8192;
constexpr int    MAX_HISTORY                  = 8192;

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
// Path history register per CPU
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
        const int64_t bs  = x.size(0);
        const int64_t seq = x.size(1);
        const int      d   = HIDDEN_SIZE / NUM_HEADS;
        auto Q = torch::matmul(x, W_q).view({bs,seq,NUM_HEADS,d}).permute({0,2,1,3});
        auto K = torch::matmul(x, W_k).view({bs,seq,NUM_HEADS,d}).permute({0,2,1,3});
        auto V = torch::matmul(x, W_v).view({bs,seq,NUM_HEADS,d}).permute({0,2,1,3});

        auto phi_Q = torch::elu(Q) + 1;
        auto phi_K = torch::elu(K) + 1;
        auto KV    = torch::matmul(phi_K.transpose(-2,-1), V);
        auto num   = torch::matmul(phi_Q, KV);
        auto den   = torch::matmul(phi_Q, phi_K.sum(2,true).transpose(-2,-1)).clamp_min(1e-6);
        auto attn  = num / den;

        if (is_training())
            attn = dropout->forward(attn);

        auto out = attn.permute({0,2,1,3}).contiguous().view({bs,seq,HIDDEN_SIZE});
        ++forward_count;
        return torch::matmul(out, W_o);
    }

private:
    torch::Tensor W_q, W_k, W_v, W_o;
    torch::nn::Dropout dropout{nullptr};
    int forward_count;
};

// -------------------------------------------------------------------------
// MultiScale Attention Predictor with persistence
// -------------------------------------------------------------------------
class MultiScaleAttentionPredictor : public torch::nn::Module {
public:
    MultiScaleAttentionPredictor()
      : update_count(0), load_count(0)
    {
        // Build local-history projections
        for (size_t s = 0; s < NUM_LOCAL_HISTORY_SCALES; s++) {
            uint8_t bits = LOCAL_HISTORY_TABLE_BITS[s];
            multiLocalHistoryTables.emplace_back(LOCAL_HISTORY_TABLE_SIZES[s], 0);
            auto lin = torch::nn::Linear(bits+1, HIDDEN_SIZE);
            fc_local_combined.push_back(
                register_module("fc_local_combined_" + std::to_string(s), lin));
        }
        fc_global_history = register_module("fc_global_history",
            torch::nn::Linear(GLOBAL_HISTORY_TOKEN_BITS, HIDDEN_SIZE));
        fc_ip_lsb = register_module("fc_ip_lsb",
            torch::nn::Linear(IP_LSB_TOKEN_BITS, HIDDEN_SIZE));
        fc_path_history = register_module("fc_path_history",
            torch::nn::Linear(PATH_HISTORY_TOKEN_BITS, HIDDEN_SIZE));

        attention  = register_module("attention", std::make_shared<MultiHeadSelfAttention>());
        fc_out     = register_module("fc_out", torch::nn::Linear(HIDDEN_SIZE,1));
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(std::vector<int64_t>{HIDDEN_SIZE}));

        // Adam with small L2 weight decay
        optimizer = std::make_unique<torch::optim::Adam>(
            parameters(),
            torch::optim::AdamOptions(LEARNING_RATE).weight_decay(1e-5)
        );

        // Ensure directory exists
        if (ENABLE_SAVE) {
            if (auto p = fs::path(MODEL_FILE_PATH).parent_path(); !p.empty() && p != ".")
                fs::create_directories(p);
        }

        // Initial load if enabled
        if (LOAD_INITIAL && fs::exists(MODEL_FILE_PATH)) {
            std::lock_guard<std::mutex> lock(save_mutex);
            try {
                torch::serialize::InputArchive ia;
                ia.load_from(MODEL_FILE_PATH);
                load(ia);
                std::cout << "[Predictor] Loaded model from " << MODEL_FILE_PATH << "\n";
            } catch (const c10::Error &e) {
                std::cerr << "[Predictor] Load warning: " << e.what() << "\n";
            }
        }
    }

    torch::Tensor build_tokens(uint64_t ip, int cpu) {
        std::vector<torch::Tensor> tokens;
        tokens.reserve(NUM_LOCAL_HISTORY_SCALES + 3);

        // Local-history tokens (zero-centered ±1)
        for (size_t s = 0; s < NUM_LOCAL_HISTORY_SCALES; s++) {
            int len    = fixedHistoryLengths[s];
            int tbits  = int(std::log2(LOCAL_HISTORY_TABLE_SIZES[s]));
            uint8_t bc = LOCAL_HISTORY_TABLE_BITS[s];
            size_t sz  = LOCAL_HISTORY_TABLE_SIZES[s];
            size_t idx = compute_branch_index(ip, Global_History[cpu], len, tbits, s, sz);

            auto raw = torch::empty({1, bc+1}, torch::kFloat32);
            float* d = raw.data_ptr<float>();
            uint16_t old = multiLocalHistoryTables[s][idx];
            for (uint8_t j = 0; j < bc; j++)
                d[j] = ((old>>j)&1) ? +1.0f : -1.0f;
            d[bc] = +1.0f;  // bias

            tokens.push_back(fc_local_combined[s](raw));
        }

        // Global-history token (zero-centered ±1)
        {
            auto tok = torch::empty({1, GLOBAL_HISTORY_TOKEN_BITS}, torch::kFloat32);
            float* d = tok.data_ptr<float>();
            for (int i = 0; i < GLOBAL_HISTORY_TOKEN_BITS; i++)
                d[i] = Global_History[cpu].test(i) ? +1.0f : -1.0f;
            tokens.push_back(fc_global_history(tok));
        }

        // IP-LSB token (zero-centered ±1)
        {
            auto tok = torch::empty({1, IP_LSB_TOKEN_BITS}, torch::kFloat32);
            float* d = tok.data_ptr<float>();
            for (int i = 0; i < IP_LSB_TOKEN_BITS; i++)
                d[i] = ((ip>>i)&1) ? +1.0f : -1.0f;
            tokens.push_back(fc_ip_lsb(tok));
        }

        // Path-history token (zero-centered ±1)
        {
            uint16_t pv = compute_path_history_token(Path_History[cpu], PATH_HISTORY_TOKEN_BITS);
            auto tok = torch::empty({1, PATH_HISTORY_TOKEN_BITS}, torch::kFloat32);
            float* d = tok.data_ptr<float>();
            for (int i = 0; i < PATH_HISTORY_TOKEN_BITS; i++)
                d[i] = ((pv>>i)&1) ? +1.0f : -1.0f;
            tokens.push_back(fc_path_history(tok));
        }

        return torch::cat(tokens, 0).unsqueeze(0);
    }

    // Predict with BCEWithLogits and calibrated threshold
    uint8_t predict(uint64_t ip, int cpu) {
        torch::NoGradGuard no_grad;
        eval();
        auto tokens = build_tokens(ip, cpu);
        auto attn   = attention->forward(layer_norm(tokens), 0).mean(1);
        auto logits = fc_out(attn);
        auto p      = torch::sigmoid(logits).clamp(1e-6,1-1e-6);
        return p.item<double>() > PRED_THRESHOLD;
    }

    // Update using binary_cross_entropy_with_logits
    void update(uint64_t ip, uint8_t taken, int cpu) {
        train();
        auto tokens = build_tokens(ip, cpu);
        auto attn   = attention->forward(layer_norm(tokens), 0).mean(1);
        auto logits = fc_out(attn);
        auto target = torch::full({1,1}, float(taken));
        auto loss   = torch::binary_cross_entropy_with_logits(logits, target);

        optimizer->zero_grad();
        loss.backward();
        optimizer->step();

        // Update tables & histories
        for (size_t s = 0; s < NUM_LOCAL_HISTORY_SCALES; s++) {
            int len = fixedHistoryLengths[s];
            int tb  = int(std::log2(LOCAL_HISTORY_TABLE_SIZES[s]));
            uint8_t bc = LOCAL_HISTORY_TABLE_BITS[s];
            size_t sz  = LOCAL_HISTORY_TABLE_SIZES[s];
            size_t idx = compute_branch_index(ip, Global_History[cpu], len, tb, s, sz);
            auto &e   = multiLocalHistoryTables[s][idx];
            e = ((e<<1)|taken) & ((1<<bc)-1);
        }
        Global_History[cpu] <<= 1;
        Global_History[cpu].set(0, taken);
        for (int i = PATH_HISTORY_REG_SIZE-1; i > 0; i--)
            Path_History[cpu][i] = Path_History[cpu][i-1];
        Path_History[cpu][0] = ip;

        // ── Persistence: save & load ────────────────────────────────────────
        std::lock_guard<std::mutex> lock(save_mutex);
        ++update_count; ++load_count;

        if (ENABLE_SAVE && update_count % SAVE_INTERVAL == 0) {
            try {
                std::string tmp = MODEL_FILE_PATH + ".tmp";
                torch::serialize::OutputArchive oa;
                save(oa);
                oa.save_to(tmp);
                std::rename(tmp.c_str(), MODEL_FILE_PATH.c_str());
                std::cout << "[Predictor] Saved model at update " << update_count << "\n";
            } catch (const c10::Error &e) {
                std::cerr << "[Predictor] Save error: " << e.what() << "\n";
            }
        }

        if (ENABLE_LOAD && load_count % LOAD_INTERVAL == 0 && fs::exists(MODEL_FILE_PATH)) {
            try {
                torch::serialize::InputArchive ia;
                ia.load_from(MODEL_FILE_PATH);
                load(ia);
                std::cout << "[Predictor] Reloaded model at load " << load_count << "\n";
            } catch (const c10::Error &e) {
                std::cerr << "[Predictor] Load error: " << e.what() << "\n";
            }
        }
    }

    void save(torch::serialize::OutputArchive &archive) const override {
        torch::nn::Module::save(archive);
    }
    void load(torch::serialize::InputArchive &archive) override {
        torch::nn::Module::load(archive);
    }

private:
    std::vector<torch::nn::Linear>    fc_local_combined;
    std::vector<std::vector<uint8_t>> multiLocalHistoryTables;
    std::shared_ptr<MultiHeadSelfAttention> attention;
    torch::nn::LayerNorm              layer_norm{nullptr};
    torch::nn::Linear                 fc_out{nullptr}, fc_global_history{nullptr},
                                      fc_ip_lsb{nullptr}, fc_path_history{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer;
    std::mutex                        save_mutex;
    int                               update_count, load_count;
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
