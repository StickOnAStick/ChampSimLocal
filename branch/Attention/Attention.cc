#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <bitset>
#include <mutex>
#include <array>
#include <cmath>
#include "ooo_cpu.h"

// -------------------------------------------------------------------------
// Model and history configuration

#define HIDDEN_SIZE 70
#define NUM_HEADS 5

// Field bit widths.
#define IP_BITS 1 // Number of bits to represent the hashed IP token.
#define DEBUG_INTERVAL 10000
#define LEARNING_RATE 0.0005

#define PATH_HISTORY_LENGTH 4
#define PATH_BITS 16

// Global history register is still stored as a long register (256 bits).
constexpr size_t GLOBAL_HISTORY_REGISTER_BITS = 1024;
constexpr uint8_t COMPRESSED_HISTORY_BITS = 16;

// New token bit widths.
#define PREV_BRANCH_TYPE_BITS 3
#define BRANCH_TARGET_TOKEN_BITS 32

#define LOCAL_HISTORY_TABLES 4
constexpr std::array<uint16_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_SIZES = {16, 128, 1024, 8196};
constexpr std::array<uint8_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_BITS = {8,8,8,8};

#define TAGE_MIN_LENGTH 3
#define HISTORY_ALPHA 1.5
constexpr size_t NUM_MULTISCALE_HISTORY = 12; 

// Compute the multi-scale history lengths at runtime using a geometric progression.
std::vector<size_t> compute_multi_scale_history_bits(size_t num_levels, double min_length, double alpha) {
    std::vector<size_t> lengths(num_levels);
    double power = 1.0;
    for (size_t i = 0; i < num_levels; i++) {
        lengths[i] = static_cast<size_t>(min_length * power + 0.5);
        power *= alpha;
    }
    std::cout << "Multi-scale history lengths:";
    for (auto l : lengths)
        std::cout << " " << l;
    std::cout << std::endl;
    return lengths;
}
std::vector<size_t> multi_scale_history_bits = compute_multi_scale_history_bits(NUM_MULTISCALE_HISTORY, TAGE_MIN_LENGTH, HISTORY_ALPHA);

// Updated NUM_TOKENS: 
//  1 (hashed IP) + LOCAL_HISTORY_TABLES + NUM_MULTISCALE_HISTORY + PATH_HISTORY_LENGTH + 2 (prev branch type and branch target)
constexpr size_t NUM_TOKENS = 1 + LOCAL_HISTORY_TABLES + NUM_MULTISCALE_HISTORY + PATH_HISTORY_LENGTH + 2;

std::mutex optimizer_mutex;

// -------------------------------------------------------------------------
// History storage

// For local history, we use a std::bitset with capacity 8.
using HistoryEntry = std::bitset<16>;
// Local history tables: for each table, for each CPU, a vector (size defined by HISTORY_TABLE_SIZES).
std::array<std::vector<std::vector<HistoryEntry>>, LOCAL_HISTORY_TABLES> Local_History_Tables;

// Global history register: now 256 bits per CPU.
std::vector<std::bitset<GLOBAL_HISTORY_REGISTER_BITS>> Global_History(NUM_CPUS);

// Multi-scale histories: using a type to hold up to 128 bits.
using MultiHistoryType = unsigned __int128;
std::vector<std::array<MultiHistoryType, NUM_MULTISCALE_HISTORY>> MultiScale_History(NUM_CPUS);

// Path history: for each CPU, store the last PATH_HISTORY_LENGTH branch targets.
std::vector<std::vector<uint64_t>> Path_History(
    NUM_CPUS, std::vector<uint64_t>(PATH_HISTORY_LENGTH, 0));

// New global state to store previous branch type and branch target for each CPU.
std::vector<uint8_t> Prev_Branch_Type(NUM_CPUS, 0);
std::vector<uint8_t> Prev_Branch_Target(NUM_CPUS, 0);

// -------------------------------------------------------------------------
// Helper function: Hash IP with Global History
// This function compresses the 256-bit global history register into 64 bits by XOR-folding
// and then combines it with the instruction pointer using XOR.
uint64_t hash_ip_with_global(uint64_t ip, const std::bitset<GLOBAL_HISTORY_REGISTER_BITS>& global) {
    uint64_t compressed = 0;
    // Fold the global history register into 64 bits.
    for (size_t i = 0; i < GLOBAL_HISTORY_REGISTER_BITS; i += 64) {
        uint64_t chunk = 0;
        for (size_t j = 0; j < 64 && (i + j) < GLOBAL_HISTORY_REGISTER_BITS; j++) {
            if (global.test(i + j))
                chunk |= (1ULL << j);
        }
        compressed ^= chunk;
    }
    return ip ^ compressed;
}

// -------------------------------------------------------------------------
// Positional Encoding

torch::Tensor positional_encoding(int seq_len, int hidden_size) {
    torch::Tensor pos = torch::arange(0, seq_len, torch::kFloat32).unsqueeze(1);
    torch::Tensor div_term = torch::exp(torch::arange(0, hidden_size, 2, torch::kFloat32) *
                                          (-std::log(10000.0) / hidden_size));
    torch::Tensor pe = torch::zeros({seq_len, hidden_size});
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::nullopt, 2)},
                    torch::sin(pos * div_term));
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::nullopt, 2)},
                    torch::cos(pos * div_term));
    return pe;
}

// -------------------------------------------------------------------------
// Multi-head Self-Attention Module

class MultiHeadSelfAttention : public torch::nn::Module {
public:
    MultiHeadSelfAttention()
        : W_q(register_parameter("W_q", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
          W_k(register_parameter("W_k", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
          W_v(register_parameter("W_v", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
          W_o(register_parameter("W_o", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE})))
    {
        at::set_num_threads(1);  // Ensure each instance runs in a separate thread
        torch::nn::init::xavier_uniform_(W_q);
        torch::nn::init::xavier_uniform_(W_k);
        torch::nn::init::xavier_uniform_(W_v);
        torch::nn::init::xavier_uniform_(W_o);
        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(0.1)));
    }
    
    torch::Tensor forward(torch::Tensor x, int forward_count) {
        // x = x + positional_encoding(x.size(1), HIDDEN_SIZE);
        auto batch_size = x.size(0);
        auto seq_len = x.size(1);
        auto Q = torch::matmul(x, W_q).view({batch_size, seq_len, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS}).permute({0, 2, 1, 3});
        auto K = torch::matmul(x, W_k).view({batch_size, seq_len, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS}).permute({0, 2, 1, 3});
        auto V = torch::matmul(x, W_v).view({batch_size, seq_len, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS}).permute({0, 2, 1, 3});
        auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt((float)(HIDDEN_SIZE / NUM_HEADS));
        scores = torch::softmax(scores, -1);

        if (forward_count % DEBUG_INTERVAL == 0) 
            print_attention_table(scores);
        // Apply dropout to attention scores.
        scores = dropout->forward(scores);
        auto output = torch::matmul(scores, V);
        output = output.permute({0, 2, 1, 3}).contiguous().view({batch_size, seq_len, HIDDEN_SIZE});
        return torch::matmul(output, W_o);
    }

private:
    torch::Tensor W_q, W_k, W_v, W_o;
    torch::nn::Dropout dropout{nullptr};

    // Helper function to print attention scores as a nicely aligned table.
    void print_attention_table(const torch::Tensor &scores) {
        // Build token labels.
        std::vector<std::string> token_labels;
        token_labels.push_back(" HashedIP");
        for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++)
            token_labels.push_back(" LocalHTR" + std::to_string(i));
        for (size_t i = 0; i < NUM_MULTISCALE_HISTORY; i++)
            token_labels.push_back(" MultiGHR" + std::to_string(i));
        for (size_t i = 0; i < PATH_HISTORY_LENGTH; i++)
            token_labels.push_back(" PathHist" + std::to_string(i));
        token_labels.push_back(" PrevBType");
        token_labels.push_back(" BrTarget");

        // Convert scores tensor to CPU and assume batch size 1.
        auto scores_cpu = scores.to(torch::kCPU);
        int num_heads = scores_cpu.size(1);
        int seq_len = scores_cpu.size(3); // scores shape: [batch, heads, seq_len, seq_len]
        const int col_width = 10;
        
        for (int head = 0; head < num_heads; ++head) {
            std::cout << "Head " << head << " Attention:" << std::endl;
            // Print header row.
            std::ostringstream header;
            header << std::setw(col_width) << "Token";
            for (int j = 0; j < seq_len; ++j) {
                std::string col_label = (j < token_labels.size()) ? token_labels[j] : "T" + std::to_string(j);
                header << std::setw(col_width) << col_label;
            }
            std::cout << header.str() << std::endl;
            
            // Print each row.
            for (int i = 0; i < seq_len; ++i) {
                std::ostringstream row;
                std::string row_label = (i < token_labels.size()) ? token_labels[i] : "T" + std::to_string(i);
                row << std::setw(col_width) << row_label;
                auto head_scores = scores_cpu[0][head]; // shape: [seq_len, seq_len]
                for (int j = 0; j < seq_len; ++j) {
                    float score = head_scores[i][j].item<float>();
                    row << std::setw(col_width) << std::fixed << std::setprecision(4) << score;
                }
                std::cout << row.str() << std::endl;
            }
            std::cout << std::endl;
        }
    }
};

// -------------------------------------------------------------------------
// Helper function to compress multi-scale history
// This function folds the original multi-history bits into COMPRESSED_HISTORY_BITS bits.
uint8_t compress_multi_history(MultiHistoryType history, size_t original_bits, size_t target_bits) {
    while (original_bits > target_bits) {
        // Fold the upper bits into the lower bits.
        history = (history & ((1ULL << target_bits) - 1)) ^ (history >> target_bits);
        original_bits = target_bits; // After one fold, we assume it is target_bits wide.
    }
    return static_cast<uint8_t>(history & ((1ULL << target_bits) - 1));
}

// We define COMPRESSED_HISTORY_BITS to be the width used when compressing multi-scale histories.


// -------------------------------------------------------------------------
// Attention-Only Predictor Module

class AttentionOnlyPredictor : public torch::nn::Module {
public:
    AttentionOnlyPredictor()
        : fc_out(register_module("fc_out", torch::nn::Linear(HIDDEN_SIZE, 1))),
          attention(register_module("attention", std::make_shared<MultiHeadSelfAttention>())),
          optimizer(std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions(LEARNING_RATE))),
          forward_count(0)
    {
        at::set_num_threads(1);
        // Projection for hashed IP.
        fc_hash_ip = register_module("fc_hash_ip", torch::nn::Linear(IP_BITS, HIDDEN_SIZE));
        // One projection per local history table.
        for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++) {
            auto proj = torch::nn::Linear(HISTORY_TABLE_BITS[i], HIDDEN_SIZE);
            fc_local.push_back(register_module("fc_local_" + std::to_string(i), proj));
        }
        // Projections for multi-scale histories: compress each to COMPRESSED_HISTORY_BITS.
        for (size_t i = 0; i < NUM_MULTISCALE_HISTORY; i++) {
            auto proj = torch::nn::Linear(COMPRESSED_HISTORY_BITS, HIDDEN_SIZE);
            fc_multi.push_back(register_module("fc_multi_" + std::to_string(i), proj));
        }
        // Projection for path history.
        fc_path = register_module("fc_path", torch::nn::Linear(PATH_BITS, HIDDEN_SIZE));
        // New projections for the additional tokens.
        fc_prev_branch_type = register_module("fc_prev_branch_type", torch::nn::Linear(PREV_BRANCH_TYPE_BITS, HIDDEN_SIZE));
        fc_branch_target = register_module("fc_branch_target", torch::nn::Linear(BRANCH_TARGET_TOKEN_BITS, HIDDEN_SIZE));
        
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({HIDDEN_SIZE})));
    }

    // Build the token sequence.
    // New token ordering:
    //   Token 0: Hashed IP token (hash(ip, global history register)) [IP_BITS]
    //   Tokens 1..LOCAL_HISTORY_TABLES: one token per local history table.
    //   Next NUM_MULTISCALE_HISTORY tokens: one token per multi-scale history (compressed to COMPRESSED_HISTORY_BITS).
    //   Next PATH_HISTORY_LENGTH tokens: path history tokens.
    //   Next token: previous branch type (8 bits)
    //   Last token: branch target (8 bits)
    torch::Tensor build_tokens(uint64_t ip, int cpu) {
        std::vector<torch::Tensor> tokens;
        
        // Token for hashed IP: compute hash(IP, global history register).
        uint64_t hashed = hash_ip_with_global(ip, Global_History[cpu]);
        torch::Tensor token_hash_ip = torch::zeros({1, IP_BITS}, torch::dtype(torch::kFloat32));
        for (size_t i = 0; i < IP_BITS; ++i)
            token_hash_ip[0][i] = ((hashed >> i) & 1) ? 1.0f : 0.0f;
        tokens.push_back(fc_hash_ip(token_hash_ip));
        
        // Tokens for each local history table.
        for (size_t t = 0; t < LOCAL_HISTORY_TABLES; t++) {
            size_t index = ip % HISTORY_TABLE_SIZES[t];
            const std::vector<HistoryEntry>& table = Local_History_Tables[t][cpu];
            const HistoryEntry& bits = table[index];
            torch::Tensor token_hist = torch::zeros({1, HISTORY_TABLE_BITS[t]}, torch::dtype(torch::kFloat32));
            for (size_t i = 0; i < HISTORY_TABLE_BITS[t]; i++)
                token_hist[0][i] = bits.test(i) ? 1.0f : 0.0f;
            tokens.push_back(fc_local[t](token_hist));
        }
        
        // Tokens for multi-scale histories.
        for (size_t m = 0; m < NUM_MULTISCALE_HISTORY; m++) {
            MultiHistoryType history = MultiScale_History[cpu][m];
            // Compress the multi-scale history down to COMPRESSED_HISTORY_BITS.
            uint8_t compressed = compress_multi_history(history, multi_scale_history_bits[m], COMPRESSED_HISTORY_BITS);
            torch::Tensor token_multi = torch::zeros({1, COMPRESSED_HISTORY_BITS}, torch::dtype(torch::kFloat32));
            for (size_t i = 0; i < COMPRESSED_HISTORY_BITS; i++) {
                token_multi[0][i] = ((compressed >> i) & 1) ? 1.0f : 0.0f;
            }
            tokens.push_back(fc_multi[m](token_multi));
        }
        
        // Tokens for path history.
        for (size_t j = 0; j < PATH_HISTORY_LENGTH; j++) {
            uint64_t path = Path_History[cpu][j];
            torch::Tensor token_path = torch::zeros({1, PATH_BITS}, torch::dtype(torch::kFloat32));
            for (size_t i = 0; i < PATH_BITS; i++)
                token_path[0][i] = ((path >> i) & 1) ? 1.0f : 0.0f;
            tokens.push_back(fc_path(token_path));
        }
        
        // New token for previous branch type.
        {
            uint8_t prev_type = Prev_Branch_Type[cpu];
            torch::Tensor token_prev_type = torch::zeros({1, PREV_BRANCH_TYPE_BITS}, torch::dtype(torch::kFloat32));
            for (size_t i = 0; i < PREV_BRANCH_TYPE_BITS; i++)
                token_prev_type[0][i] = ((prev_type >> i) & 1) ? 1.0f : 0.0f;
            tokens.push_back(fc_prev_branch_type(token_prev_type));
        }
        
        // New token for branch target.
        {
            uint8_t br_target = Prev_Branch_Target[cpu];
            torch::Tensor token_br_target = torch::zeros({1, BRANCH_TARGET_TOKEN_BITS}, torch::dtype(torch::kFloat32));
            for (size_t i = 0; i < BRANCH_TARGET_TOKEN_BITS; i++)
                token_br_target[0][i] = ((br_target >> i) & 1) ? 1.0f : 0.0f;
            tokens.push_back(fc_branch_target(token_br_target));
        }
        
        // Stack tokens along the sequence dimension: resulting shape [1, NUM_TOKENS, HIDDEN_SIZE]
        torch::Tensor tokens_tensor = torch::cat(tokens, 0).unsqueeze(0);
        // Print diagnostics on the first call.
        static bool printed = false;
        if (!printed) {
            printed = true;
            std::cout << "Token tensor shape: " << tokens_tensor.sizes() << std::endl;
            int total_params = 0;
            // Parameters from fc_hash_ip.
            int hash_ip_params = fc_hash_ip->weight.numel() + fc_hash_ip->bias.numel();
            std::cout << "Hashed IP token parameters: " << hash_ip_params << std::endl;
            total_params += hash_ip_params;
            // Parameters from each local history token projection.
            for (size_t t = 0; t < fc_local.size(); t++) {
                int local_params = fc_local[t]->weight.numel() + fc_local[t]->bias.numel();
                std::cout << "Local token " << t << " parameters: " << local_params << std::endl;
                total_params += local_params;
            }
            // Parameters from multi-scale history projections.
            for (size_t m = 0; m < fc_multi.size(); m++) {
                int multi_params = fc_multi[m]->weight.numel() + fc_multi[m]->bias.numel();
                std::cout << "Multi-scale token " << m << " parameters: " << multi_params << std::endl;
                total_params += multi_params;
            }
            // Parameters from path history projection.
            int path_params = fc_path->weight.numel() + fc_path->bias.numel();
            std::cout << "Path token parameters: " << path_params << std::endl;
            total_params += path_params;
            // Parameters from new tokens.
            int prev_type_params = fc_prev_branch_type->weight.numel() + fc_prev_branch_type->bias.numel();
            std::cout << "Prev branch type token parameters: " << prev_type_params << std::endl;
            total_params += prev_type_params;
            int br_target_params = fc_branch_target->weight.numel() + fc_branch_target->bias.numel();
            std::cout << "Branch target token parameters: " << br_target_params << std::endl;
            total_params += br_target_params;
            
            std::cout << "Total token projection parameters: " << total_params << std::endl;
        }
        
        return tokens_tensor;
    }
    
    // Predict: build tokens, apply layer norm and attention, pool, and produce prediction.
    torch::Tensor predict(uint64_t ip, int cpu) {
        torch::Tensor tokens = build_tokens(ip, cpu);
        torch::Tensor x = layer_norm(tokens);
        x = attention->forward(x, forward_count);
        auto pooled = x.mean(1);
        forward_count++;
        return torch::sigmoid(fc_out(pooled));
    }

    // Update: compute loss and update model parameters and history.
    void update(uint64_t ip, uint8_t taken, uint8_t branch_type, int cpu, uint64_t branch_target) {
        torch::Tensor tokens = build_tokens(ip, cpu);
        torch::Tensor x = layer_norm(tokens);
        x = attention->forward(x, forward_count);
        forward_count++;
        torch::Tensor prediction = torch::sigmoid(fc_out(x.mean(1)));
        torch::Tensor target = torch::full({1, 1}, static_cast<float>(taken));
        torch::Tensor loss = torch::binary_cross_entropy(prediction, target);

        {
            std::lock_guard<std::mutex> lock(optimizer_mutex);
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
        }
        
        // Update local history tables.
        for (size_t t = 0; t < LOCAL_HISTORY_TABLES; t++) {
            size_t index = ip % HISTORY_TABLE_SIZES[t];
            Local_History_Tables[t][cpu][index] >>= 1;
            Local_History_Tables[t][cpu][index].set(HISTORY_TABLE_BITS[t] - 1, taken);
        }
        
        // Update global history register.
        Global_History[cpu] <<= 1;
        Global_History[cpu].set(0, taken);
        
        // Update multi-scale histories.
        for (size_t m = 0; m < NUM_MULTISCALE_HISTORY; m++) {
            MultiScale_History[cpu][m] <<= 1;
            MultiScale_History[cpu][m] |= taken;
        }
        
        // Update path history.
        for (int j = PATH_HISTORY_LENGTH - 1; j > 0; j--)
            Path_History[cpu][j] = Path_History[cpu][j - 1];
        Path_History[cpu][0] = branch_target & ((1ULL << PATH_BITS) - 1);
        
        // Update the new state: store the current branch type and branch target
        // so that on the next prediction, these values become the "previous" ones.
        Prev_Branch_Type[cpu] = branch_type;
        Prev_Branch_Target[cpu] = branch_target & ((1ULL << BRANCH_TARGET_TOKEN_BITS) - 1);
    }

private:
    // Projection layers.
    torch::nn::Linear fc_hash_ip{nullptr}; // Projection for hashed IP token.
    std::vector<torch::nn::Linear> fc_local;  // One projection per local history table.
    std::vector<torch::nn::Linear> fc_multi;    // One projection per multi-scale history level.
    torch::nn::Linear fc_path{nullptr};
    // New projection layers for the extra tokens.
    torch::nn::Linear fc_prev_branch_type{nullptr};
    torch::nn::Linear fc_branch_target{nullptr};

    std::shared_ptr<MultiHeadSelfAttention> attention;
    torch::nn::LayerNorm layer_norm{nullptr};
    torch::nn::Linear fc_out{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer;
    int forward_count;
};

AttentionOnlyPredictor transformer_net[NUM_CPUS];

// -------------------------------------------------------------------------
// Initialization and Interface Functions

void O3_CPU::initialize_branch_predictor() {
    // Initialize local history tables for each table and each CPU.
    for (size_t t = 0; t < LOCAL_HISTORY_TABLES; t++) {
        Local_History_Tables[t].resize(NUM_CPUS);
        for (size_t cpu = 0; cpu < NUM_CPUS; cpu++) {
            Local_History_Tables[t][cpu] = std::vector<HistoryEntry>(HISTORY_TABLE_SIZES[t]);
        }
    }
    // Initialize multi-scale histories to 0 for each CPU.
    MultiScale_History.assign(NUM_CPUS, std::array<MultiHistoryType, NUM_MULTISCALE_HISTORY>{0});
}

uint8_t O3_CPU::predict_branch(uint64_t ip) {
    return transformer_net[cpu].predict(ip, cpu).item<float>() > 0.5 ? 1 : 0;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
    transformer_net[cpu].update(ip, taken, branch_type, cpu, branch_target);
}
