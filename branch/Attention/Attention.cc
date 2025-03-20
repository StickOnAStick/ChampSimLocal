#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <bitset>
#include <mutex>
#include <array>
#include <cmath>
#include "ooo_cpu.h"
// -------------------------------------------------------------------------
// Configuration Parameters and Constants

// Transformer model hyperparameters.
#define HIDDEN_SIZE 64
#define NUM_HEADS 8
#define LEARNING_RATE 0.0001
#define DEBUG_INTERVAL 10000
// Branch predictor bit widths.
#define IP_BITS 24             // Bits representing the instruction pointer.
#define PATH_HISTORY_LENGTH 1  // Number of branch target tokens.
#define LOCAL_HISTORY_TABLES 1 // Number of local history tables.

// Global history configuration.
constexpr uint8_t GLOBAL_HISTORY_BITS = 24;
constexpr uint8_t PATH_BITS = 16;

// Local history tables configuration.
constexpr std::array<uint16_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_SIZES = {4096};
constexpr std::array<uint8_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_BITS = {8};

// Multi-scale history configuration (TAGE-inspired).
#define TAGE_MIN_LENGTH 24
#define HISTORY_ALPHA 1.6
constexpr size_t NUM_MULTISCALE_HISTORY = 4;

// Compute multi-scale history lengths using a geometric progression.
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

// Total number of tokens (features) for the transformer input.
// Tokens: IP, one per local history table, global history, multi-scale histories, and path history.
constexpr size_t NUM_TOKENS = 1 + LOCAL_HISTORY_TABLES + 1 + NUM_MULTISCALE_HISTORY + PATH_HISTORY_LENGTH;

// Mutex to protect optimizer updates across threads.
std::mutex optimizer_mutex;

// -------------------------------------------------------------------------
// Data Structures for History Storage

// Local history entry stored as an 8-bit bitset.
using HistoryEntry = std::bitset<8>;
// Local history tables: one per table, each table holds a vector of HistoryEntry per CPU.
std::array<std::vector<std::vector<HistoryEntry>>, LOCAL_HISTORY_TABLES> Local_History_Tables;

// Global branch history: one per CPU.
std::vector<std::bitset<GLOBAL_HISTORY_BITS>> Global_History(NUM_CPUS);

// Multi-scale history stored in a 128-bit integer.
using MultiHistoryType = unsigned __int128;
std::vector<std::array<MultiHistoryType, NUM_MULTISCALE_HISTORY>> MultiScale_History(NUM_CPUS);

// Path history: stores the last branch target (or part of it) per CPU.
std::vector<std::vector<uint64_t>> Path_History(NUM_CPUS, std::vector<uint64_t>(PATH_HISTORY_LENGTH, 0));

// -------------------------------------------------------------------------
// Positional Encoding Function for the Transformer

// Computes a sinusoidal positional encoding matrix of size [seq_len, hidden_size].
torch::Tensor positional_encoding(int seq_len, int hidden_size) {
    auto pos = torch::arange(0, seq_len, torch::kFloat32).unsqueeze(1);
    auto div_term = torch::exp(torch::arange(0, hidden_size, 2, torch::kFloat32) * (-std::log(10000.0) / hidden_size));
    auto pe = torch::zeros({seq_len, hidden_size});
    // Apply sine to even indices in the positional encoding.
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::nullopt, 2)},torch::sin(pos * div_term));
    // Apply cosine to odd indices in the positional encoding.
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::nullopt, 2)},torch::cos(pos * div_term));
    return pe;
}

// -------------------------------------------------------------------------
// Multi-Head Self-Attention Module

class MultiHeadSelfAttention : public torch::nn::Module {
public:
    // Constructor: Initialize projection matrices and dropout.
    MultiHeadSelfAttention()
      : W_q(register_parameter("W_q", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
        W_k(register_parameter("W_k", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
        W_v(register_parameter("W_v", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
        W_o(register_parameter("W_o", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))) {
        // Use Xavier initialization for better convergence.
        torch::nn::init::xavier_uniform_(W_q);
        torch::nn::init::xavier_uniform_(W_k);
        torch::nn::init::xavier_uniform_(W_v);
        torch::nn::init::xavier_uniform_(W_o);
        dropout = register_module("dropout", torch::nn::Dropout(0.1));
    }
    
    // Forward pass of self-attention.
    torch::Tensor forward(torch::Tensor x, int forward_count) {
        // Add positional encoding to the token embeddings.
        x = x + positional_encoding(x.size(1), HIDDEN_SIZE);
        auto batch_size = x.size(0), seq_len = x.size(1);
        // Compute queries, keys, and values.
        auto Q = torch::matmul(x, W_q).view({batch_size, seq_len, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS}).permute({0, 2, 1, 3});
        auto K = torch::matmul(x, W_k).view({batch_size, seq_len, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS}).permute({0, 2, 1, 3});
        auto V = torch::matmul(x, W_v).view({batch_size, seq_len, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS}).permute({0, 2, 1, 3});
        // Compute scaled dot-product attention scores.
        auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt((float)(HIDDEN_SIZE / NUM_HEADS));
        scores = torch::softmax(scores, -1);
        // Optionally print attention scores for debugging.
        if (forward_count % DEBUG_INTERVAL == 0)
            print_attention_table(scores);
        // Apply dropout to the scores.
        scores = dropout->forward(scores);
        // Compute weighted values.
        auto output = torch::matmul(scores, V);
        output = output.permute({0, 2, 1, 3}).contiguous()
                     .view({batch_size, seq_len, HIDDEN_SIZE});
        // Final linear projection.
        return torch::matmul(output, W_o);
    }
    
private:
    torch::Tensor W_q, W_k, W_v, W_o;
    torch::nn::Dropout dropout{nullptr};
    
    // Print attention scores in a table format for debugging.
    void print_attention_table(const torch::Tensor &scores) {
        std::vector<std::string> token_labels = {" IP"};
        for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++)
            token_labels.push_back(" LocalHTR" + std::to_string(i));
        token_labels.push_back(" GlobalHTR");
        for (size_t i = 0; i < NUM_MULTISCALE_HISTORY; i++)
            token_labels.push_back(" MultiGHR" + std::to_string(i));
        for (size_t i = 0; i < PATH_HISTORY_LENGTH; i++)
            token_labels.push_back(" PathHist" + std::to_string(i));
        
        auto scores_cpu = scores.to(torch::kCPU);
        int num_heads = scores_cpu.size(1);
        int seq_len = scores_cpu.size(3);
        const int col_width = 10;
        for (int head = 0; head < num_heads; ++head) {
            std::cout << "Head " << head << " Attention:" << std::endl;
            std::ostringstream header;
            header << std::setw(col_width) << "Token";
            for (int j = 0; j < seq_len; ++j) {
                std::string col_label = (j < token_labels.size()) ? token_labels[j] : "T" + std::to_string(j);
                header << std::setw(col_width) << col_label;
            }
            std::cout << header.str() << std::endl;
            for (int i = 0; i < seq_len; ++i) {
                std::ostringstream row;
                std::string row_label = (i < token_labels.size()) ? token_labels[i] : "T" + std::to_string(i);
                row << std::setw(col_width) << row_label;
                auto head_scores = scores_cpu[0][head];
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
// Helper Function to Compress Multi-Scale History

// Folds multi-history bits into a target bit-width.
uint8_t compress_multi_history(MultiHistoryType history, size_t original_bits, size_t target_bits) {
    while (original_bits > target_bits) {
        history = (history & ((1ULL << target_bits) - 1)) ^ (history >> target_bits);
        original_bits = target_bits;
    }
    return static_cast<uint8_t>(history & ((1ULL << target_bits) - 1));
}

// -------------------------------------------------------------------------
// Attention-Only Predictor Module

class AttentionOnlyPredictor : public torch::nn::Module {
public:
    // Constructor: Initialize all projection layers, attention module, and optimizer.
    AttentionOnlyPredictor()
      : fc_out(register_module("fc_out", torch::nn::Linear(HIDDEN_SIZE, 1))),
        attention(register_module("attention", std::make_shared<MultiHeadSelfAttention>())),
        optimizer(std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions(LEARNING_RATE))),
        forward_count(0) {
        // Projection for IP token.
        fc_ip = register_module("fc_ip", torch::nn::Linear(IP_BITS, HIDDEN_SIZE));

        // Projection for local history tokens.
        for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++)
            fc_local.push_back(register_module("fc_local_" + std::to_string(i),torch::nn::Linear(HISTORY_TABLE_BITS[i], HIDDEN_SIZE)));
        
        // Projection for global history token.
        fc_global = register_module("fc_global", torch::nn::Linear(GLOBAL_HISTORY_BITS, HIDDEN_SIZE));
        
        // Projections for each multi-scale history token.
        for (size_t i = 0; i < NUM_MULTISCALE_HISTORY; i++)
            fc_multi.push_back(register_module("fc_multi_" + std::to_string(i),torch::nn::Linear(GLOBAL_HISTORY_BITS, HIDDEN_SIZE)));
        
            // Projection for path history token.
        fc_path = register_module("fc_path", torch::nn::Linear(PATH_BITS, HIDDEN_SIZE));
        // Layer normalization for the token sequence.
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({HIDDEN_SIZE})));
    }
    
    // Build the token sequence from various history sources.
    // Tokens include: IP, local histories, global history, multi-scale histories, and path history.
    torch::Tensor build_tokens(uint64_t ip, int cpu) {
        std::vector<torch::Tensor> tokens;
        
        // Token for IP: convert each bit into a float value.
        torch::Tensor token_ip = torch::zeros({1, IP_BITS}, torch::kFloat32);
        for (size_t i = 0; i < IP_BITS; ++i)
            token_ip[0][i] = ((ip >> i) & 1) ? 1.0f : 0.0f;
        tokens.push_back(fc_ip(token_ip));
        
        // Tokens for local history tables.
        for (size_t t = 0; t < LOCAL_HISTORY_TABLES; t++) {
            size_t index = ip % HISTORY_TABLE_SIZES[t];
            const std::vector<HistoryEntry>& table = Local_History_Tables[t][cpu];
            const HistoryEntry& bits = table[index];
            torch::Tensor token_hist = torch::zeros({1, HISTORY_TABLE_BITS[t]}, torch::kFloat32);
            for (size_t i = 0; i < HISTORY_TABLE_BITS[t]; i++)
                token_hist[0][i] = bits.test(i) ? 1.0f : 0.0f;
            tokens.push_back(fc_local[t](token_hist));
        }
        
        // Token for global history.
        auto& glob = Global_History[cpu];
        torch::Tensor token_global = torch::zeros({1, GLOBAL_HISTORY_BITS}, torch::kFloat32);
        for (size_t i = 0; i < GLOBAL_HISTORY_BITS; i++)
            token_global[0][i] = glob.test(i) ? 1.0f : 0.0f;
        tokens.push_back(fc_global(token_global));
        
        // Tokens for multi-scale histories.
        for (size_t m = 0; m < NUM_MULTISCALE_HISTORY; m++) {
            MultiHistoryType history = MultiScale_History[cpu][m];
            // Compress the history to GLOBAL_HISTORY_BITS.
            uint8_t compressed = compress_multi_history(history, multi_scale_history_bits[m], GLOBAL_HISTORY_BITS);
            torch::Tensor token_multi = torch::zeros({1, GLOBAL_HISTORY_BITS}, torch::kFloat32);
            for (size_t i = 0; i < GLOBAL_HISTORY_BITS; i++)
                token_multi[0][i] = ((compressed >> i) & 1) ? 1.0f : 0.0f;
            tokens.push_back(fc_multi[m](token_multi));
        }
        
        // Token for path history.
        for (size_t j = 0; j < PATH_HISTORY_LENGTH; j++) {
            uint64_t path = Path_History[cpu][j];
            torch::Tensor token_path = torch::zeros({1, PATH_BITS}, torch::kFloat32);
            for (size_t i = 0; i < PATH_BITS; i++)
                token_path[0][i] = ((path >> i) & 1) ? 1.0f : 0.0f;
            tokens.push_back(fc_path(token_path));
        }
        
        // Concatenate all tokens along the sequence dimension.
        torch::Tensor tokens_tensor = torch::cat(tokens, 0).unsqueeze(0);
        
        // Print diagnostic information once.
        static bool printed = false;
        if (!printed) {
            printed = true;
            std::cout << "Token tensor shape: " << tokens_tensor.sizes() << std::endl;
            int total_params = 0;
            int ip_params = fc_ip->weight.numel() + fc_ip->bias.numel();
            std::cout << "IP token parameters: " << ip_params << std::endl;
            total_params += ip_params;
            for (size_t t = 0; t < fc_local.size(); t++) {
                int local_params = fc_local[t]->weight.numel() + fc_local[t]->bias.numel();
                std::cout << "Local token " << t << " parameters: " << local_params << std::endl;
                total_params += local_params;
            }
            int global_params = fc_global->weight.numel() + fc_global->bias.numel();
            std::cout << "Global token parameters: " << global_params << std::endl;
            total_params += global_params;
            for (size_t m = 0; m < fc_multi.size(); m++) {
                int multi_params = fc_multi[m]->weight.numel() + fc_multi[m]->bias.numel();
                std::cout << "Multi-scale token " << m << " parameters: " << multi_params << std::endl;
                total_params += multi_params;
            }
            int path_params = fc_path->weight.numel() + fc_path->bias.numel();
            std::cout << "Path token parameters: " << path_params << std::endl;
            total_params += path_params;
            std::cout << "Total token projection parameters: " << total_params << std::endl;
        }
        return tokens_tensor;
    }
    
    // Predict the branch outcome.
    // Builds tokens, normalizes them, applies self-attention, pools the output, and produces a probability.
    torch::Tensor predict(uint64_t ip, int cpu) {
        auto tokens = build_tokens(ip, cpu);
        auto x = layer_norm(tokens);
        x = attention->forward(x, forward_count);
        forward_count++;
        return torch::sigmoid(fc_out(x.mean(1)));
    }
    
    // Update the model parameters based on the branch outcome.
    // Also updates the various history storage tables.
    void update(uint64_t ip, uint8_t taken, uint8_t branch_type, int cpu, uint64_t branch_target) {
        // Build tokens and run forward pass.
        auto tokens = build_tokens(ip, cpu);
        auto x = layer_norm(tokens);
        x = attention->forward(x, forward_count);
        forward_count++;
        auto prediction = torch::sigmoid(fc_out(x.mean(1)));
        
        // Create target tensor.
        auto target = torch::full({1, 1}, static_cast<float>(taken));
        
        // Focal loss parameters.
        const float gamma = 2.0;
        const float alpha = 0.25;
        
        // Compute standard binary cross entropy loss.
        auto bce_loss = torch::binary_cross_entropy(prediction, target, /*weight=*/{}, torch::Reduction::None);
        
        // Compute focal loss factor
        auto pt = torch::where(target == 1.0, prediction, 1 - prediction);
        auto focal_factor = alpha * torch::pow(1 - pt, gamma);
        auto loss = (focal_factor * bce_loss).mean();
        
        // Optimize: zero gradients, backward, clip gradients, and step.
        {
            std::lock_guard<std::mutex> lock(optimizer_mutex);
            optimizer->zero_grad();
            loss.backward();
            // Clip gradients to improve training stability.
            torch::nn::utils::clip_grad_norm_(this->parameters(), 5.0);
            optimizer->step();
        }
        
        // Update local history tables.
        for (size_t t = 0; t < LOCAL_HISTORY_TABLES; t++) {
            size_t index = ip % HISTORY_TABLE_SIZES[t];
            Local_History_Tables[t][cpu][index] >>= 1;
            Local_History_Tables[t][cpu][index].set(HISTORY_TABLE_BITS[t] - 1, taken);
        }
        // Update global history.
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
    }
    
    
private:
    // Projection layers for each token type.
    torch::nn::Linear fc_ip{nullptr};
    std::vector<torch::nn::Linear> fc_local;
    torch::nn::Linear fc_global{nullptr};
    std::vector<torch::nn::Linear> fc_multi;
    torch::nn::Linear fc_path{nullptr};
    
    // Self-attention module and layer normalization.
    std::shared_ptr<MultiHeadSelfAttention> attention;
    torch::nn::LayerNorm layer_norm{nullptr};
    
    // Final classification layer.
    torch::nn::Linear fc_out{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer;
    int forward_count;
};

// -------------------------------------------------------------------------
// Global Instance of the Predictor and Interface Functions

// One predictor per CPU.
AttentionOnlyPredictor transformer_net[NUM_CPUS];

// Initialize branch predictor history structures.
void O3_CPU::initialize_branch_predictor() {
    for (size_t t = 0; t < LOCAL_HISTORY_TABLES; t++) {
        Local_History_Tables[t].resize(NUM_CPUS);
        for (size_t cpu = 0; cpu < NUM_CPUS; cpu++)
            Local_History_Tables[t][cpu] = std::vector<HistoryEntry>(HISTORY_TABLE_SIZES[t]);
    }
    MultiScale_History.assign(NUM_CPUS, std::array<MultiHistoryType, NUM_MULTISCALE_HISTORY>{0});
}

// Predict branch outcome: return 1 if prediction probability > 0.5, else 0.
uint8_t O3_CPU::predict_branch(uint64_t ip) {
    return transformer_net[cpu].predict(ip, cpu).item<float>() > 0.5 ? 1 : 0;
}

// Update predictor based on the branch outcome.
void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
    transformer_net[cpu].update(ip, taken, branch_type, cpu, branch_target);
}
