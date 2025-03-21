#include <torch/torch.h>
#include <array>
#include <vector>
#include <bitset>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Dummy definitions for NUM_CPUS etc.
#ifndef NUM_CPUS
#define NUM_CPUS 4
#endif

// -------------------------------------------------------------------------
// TAGE configuration parameters

using Tag = uint16_t;
using Index = uint16_t;
using Path = uint64_t;
using History = uint64_t;

#define BIMODAL_TABLE_SIZE 8192
#define MAX_INDEX_BITS 12
#define TAGE_TABLES 4
#define TAGE_TAG_BITS 7
#define TAGE_CONTER_BITS 3
#define TAGE_USEFUL_BITS 2
#define GLOBAL_HISTORY_LENGTH 1024
#define PATH_HISTORY_BUFFER_LENGTH 32
#define TAGE_MIN_LENGTH 5
#define HISTORY_ALPHA 1.5
#define TAGE_RESET_INTERVAL 512000
#define BIMODE_COUNTER_BITS 3

// -------------------------------------------------------------------------
// Traditional TAGE data structures

struct TAGs {
    Tag tag;
    uint8_t useful;
    uint8_t counter;
};

int debug_counter = 0;

// Global tables (we assume these are indexed by CPU)
// Bimodal table T0.
std::array<uint8_t, BIMODAL_TABLE_SIZE> T0;
// TAGE predictor tables: T[table][index] for table = 0 .. TAGE_TABLES-1.
std::array<std::array<TAGs, BIMODAL_TABLE_SIZE>, TAGE_TABLES> T;
// For simplicity, we use fixed-size arrays for the tables here.

int table_history_lengths[TAGE_TABLES];  // history lengths for each TAGE table
uint8_t use_alt_on_na;  // alternate prediction selection flag
uint8_t tage_pred, pred, alt_pred;
int pred_comp, alt_comp;
int STRONG;



// -------------------------------------------------------------------------
// Transformer Aggregator for TAGE
// We will form tokens as follows:
//   Token 0: Bimodal token – BIMODE_COUNTER_BITS bits (from T0).
//   Tokens 1..TAGE_TABLES: Each token is a concatenation of tag, counter, and useful bits from T[table-1].
// We then feed these tokens to a transformer and pool to yield a 2-class output.

#define HIDDEN_SIZE 64

class MultiHeadSelfAttention : public torch::nn::Module {
public:
    MultiHeadSelfAttention()
        : W_q(register_parameter("W_q", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
          W_k(register_parameter("W_k", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
          W_v(register_parameter("W_v", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE}))),
          W_o(register_parameter("W_o", torch::randn({HIDDEN_SIZE, HIDDEN_SIZE})))
    {
        torch::nn::init::xavier_uniform_(W_q);
        torch::nn::init::xavier_uniform_(W_k);
        torch::nn::init::xavier_uniform_(W_v);
        torch::nn::init::xavier_uniform_(W_o);
    }
    torch::Tensor forward(torch::Tensor x, int forward_count) {
        // For simplicity, we do not add positional encoding here.
        auto batch_size = x.size(0);
        auto seq_len = x.size(1);
        auto Q = torch::matmul(x, W_q).view({batch_size, seq_len, 1, HIDDEN_SIZE});
        auto K = torch::matmul(x, W_k).view({batch_size, seq_len, 1, HIDDEN_SIZE});
        auto V = torch::matmul(x, W_v).view({batch_size, seq_len, 1, HIDDEN_SIZE});
        auto scores = torch::matmul(Q, K.transpose(-2,-1)) / std::sqrt((float)HIDDEN_SIZE);
        scores = torch::softmax(scores, -1);
        auto output = torch::matmul(scores, V);
        output = output.view({batch_size, seq_len, HIDDEN_SIZE});
        return torch::matmul(output, W_o);
    }
private:
    torch::Tensor W_q, W_k, W_v, W_o;
};

class TransformerTagePredictor : public torch::nn::Module {
public:
    TransformerTagePredictor()
        : fc_out(register_module("fc_out", torch::nn::Linear(HIDDEN_SIZE, 2))),
          attention(register_module("attention", std::make_shared<MultiHeadSelfAttention>())),
          forward_count(0)
    {
        // Projection for bimodal token: maps BIMODE_COUNTER_BITS -> HIDDEN_SIZE.
        fc_bimodal = register_module("fc_bimodal", torch::nn::Linear(BIMODE_COUNTER_BITS, HIDDEN_SIZE));
        // For each TAGE table, projection from (TAGE_TAG_BITS + TAGE_CONTER_BITS + TAGE_USEFUL_BITS) -> HIDDEN_SIZE.
        for (size_t i = 0; i < TAGE_TABLES; i++) {
            auto proj = torch::nn::Linear(TAGE_TAG_BITS + TAGE_CONTER_BITS + TAGE_USEFUL_BITS, HIDDEN_SIZE);
            fc_table.push_back(register_module("fc_table_" + std::to_string(i), proj));
        }
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({HIDDEN_SIZE})));
    }
    
    // Build tokens from T0 and T tables.
    torch::Tensor build_tokens(uint64_t ip) {
        std::vector<torch::Tensor> tokens;
        // Token 0: Bimodal token.
        Index bimodal_index = get_bimodal_index(ip);
        torch::Tensor token_bimodal = torch::zeros({1, BIMODE_COUNTER_BITS}, torch::dtype(torch::kFloat32));
        for (size_t i = 0; i < BIMODE_COUNTER_BITS; i++) {
            token_bimodal[0][i] = ((T0[bimodal_index] >> i) & 1) ? 1.0f : 0.0f;
        }
        tokens.push_back(fc_bimodal(token_bimodal));
        
        // Tokens for each TAGE table.
        for (size_t t = 0; t < TAGE_TABLES; t++) {
            Index idx = get_predictor_index(ip, t + 1);
            Tag tag = T[t][idx].tag;
            uint8_t counter = T[t][idx].counter;
            uint8_t useful = T[t][idx].useful;
            torch::Tensor token_table = torch::zeros({1, TAGE_TAG_BITS + TAGE_CONTER_BITS + TAGE_USEFUL_BITS}, torch::dtype(torch::kFloat32));
            // Fill tag bits.
            for (size_t i = 0; i < TAGE_TAG_BITS; i++) {
                token_table[0][i] = ((tag >> i) & 1) ? 1.0f : 0.0f;
            }
            // Fill counter bits.
            for (size_t i = 0; i < TAGE_CONTER_BITS; i++) {
                token_table[0][TAGE_TAG_BITS + i] = ((counter >> i) & 1) ? 1.0f : 0.0f;
            }
            // Fill useful bits.
            for (size_t i = 0; i < TAGE_USEFUL_BITS; i++) {
                token_table[0][TAGE_TAG_BITS + TAGE_CONTER_BITS + i] = ((useful >> i) & 1) ? 1.0f : 0.0f;
            }
            tokens.push_back(fc_table[t](token_table));
        }
        // Stack tokens along sequence dimension: shape [1, NUM_TOKENS, HIDDEN_SIZE]
        torch::Tensor tokens_tensor = torch::cat(tokens, 0).unsqueeze(0);
        return tokens_tensor;
    }
    
    torch::Tensor predict(uint64_t ip) {
        torch::Tensor tokens = build_tokens(ip);
        torch::Tensor x = layer_norm(tokens);
        x = attention->forward(x, forward_count);
        auto pooled = x.mean(1); // simple average pooling over tokens.
        forward_count++;
        return torch::sigmoid(fc_out(pooled));
    }
    
private:
    torch::nn::Linear fc_bimodal{nullptr};
    std::vector<torch::nn::Linear> fc_table;  // one projection per TAGE table
    std::shared_ptr<MultiHeadSelfAttention> attention;
    torch::nn::LayerNorm layer_norm{nullptr};
    torch::nn::Linear fc_out{nullptr};
    int forward_count;
};

// Global transformer aggregator (one per CPU).
std::array<TransformerTagePredictor, NUM_CPUS> tage_transformer;

// -------------------------------------------------------------------------
// Modified TAGE Functions using the transformer aggregator

class Tage {
public:
    // For simplicity, we assume the member variables (T0, T, table_history_lengths, etc.)
    // are global variables defined above.
    int cpu; // the CPU id for this predictor instance
    
    void init() {
        std::srand(std::time(0));
        use_alt_on_na = 8;
        tage_pred = 0;
        // Initialize bimodal table.
        for (size_t i = 0; i < BIMODAL_TABLE_SIZE; i++) {
            T0[i] = (1 << (BIMODE_COUNTER_BITS - 1)); // weakly taken
        }
        // Initialize TAGE tables.
        for (size_t i = 0; i < TAGE_TABLES; i++) {
            for (size_t j = 0; j < (1 << MAX_INDEX_BITS); j++) {
                T[i][j].counter = (1 << (TAGE_CONTER_BITS - 1));
                T[i][j].useful = 0;
                T[i][j].tag = 0;
            }
        }
        double power = 1.0;
        for (int i = 0; i < TAGE_TABLES; i++) {
            table_history_lengths[i] = int(TAGE_MIN_LENGTH * power + 0.5);
            power *= HISTORY_ALPHA;
            std::cout << "TAGE table " << i << " history length: " << table_history_lengths[i] << std::endl;
        }
        // Initialize transformer aggregator for this CPU.
        tage_transformer[cpu] = TransformerTagePredictor();
    }
    
    uint8_t predict(uint64_t ip) {
        // Use the transformer aggregator to produce a two-class output.
        torch::Tensor out = tage_transformer[cpu].predict(ip);
        // Here we assume the second component is the taken probability.
        return (out[0][1].item<float>() > 0.5) ? 1 : 0;
    }
    Index get_bimodal_index(uint64_t ip) {
        return ip & (BIMODAL_TABLE_SIZE - 1);
    }
    
    Index get_predictor_index(uint64_t ip, int table) {
        Path path_history_hash = get_path_history_hash(table);
    
        // Hash of global history
        History global_histor_hash = get_compressed_global_history(table_history_lengths[table-1],MAX_INDEX_BITS);
    
        // Really complex hashing function 
        return(global_histor_hash ^ ip ^ (ip >> (abs(MAX_INDEX_BITS-table)+1)) ^ path_history_hash) & ((1 << MAX_INDEX_BITS)-1);
    }
    
    
    void ctr_update(uint8_t &ctr, int cond, int low, int high) {
        if (cond && ctr < high)
            ctr++;
        else if (!cond && ctr > low)
            ctr--;
    }
    
    History get_compressed_global_history(int inSize, int outSize){
        History compressed_history = 0;
        History temporary_history = 0;
        int compressed_history_length = outSize;
        for (int i = 0; i < inSize; i++)
        {
            if (i % compressed_history_length == 0)
            {
                compressed_history ^= temporary_history;
                temporary_history = 0;
            }
            temporary_history = (temporary_history << 1) | GLOBAL_HISTORY[i];
        }
        compressed_history ^= temporary_history;
        return compressed_history;
    }

        Path Tage::get_path_history_hash(int table)
    {
        Path A = 0; 
        Path size = table_history_lengths[table-1] > 16 ? 16 : table_history_lengths[table-1];
        for (int i = PATH_HISTORY_BUFFER_LENGTH -1; i>= 0; i--)
            A = (A << 1) | PATH_HISTORY[i];
        A = A & ((1 << size)-1);
        
        Path A1, A2;
        A1 = A & ((1 << MAX_INDEX_BITS)-1);
        A2 = A >> MAX_INDEX_BITS;

        // Use hashign from CBP-4 L-tage submission
        A2 = ((A2 << table) & ((1 << MAX_INDEX_BITS) - 1)) + (A2 >> abs(MAX_INDEX_BITS - table));
        A = A1 ^ A2;
        A = ((A << table) & ((1 << MAX_INDEX_BITS) - 1)) + (A >> abs(MAX_INDEX_BITS - table));
        return(A);
    }

    void update(uint64_t ip, uint8_t taken){
        if (pred_comp > 0)
        {
            struct TAGs *entry = &T[pred_comp-1][get_predictor_index(ip,pred_comp)];
            uint8_t useful = entry->useful;
            if(!STRONG)
            {
                if (pred != alt_pred)
                    // std::cout << "ENTRY UPDATED" << std::endl;
                    ctr_update(use_alt_on_na, !(pred = taken), 0, 15);
            }
    
            if (alt_comp > 0) 
            {
                struct TAGs *alt_entry = &T[alt_comp-1][get_predictor_index(ip,alt_comp)];
                if(useful == 0)
                    // std::cout << "ENTRY UPDATED" << std::endl;
                    ctr_update(alt_entry->counter,taken,0,((1 << TAGE_CONTER_BITS) -1));
            }
    
            else 
            {
                Index index = get_bimodal_index(ip);
                if (useful == 0)
                    ctr_update(T0[index],taken,0,((1 << BIMODE_COUNTER_BITS)-1));
            }
    
            if(pred != alt_pred)
            {
                if (pred == taken)
                {
                    if (entry->useful < ((1 << TAGE_USEFUL_BITS)-1))
                        entry->useful++;
                }
                else 
                {
                    if(use_alt_on_na < 8)
                    {
                        if (entry->useful > 0)
                            entry->useful--;
                    }
                }
            }
            ctr_update(entry->counter, taken, 0, ((1 <<TAGE_CONTER_BITS) -1));
        }
    
        else
        {
            Index index = get_bimodal_index(ip);
            ctr_update(T0[index], taken, 0, ((1 << TAGE_CONTER_BITS)-1));
        }
        if (tage_pred != taken)
        {
            long random = static_cast <long> (rand()) / static_cast <long> (RAND_MAX);
            random = random & ((1 << (TAGE_TABLES - pred_comp -1))-1);
            int start_component = pred_comp + 1;
    
            if(random & 1)
            {
                start_component++;
                if(random & 2)
                    start_component++;
            }
            int isFree = 0;
            for (int i = pred_comp + 1; i <= TAGE_TABLES; i++)
            {
                struct TAGs *entry_new = &T[i-1][get_predictor_index(ip,i)];
                if(entry_new->useful == 0)
                    isFree = 1;
            }
            if (!isFree && start_component <= TAGE_TABLES)
                T[start_component-1][get_predictor_index(ip,start_component)].useful = 0;
    
            for(int i = start_component; i <= TAGE_TABLES; i++)
            {
                struct TAGs *entry_new = &T[i-1][get_predictor_index(ip,i)];
                if(entry_new->useful == 0)
                {
                    entry_new->tag = get_tag(ip,i);
                    entry_new->counter = (1 << (TAGE_CONTER_BITS - 1));
                    break;
                }
            }
        }
        for(int i = GLOBAL_HISTORY_LENGTH - 1; i > 0; i--) 
            GLOBAL_HISTORY[i] = GLOBAL_HISTORY[i-1];
        GLOBAL_HISTORY[0] = taken;
        
        for(int i = PATH_HISTORY_BUFFER_LENGTH - 1; i > 0; i--)
            PATH_HISTORY[i] = PATH_HISTORY[i-1];
        PATH_HISTORY[0] = ip & 1;
    
        count++;
        if (count % TAGE_RESET_INTERVAL == 0)
        {
            count = 0;
            for(int i = 0; i < TAGE_TABLES; i++)
            {
                for (int j =0; j < (1 << MAX_INDEX_BITS); j++)
                    T[i][j].useful >>= 1;
            }
        }
    }

// Global TAGE predictors (one per CPU)
std::array<Tage, NUM_CPUS> tage_predictor;


// These functions are called from outside.
namespace O3_CPU {
    void initialize_branch_predictor() {
        // For each CPU, initialize the TAGE predictor and its transformer aggregator.
        for (int cpu = 0; cpu < NUM_CPUS; cpu++) {
            tage_predictor[cpu].cpu = cpu;
            tage_predictor[cpu].init();
        }
    }
    uint8_t predict_branch(uint64_t ip) {
        // For this example, use CPU 0.
        return tage_predictor[0].predict(ip);
    }
    void last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
        tage_predictor[0].update(ip, taken);
    }
}
