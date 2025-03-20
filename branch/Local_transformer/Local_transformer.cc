#define TORCH_WARN_OFF
#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <mutex>
#include <array>
#include <numeric>
#include <filesystem>

#include "ooo_cpu.h"

// debug settings
#define SAVE_INTERVAL 10000 // how often the model saves the weights 
#define TOGGLE_LEARNING 1 // whether or not the model will learn
#define SAVE 0  // enable or disable saving the model 
#define LOAD 0  // enable or disable load the model from the weight file 

std::filesystem::path weight_dir;
std::string WEIGHT_FILE;

void setup_weights_directory() {
    std::filesystem::path current_dir = std::filesystem::current_path();
    weight_dir = current_dir / "ChampSim" / "torch_weights";
    std::filesystem::create_directories(weight_dir);
    WEIGHT_FILE = (weight_dir / "hashed_branch_predictor_weights.pt").string();
    std::cout << "Weights will be saved in: " << WEIGHT_FILE << std::endl;
}

// model params
#define LOCAL_HISTORY_TABLES 1
#define HIDDEN_SIZE 64
#define NUM_LAYERS 1
#define LEARNING_RATE 0.0001
#define NUM_HEADS 8
#define PATH_HISTORY_LENGTH 1024
#define GLOBAL_HISTORY_LENGTH 1024

constexpr std::array<uint16_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_SIZES = {4096};
constexpr std::array<uint8_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_BITS = {8};

// Compile-time calculation of LOCAL_HISTORY_BITS
constexpr size_t LOCAL_HISTORY_BITS = [] {
    size_t sum = 0;
    for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++)
        sum += HISTORY_TABLE_BITS[i];
    return sum;}();

// Compute INPUT_SIZE at compile time
constexpr size_t INPUT_SIZE = LOCAL_HISTORY_BITS;

// Mutex for optimizer updates
std::mutex optimizer_mutex;

class TransformerPredictor : public torch::nn::Module {
    private:
        torch::Tensor forward_impl(torch::Tensor input) {
            // Input layer with activation, dropout, and layer normalization
            auto x = torch::relu(fc_in(input));
            x = dropout(x);
            x = layer_norm(x);
            
            // Transformer Encoder: consider using a CLS token if switching to sequence-based input
            auto memory = transformer_encoder(x);
            
            // Pooling strategy: you might experiment with a dedicated token instead of mean pooling
            auto pooled = memory.mean(1);
            auto output = torch::softmax(fc_out(pooled), /*dim=*/1);
            forward_count++;
            return output;
        }
    
        torch::nn::Linear fc_in{nullptr}, fc_out{nullptr};
        torch::nn::TransformerEncoderLayer transformer_encoder_layer{nullptr};
        torch::nn::TransformerEncoder transformer_encoder{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::LayerNorm layer_norm{nullptr};
        std::unique_ptr<torch::optim::Adam> optimizer;
    
        // History Tables and other state variables remain as before
        std::array<std::vector<std::bitset<LOCAL_HISTORY_BITS>>, LOCAL_HISTORY_TABLES> Local_History_tables;
        std::bitset<GLOBAL_HISTORY_LENGTH> Global_History;
        std::bitset<PATH_HISTORY_LENGTH> Path_History;
    
        int update_count;
        int forward_count;
    
    public:
        TransformerPredictor()
            : fc_in(register_module("fc_in", torch::nn::Linear(INPUT_SIZE, HIDDEN_SIZE))),
              transformer_encoder_layer(register_module("encoder_layer", 
                  torch::nn::TransformerEncoderLayer(torch::nn::TransformerEncoderLayerOptions(HIDDEN_SIZE, NUM_HEADS)))),
              transformer_encoder(register_module("encoder", 
                  torch::nn::TransformerEncoder(transformer_encoder_layer, NUM_LAYERS))),
              fc_out(register_module("fc_out", torch::nn::Linear(HIDDEN_SIZE, 2))),
              dropout(register_module("dropout", torch::nn::Dropout(0.1))),
              layer_norm(register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({HIDDEN_SIZE})))),
              optimizer(std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(LEARNING_RATE))),
              update_count(0),
              forward_count(0)
        {
            torch::set_num_threads(1);
        }
    
        void init();
        torch::Tensor predict(uint64_t ip);
        void update(uint64_t ip, uint8_t taken);
    };
    
    void TransformerPredictor::init() {
        for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++)
            Local_History_tables[i] = std::vector<std::bitset<LOCAL_HISTORY_BITS>>(HISTORY_TABLE_SIZES[i]);
    }
    
    torch::Tensor TransformerPredictor::predict(uint64_t ip) {
        std::array<float, INPUT_SIZE> features = {}; // Initialize to 0.0f
        int table_offset = 0;
        for (size_t j = 0; j < LOCAL_HISTORY_TABLES; ++j) {
            size_t index = ip % HISTORY_TABLE_SIZES[j];
            auto& local_history = Local_History_tables[j][index];
            for (size_t i = 0; i < HISTORY_TABLE_BITS.at(j); ++i) {
                features[table_offset + i] = local_history.test(i) ? 1.0f : 0.0f;
            }
            table_offset += HISTORY_TABLE_BITS.at(j);
        }
    
        torch::Tensor input = torch::tensor(std::vector<float>(features.begin(), features.end()),torch::dtype(torch::kFloat32)).view({1, 1, INPUT_SIZE});
        return forward_impl(input);
    }
    
    void TransformerPredictor::update(uint64_t ip, uint8_t taken) {
        // Re-use the forward pass output
        torch::Tensor prediction = this->predict(ip);
        torch::Tensor target = torch::tensor({1.0f - static_cast<float>(taken), static_cast<float>(taken)})
                                    .view({1, 2});
        torch::Tensor loss = torch::binary_cross_entropy(prediction, target.to(torch::kFloat32));
    
        if (TOGGLE_LEARNING) {
            std::lock_guard<std::mutex> lock(optimizer_mutex);
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
        }
    
        // Update local history tables, global and path histories remain unchanged
        for (size_t j = 0; j < LOCAL_HISTORY_TABLES; ++j) {
            size_t index = ip % HISTORY_TABLE_SIZES[j];
            auto& local_history = Local_History_tables[j][index];
            local_history >>= 1;
            local_history.set(HISTORY_TABLE_BITS.at(j) - 1, taken);
        }
    
        for (int i = GLOBAL_HISTORY_LENGTH - 1; i > 0; i--) 
            Global_History[i] = Global_History[i-1];
        Global_History[0] = taken;
    
        for (int i = PATH_HISTORY_LENGTH - 1; i > 0; i--)
            Path_History[i] = Path_History[i-1];
        Path_History[0] = ip & 1;
    }
    

TransformerPredictor transformer_net[NUM_CPUS];

void O3_CPU::initialize_branch_predictor() {
    transformer_net[cpu].init();
}

uint8_t O3_CPU::predict_branch(uint64_t ip) {  
    return transformer_net[cpu].predict(ip).argmax(1).item<int>();
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
    transformer_net[cpu].update(ip, taken);
}
