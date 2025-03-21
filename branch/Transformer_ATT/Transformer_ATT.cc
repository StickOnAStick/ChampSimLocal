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

#define SEQ_LEN 24 // Define sequence length for input features

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
#define LOCAL_HISTORY_TABLES 4
#define HIDDEN_SIZE 70
#define NUM_LAYERS 1
#define LEARNING_RATE 0.0001
#define NUM_HEADS 5
#define PATH_HISTORY_LENGTH 1024
#define GLOBAL_HISTORY_LENGTH 1024

constexpr std::array<uint16_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_SIZES = {16, 64, 256, 1024}; // make sure to update local history tables if you update these
constexpr std::array<uint8_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_BITS = {2, 4, 6, 8};

constexpr size_t LOCAL_HISTORY_BITS = [] {
    size_t sum = 0;
    for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++)
        sum += HISTORY_TABLE_BITS[i];
    return sum;}();

constexpr size_t INPUT_SIZE = 64;

std::mutex optimizer_mutex;

class TransformerPredictor : torch::nn::Module {
private:
    torch::Tensor forward(torch::Tensor input) {
        // Directly pass input to the transformer encoder without any fully connected layers
        auto memory = transformer_encoder(input);
        forward_count++;
        // We no longer have fc_out; you can modify this to output something appropriate
        return torch::softmax(memory.mean(1), 1);  // Softmax over the mean of the attention output
    }

    torch::nn::TransformerEncoderLayer transformer_encoder_layer;
    torch::nn::TransformerEncoder transformer_encoder;
    std::unique_ptr<torch::optim::Adam> optimizer;

    std::array<std::vector<std::bitset<LOCAL_HISTORY_BITS>>, LOCAL_HISTORY_TABLES> Local_History_tables;
    std::bitset<GLOBAL_HISTORY_LENGTH> Global_History;
    std::bitset<PATH_HISTORY_LENGTH> Path_History;

    std::vector<std::array<float, INPUT_SIZE>> input_sequence;

    int update_count;
    int forward_count;

public:
torch::nn::TransformerEncoderLayer transformer_encoder_layer;
torch::nn::TransformerEncoder transformer_encoder;

TransformerPredictor()
    : transformer_encoder_layer(register_module("encoder_layer", 
          torch::nn::TransformerEncoderLayer(torch::nn::TransformerEncoderLayerOptions(HIDDEN_SIZE, NUM_HEADS)
              .dropout(0.1)
              .activation(torch::kReLU))),
      transformer_encoder(register_module("encoder", 
          torch::nn::TransformerEncoder(transformer_encoder_layer, NUM_LAYERS))),
      optimizer(std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(LEARNING_RATE))),
      update_count(0),
      forward_count(0)
{
    torch::set_num_threads(1);
    input_sequence.reserve(SEQ_LEN);
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
    std::array<float, INPUT_SIZE> features = {};
    int table_offset = 0;

    // GLOBAL HISTORY AND IP XOR
    uint64_t transformed_global_history = Global_History.to_ullong() ^ ip;
    for (size_t i = 0; i < INPUT_SIZE; ++i)
        features[i] = ((transformed_global_history >> i) & 1) ? 1.0f : 0.0f;

    input_sequence.push_back(features);
    if (input_sequence.size() > SEQ_LEN)
        input_sequence.erase(input_sequence.begin());

    std::vector<float> flattened_sequence;
    for (const auto& seq : input_sequence) 
        flattened_sequence.insert(flattened_sequence.end(), seq.begin(), seq.end());
    size_t actual_seq_len = input_sequence.size();  // Adjust sequence length dynamically
    torch::Tensor input = torch::tensor(flattened_sequence,
        torch::dtype(torch::kFloat32)).view({1, static_cast<int64_t>(actual_seq_len), INPUT_SIZE});
  

    // Pass input to the forward method, which uses the transformer encoder
    return this->forward(input);
}

void TransformerPredictor::update(uint64_t ip, uint8_t taken) {
    torch::Tensor target = torch::tensor({1.0f - static_cast<float>(taken), static_cast<float>(taken)}).view({1, 2});
    torch::Tensor prediction = this->predict(ip);
    torch::Tensor loss = torch::binary_cross_entropy(prediction, target.to(torch::kFloat32));

    if (TOGGLE_LEARNING) {
        std::lock_guard<std::mutex> lock(optimizer_mutex);
        this->optimizer->zero_grad();
        loss.backward();
        this->optimizer->step();
    }

    // Update local history tables
    for (size_t j = 0; j < LOCAL_HISTORY_TABLES; ++j) {
        size_t index = ip % HISTORY_TABLE_SIZES[j];
        auto& local_history = Local_History_tables[j][index];
        local_history >>= 1;
        local_history.set(HISTORY_TABLE_BITS.at(j) - 1, taken);
    }
}

TransformerPredictor transformer_net[NUM_CPUS];

void O3_CPU::initialize_branch_predictor() {
    transformer_net[cpu].init();
}

uint8_t O3_CPU::predict_branch(uint64_t ip) {  
    // The output will now directly be from the attention mechanism (not from any fc layer)
    return transformer_net[cpu].predict(ip).argmax(1).item<int>();
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
    transformer_net[cpu].update(ip, taken);
}
