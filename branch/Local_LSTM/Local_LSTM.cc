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
#define LOCAL_HISTORY_TABLES 3
#define HIDDEN_SIZE 32
#define NUM_LAYERS 1
#define LEARNING_RATE 0.0001
#define PATH_HISTORY_LENGTH 1024
#define GLOBAL_HISTORY_LENGTH 1024

constexpr std::array<uint16_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_SIZES = {16, 64, 256}; // make sure to update local history tables if you update these 
constexpr std::array<uint8_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_BITS = {2, 4, 8};

// Compile-time calculation of LOCAL_HISTORY_BITS i think
constexpr size_t LOCAL_HISTORY_BITS = [] {
    size_t sum = 0; 
    for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++) 
        sum += HISTORY_TABLE_BITS[i]; 
    return sum;}();

// Compute INPUT_SIZE at compile time
constexpr size_t INPUT_SIZE = LOCAL_HISTORY_BITS;

// Mutex for optimizer updates
std::mutex optimizer_mutex;

class LSTMPredictor : torch::nn::Module {
private:
torch::Tensor forward(torch::Tensor input) {
    input = torch::relu(fc_in(input));
    auto rnn_output = lstm(input); // This returns a tuple
    forward_count++;
    
    // Use std::get to access the first element of the tuple
    return torch::softmax(fc_out(std::get<0>(rnn_output).mean(1)), 1); // Access the first tensor in the tuple
}



    torch::nn::Linear fc_in, fc_out;
    torch::nn::LSTM lstm;
    std::unique_ptr<torch::optim::Adam> optimizer;

    std::array<std::vector<std::bitset<LOCAL_HISTORY_BITS>>, LOCAL_HISTORY_TABLES> Local_History_tables;  // Local History Tables 
    std::bitset<GLOBAL_HISTORY_LENGTH> Global_History; // Global branch history
    std::bitset<PATH_HISTORY_LENGTH> Path_History; // Path History Table

    int update_count;
    int forward_count;

public: 
    LSTMPredictor()
        : fc_in(register_module("fc_in", torch::nn::Linear(INPUT_SIZE, HIDDEN_SIZE))),
          lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(HIDDEN_SIZE, HIDDEN_SIZE).num_layers(NUM_LAYERS)))),
          fc_out(register_module("fc_out", torch::nn::Linear(HIDDEN_SIZE, 2))),
          optimizer(std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(LEARNING_RATE))),
          update_count(0),
          forward_count(0)
        {torch::set_num_threads(1);} // Limit PyTorch threading

    void init();  // initialise the member variables
    torch::Tensor predict(uint64_t ip);  // return the tensor from forward
    void update(uint64_t ip, uint8_t taken);  // updates the state of tage
};

void LSTMPredictor::init(){
    for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++)
        Local_History_tables[i] = std::vector<std::bitset<LOCAL_HISTORY_BITS>>(HISTORY_TABLE_SIZES[i]);
}


torch::Tensor LSTMPredictor::predict(uint64_t ip) {
    std::array<float, INPUT_SIZE> features = {}; // Initialize to 0.0f

    // LOCAL HISTORY TABLES     
    int table_offset = 0;
    for (size_t j = 0; j < LOCAL_HISTORY_TABLES; ++j) {
        size_t index = ip % HISTORY_TABLE_SIZES[j];
        auto& local_history = Local_History_tables[j][index];
        for (size_t i = 0; i < HISTORY_TABLE_BITS.at(j); ++i)
            features[i] = local_history.test(i) ? 1.0f : 0.0f;
        table_offset += HISTORY_TABLE_BITS.at(j);
    }

    torch::Tensor input = torch::tensor(std::vector<float>(features.begin(), features.end()), torch::dtype(torch::kFloat32)).view({1, 1, INPUT_SIZE});
    return this->forward(input);
}


void LSTMPredictor::update(uint64_t ip, uint8_t taken){

    // Get loss
    torch::Tensor target = torch::tensor({1.0f - static_cast<float>(taken), static_cast<float>(taken)}).view({1, 2});
    torch::Tensor prediction = this->predict(ip);
    torch::Tensor loss = torch::binary_cross_entropy(prediction, target.to(torch::kFloat32));

    // learn
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

    for(int i = GLOBAL_HISTORY_LENGTH - 1; i > 0; i--) 
        Global_History[i] = Global_History[i-1];
    Global_History[0] = taken;
    
    for(int i = PATH_HISTORY_LENGTH - 1; i > 0; i--)
        Path_History[i] = Path_History[i-1];
    Path_History[0] = ip & 1;
}

LSTMPredictor lstm_net[NUM_CPUS];

void O3_CPU::initialize_branch_predictor() 
{
    lstm_net[cpu].init();
}

uint8_t O3_CPU::predict_branch(uint64_t ip)
{  
    return lstm_net[cpu].predict(ip).argmax(1).item<int>();
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
    lstm_net[cpu].update(ip,taken);
}
