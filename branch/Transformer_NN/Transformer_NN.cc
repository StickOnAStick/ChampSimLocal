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
#define SAVE 1  // enable or disable saving the model 
#define LOAD 1  // enable or disable load the model from the weight file 

std::filesystem::path weight_dir;
std::string WEIGHT_FILE;

void setup_weights_directory() {
    std::filesystem::path current_dir = std::filesystem::current_path();
    weight_dir = current_dir / "ChampSim" / "torch_weights";
    std::filesystem::create_directories(weight_dir);
    WEIGHT_FILE = (weight_dir / "branch_predictor_weights.pt").string();
    std::cout << "Weights will be saved in: " << WEIGHT_FILE << std::endl;
}


// model params 
#define LOCAL_HISTORY_TABLES 3
#define HIDDEN_SIZE 32
#define NUM_LAYERS 1
#define LEARNING_RATE 0.0001
#define NUM_HEADS 8
#define PATH_HISTORY_BITS 8
#define PATH_HISTORY_TABLE_SIZE 4
constexpr size_t GLOBAL_HISTORY_BITS = 12; 
constexpr int PATH_TABLE_SIZE = PATH_HISTORY_BITS * PATH_HISTORY_TABLE_SIZE;
constexpr std::array<uint16_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_SIZES = {64,256,2048}; // make sure to update local history tables if you update these 
constexpr std::array<uint8_t, LOCAL_HISTORY_TABLES> HISTORY_TABLE_BITS = {4,8,10};

// Compile-time calculation of LOCAL_HISTORY_BITS i think
constexpr size_t LOCAL_HISTORY_BITS = [] {
    size_t sum = 0; 
    for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++) 
        sum += HISTORY_TABLE_BITS[i]; 
    return sum;}();

// Compute INPUT_SIZE at compile time
constexpr size_t INPUT_SIZE = LOCAL_HISTORY_BITS + GLOBAL_HISTORY_BITS + PATH_TABLE_SIZE + 1;

// Mutex for optimizer updates
std::mutex optimizer_mutex;

class TransformerPredictor : torch::nn::Module {
private:
    torch::Tensor forward(torch::Tensor input) {
        input = torch::relu(fc_in(input));
        auto memory = transformer_encoder(input);
        input = transformer_decoder(input, memory);
        forward_count++;
        return torch::softmax(fc_out(input.mean(1)), 1);
    }

    torch::nn::Linear fc_in, fc_out;
    torch::nn::TransformerEncoderLayer transformer_encoder_layer;
    torch::nn::TransformerEncoder transformer_encoder;
    torch::nn::TransformerDecoderLayer transformer_decoder_layer;
    torch::nn::TransformerDecoder transformer_decoder;
    std::unique_ptr<torch::optim::Adam> optimizer;

    std::array<std::vector<std::bitset<LOCAL_HISTORY_BITS>>, LOCAL_HISTORY_TABLES> Local_History_tables;  // Local History Tables 
    std::bitset<GLOBAL_HISTORY_BITS> Global_History; // Global branch history
    std::bitset<PATH_TABLE_SIZE> Path_history_table; // Path History Table

    int update_count;
    int forward_count;

public: 
    TransformerPredictor()
        : fc_in(register_module("fc_in", torch::nn::Linear(INPUT_SIZE, HIDDEN_SIZE))),
          transformer_encoder_layer(register_module("encoder_layer", 
              torch::nn::TransformerEncoderLayer(torch::nn::TransformerEncoderLayerOptions(HIDDEN_SIZE, NUM_HEADS)))),
          transformer_encoder(register_module("encoder", 
              torch::nn::TransformerEncoder(transformer_encoder_layer, NUM_LAYERS))),
          transformer_decoder_layer(register_module("decoder_layer", 
              torch::nn::TransformerDecoderLayer(torch::nn::TransformerDecoderLayerOptions(HIDDEN_SIZE, NUM_HEADS)))),
          transformer_decoder(register_module("decoder", 
              torch::nn::TransformerDecoder(transformer_decoder_layer, NUM_LAYERS))),
          fc_out(register_module("fc_out", torch::nn::Linear(HIDDEN_SIZE, 2))),
          optimizer(std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(LEARNING_RATE))),
          update_count(0),
        forward_count(0)
        {torch::set_num_threads(1);} // Limit PyTorch threading

    void init();  // initialise the member variables
    torch::Tensor predict(uint64_t ip);  // return the tensor from forward
    void update(uint64_t ip, uint8_t taken);  // updates the state of tage
    void save_model(torch::nn::Module &model);
    void load_model(torch::nn::Module &model);
};


void TransformerPredictor::save_model(torch::nn::Module &model) {
    if (TOGGLE_LEARNING && SAVE) {
        std::filesystem::path weight_path(WEIGHT_FILE);
        std::filesystem::create_directories(weight_path.parent_path()); // Ensure directory exists
        
        torch::serialize::OutputArchive archive;
        model.save(archive);
        archive.save_to(WEIGHT_FILE);
    }
}

void TransformerPredictor::load_model(torch::nn::Module &model) {
    if (LOAD) {
        if (std::ifstream(WEIGHT_FILE)) {
            torch::serialize::InputArchive archive;
            archive.load_from(WEIGHT_FILE);
            model.load(archive);
        } else 
            save_model(model);
    }
}

void TransformerPredictor::init(){
    setup_weights_directory();
    load_model(*this);
    for (size_t i = 0; i < LOCAL_HISTORY_TABLES; i++)
        Local_History_tables[i] = std::vector<std::bitset<LOCAL_HISTORY_BITS>>(HISTORY_TABLE_SIZES[i]);
}

torch::Tensor TransformerPredictor::predict(uint64_t ip)
{
    std::array<float, INPUT_SIZE> features = {}; // Initialize to 0.0f
    // save every X predictions 
    if (forward_count % SAVE_INTERVAL == 0) 
        save_model(*this);

    
    // GLOBAL HISTORY AND IP XOR
    uint64_t transformed_global_history = Global_History.to_ullong() ^ ip;
    for (size_t i = 0; i < GLOBAL_HISTORY_BITS; ++i)
        features[i] = ((transformed_global_history >> i) & 1) ? 1.0f : 0.0f;

    // LOCAL HISTORY TABLES     
    int table_offset = 0;
    for (size_t j = 0; j < LOCAL_HISTORY_TABLES; ++j) {
        size_t index = ip % HISTORY_TABLE_SIZES[j];
        auto& local_history = Local_History_tables[j][index];
        for (size_t i = 0; i < HISTORY_TABLE_BITS.at(j); ++i)
            features[GLOBAL_HISTORY_BITS + table_offset + i] = local_history.test(i) ? 1.0f : 0.0f;

        table_offset += HISTORY_TABLE_BITS.at(j);
    }

    // Path History Table 
    for (size_t i = 0; i < PATH_TABLE_SIZE-1; i++) 
        features[table_offset + GLOBAL_HISTORY_BITS + i] = Path_history_table[i];

    torch::Tensor input = torch::tensor(std::vector<float>(features.begin(), features.end()), torch::dtype(torch::kFloat32)).view({1, 1, INPUT_SIZE});
    return this->forward(input);
}


void TransformerPredictor::update(uint64_t ip, uint8_t taken){

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

    // Update path history table 
    std::bitset<64> ipbitset(ip);
    Path_history_table <<= PATH_HISTORY_BITS;
    for (int i = 0; i < PATH_HISTORY_BITS ; i++) 
        Path_history_table[i] = ipbitset[i]; 
    Global_History >>= 1;
    Global_History.set(GLOBAL_HISTORY_BITS - 1, taken);
}
    
TransformerPredictor transformer_net[NUM_CPUS];

void O3_CPU::initialize_branch_predictor() 
{
    transformer_net[cpu].init();
}

uint8_t O3_CPU::predict_branch(uint64_t ip)
{  
    return transformer_net[cpu].predict(ip).argmax(1).item<int>();
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
    transformer_net[cpu].update(ip,taken);
}
