#define TORCH_WARN_OFF
#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <mutex>
#include "ooo_cpu.h"

#define LOCAL_HISTORY_SIZE 8192  
#define HIDDEN_SIZE 16
#define NUM_LAYERS 1
#define LEARNING_RATE 0.0001
#define NUM_HEADS 4

constexpr size_t LOCAL_HISTORY_BITS = 8;  
constexpr size_t GLOBAL_HISTORY_BITS = 64; 
constexpr size_t INPUT_SIZE = LOCAL_HISTORY_BITS + GLOBAL_HISTORY_BITS + 1;

// Bimodal table for local branch history
std::vector<std::bitset<LOCAL_HISTORY_BITS>> Local_History(LOCAL_HISTORY_SIZE);
// Global branch history
std::bitset<GLOBAL_HISTORY_BITS> Global_History;

// Mutex for optimizer updates
std::mutex optimizer_mutex;

// Log file for async logging
std::ofstream log_file("debug.log", std::ios::app);

void log_debug(const std::string &msg) {
    std::lock_guard<std::mutex> lock(optimizer_mutex);
    log_file << msg << std::endl;
}

// this printout could be entirely wrong I really have no idea how to calculate this 
void print_model_size() {
    size_t fc_in_params = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE;
    size_t fc_out_params = HIDDEN_SIZE * 2 + 2;
    size_t transformer_encoder_params = NUM_LAYERS * (4 * (HIDDEN_SIZE * HIDDEN_SIZE / NUM_HEADS) + 2 * HIDDEN_SIZE * HIDDEN_SIZE);
    size_t transformer_decoder_params = NUM_LAYERS * (4 * (HIDDEN_SIZE * HIDDEN_SIZE / NUM_HEADS) + 2 * HIDDEN_SIZE * HIDDEN_SIZE);

    size_t total_params = fc_in_params + fc_out_params + transformer_encoder_params + transformer_decoder_params;
    double model_size_kb = (total_params * sizeof(float)) / 1024.0;
    double local_history_size_kb = (LOCAL_HISTORY_SIZE * LOCAL_HISTORY_BITS) / 1024.0;

    std::cout << "Model size: " << model_size_kb << " KB (" << total_params << " parameters)" << std::endl;
    std::cout << "Local history size: " << local_history_size_kb << " KB (" << LOCAL_HISTORY_SIZE 
              << " entries, " << LOCAL_HISTORY_BITS << " bits per entry)" << std::endl;
}

struct TransformerPredictor : torch::nn::Module {
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
          update_count(0), forward_count(0) {
        
        torch::set_num_threads(1); // Limit PyTorch threading
    }

    torch::Tensor forward(torch::Tensor input) {
        input = torch::relu(fc_in(input));
        auto memory = transformer_encoder(input);
        input = transformer_decoder(input, memory);
        forward_count++;

        if (forward_count % 10000 == 0) {
            log_debug("[DEBUG] Forward count: " + std::to_string(forward_count));
        }

        return torch::softmax(fc_out(input.mean(1)), 1);
    }

    torch::nn::Linear fc_in, fc_out;
    torch::nn::TransformerEncoderLayer transformer_encoder_layer;
    torch::nn::TransformerEncoder transformer_encoder;
    torch::nn::TransformerDecoderLayer transformer_decoder_layer;
    torch::nn::TransformerDecoder transformer_decoder;
    std::unique_ptr<torch::optim::Adam> optimizer;
    int update_count;
    int forward_count;
};

TransformerPredictor transformer_net;

void O3_CPU::initialize_branch_predictor() {
    print_model_size();
}

uint8_t O3_CPU::predict_branch(uint64_t ip) {
  size_t index = ip % LOCAL_HISTORY_SIZE;
  std::bitset<LOCAL_HISTORY_BITS>& local_history = Local_History[index];

  // XOR global history with the IP
  uint64_t transformed_global_history = Global_History.to_ullong() ^ ip;

  std::array<float, INPUT_SIZE> features;
  features[0] = static_cast<float>(ip) / static_cast<float>(UINT64_MAX);

  for (size_t i = 0; i < LOCAL_HISTORY_BITS; ++i)
      features[i + 1] = local_history[i] ? 1.0f : 0.0f;

  for (size_t i = 0; i < GLOBAL_HISTORY_BITS; ++i)
      features[LOCAL_HISTORY_BITS + 1 + i] = (transformed_global_history >> i) & 1 ? 1.0f : 0.0f;

  // Convert std::array to std::vector before creating tensor
  torch::Tensor input = torch::tensor(std::vector<float>(features.begin(), features.end()), 
                                      torch::dtype(torch::kFloat32)).view({1, 1, INPUT_SIZE});

  return transformer_net.forward(input).argmax(1).item<int>();
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
  size_t index = ip % LOCAL_HISTORY_SIZE;
  std::bitset<LOCAL_HISTORY_BITS>& local_history = Local_History[index];

  // XOR global history with the IP
  uint64_t transformed_global_history = Global_History.to_ullong() ^ ip;

  std::array<float, INPUT_SIZE> features;
  features[0] = static_cast<float>(ip) / static_cast<float>(UINT64_MAX);

  for (size_t i = 0; i < LOCAL_HISTORY_BITS; ++i)
      features[i + 1] = local_history[i] ? 1.0f : 0.0f;

  for (size_t i = 0; i < GLOBAL_HISTORY_BITS; ++i)
      features[LOCAL_HISTORY_BITS + 1 + i] = (transformed_global_history >> i) & 1 ? 1.0f : 0.0f;

  // Convert std::array to std::vector before creating tensor
  torch::Tensor input = torch::tensor(std::vector<float>(features.begin(), features.end()), 
                                      torch::dtype(torch::kFloat32)).view({1, 1, INPUT_SIZE});

  torch::Tensor target = torch::tensor({1.0f - static_cast<float>(taken), static_cast<float>(taken)}).view({1, 2});
  torch::Tensor prediction = transformer_net.forward(input);
  torch::Tensor loss = torch::binary_cross_entropy(prediction, target);

  // Mutex to protect optimizer in parallel tests
  {
      std::lock_guard<std::mutex> lock(optimizer_mutex);
      transformer_net.optimizer->zero_grad();
      loss.backward();
      transformer_net.optimizer->step();
  }

  // Update histories
  local_history >>= 1;
  local_history[LOCAL_HISTORY_BITS - 1] = taken;
  Global_History >>= 1;
  Global_History[GLOBAL_HISTORY_BITS - 1] = taken;
}