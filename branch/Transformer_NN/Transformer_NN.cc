#include <torch/torch.h>
#include <vector>
#include "ooo_cpu.h"

struct TransformerPredictor : torch::nn::Module {
    TransformerPredictor(int input_size, int hidden_size, int num_layers)
        : fc_in(register_module("fc_in", torch::nn::Linear(input_size, hidden_size))),
          transformer_encoder_layer(register_module("encoder_layer", 
              torch::nn::TransformerEncoderLayer(torch::nn::TransformerEncoderLayerOptions(hidden_size, 4)))),
          transformer_encoder(register_module("encoder", 
              torch::nn::TransformerEncoder(transformer_encoder_layer, num_layers))),
          fc_out(register_module("fc_out", torch::nn::Linear(hidden_size, 1))),
          optimizer(std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(0.0001))) {}

    torch::Tensor forward(torch::Tensor input) {
        input = torch::relu(fc_in(input));  // Initial linear projection
        input = transformer_encoder(input); // Pass through Transformer
        return torch::sigmoid(fc_out(input.mean(1))); // Mean pool over sequence and classify
    }

    torch::nn::Linear fc_in, fc_out;
    torch::nn::TransformerEncoderLayer transformer_encoder_layer;
    torch::nn::TransformerEncoder transformer_encoder;
    std::unique_ptr<torch::optim::Adam> optimizer;
};

constexpr std::size_t HISTORY_LENGTH = 64;
std::bitset<HISTORY_LENGTH> Global_History;
TransformerPredictor transformer_net(HISTORY_LENGTH + 1, 16, 1);  // 16 hidden units, 2 transformer layers

void O3_CPU::initialize_branch_predictor() {}

uint8_t O3_CPU::predict_branch(uint64_t ip) {
    std::array<float, HISTORY_LENGTH + 1> features;
    features[0] = static_cast<float>(ip) / static_cast<float>(UINT64_MAX);
    for (size_t i = 0; i < HISTORY_LENGTH; ++i)
        features[i + 1] = Global_History[i] ? 1.0f : 0.0f;

    torch::Tensor input = torch::from_blob(features.data(), {1, 1, HISTORY_LENGTH + 1}).clone();
    return transformer_net.forward(input).item<float>() > 0.5 ? 1 : 0;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
    std::array<float, HISTORY_LENGTH + 1> features;
    features[0] = static_cast<float>(ip) / static_cast<float>(UINT64_MAX);
    for (size_t i = 0; i < HISTORY_LENGTH; ++i)
        features[i + 1] = Global_History[i] ? 1.0f : 0.0f;

    torch::Tensor input = torch::from_blob(features.data(), {1, 1, HISTORY_LENGTH + 1}).clone();
    torch::Tensor target = torch::tensor(static_cast<float>(taken)).view({1, 1});

    torch::Tensor prediction = transformer_net.forward(input);
    torch::Tensor loss = torch::binary_cross_entropy(prediction, target);

    transformer_net.optimizer->zero_grad();
    loss.backward();
    transformer_net.optimizer->step();

    Global_History >>= 1;
    Global_History[HISTORY_LENGTH - 1] = taken;
}
