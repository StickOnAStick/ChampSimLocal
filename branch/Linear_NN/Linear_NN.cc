#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "ooo_cpu.h"

constexpr std::size_t HISTORY_LENGTH = 64;

struct Net : torch::nn::Module {
  Net()
      : linear1(register_module("linear1", torch::nn::Linear(1 + HISTORY_LENGTH, 16))), // Input: IP + History
        linear2(register_module("linear2", torch::nn::Linear(16, 1))) {
    another_bias = register_parameter("b", torch::randn(1));
  }

  torch::Tensor forward(torch::Tensor input) {
    input = torch::relu(linear1(input));  // Apply ReLU activation
    return torch::sigmoid(linear2(input) + another_bias);  // Sigmoid to output probability
  }

  torch::nn::Linear linear1, linear2;
  torch::Tensor another_bias;
};

std::bitset<HISTORY_LENGTH> Global_History;
Net net;

void O3_CPU::initialize_branch_predictor() {}

uint8_t O3_CPU::predict_branch(uint64_t ip)
{
    // Convert history into Tensor
    std::vector<float> history_features;
    for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
        history_features.push_back(Global_History[i] ? 1.0f : 0.0f);  // Convert bitset to float
    }

    // Normalize IP
    float norm_ip = static_cast<float>(ip) / static_cast<float>(UINT64_MAX);
    history_features.insert(history_features.begin(), norm_ip);  // Insert normalized IP

    // Convert to Tensor
    torch::Tensor input = torch::tensor(history_features).view({1, HISTORY_LENGTH + 1});

    // Forward pass through neural network
    torch::Tensor output = net.forward(input);
    float prediction = output.item<float>();

    std::cout << "Prediction: " << prediction << std::endl;
    return prediction > 0.5 ? 1 : 0;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
    // Convert history into Tensor
    std::vector<float> history_features;
    for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
        history_features.push_back(Global_History[i] ? 1.0f : 0.0f);
    }

    // Normalize IP
    float norm_ip = static_cast<float>(ip) / static_cast<float>(UINT64_MAX);
    history_features.insert(history_features.begin(), norm_ip);

    // Convert to Tensor
    torch::Tensor input = torch::tensor(history_features).view({1, HISTORY_LENGTH + 1});

    std::cout << "Taken: " << (int)taken << std::endl;

    // Convert expected output
    torch::Tensor target = torch::tensor(static_cast<float>(taken), torch::dtype(torch::kFloat32)).view({1, 1});

    // Define optimizer (Adam for adaptive learning)
    static torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.001));

    // Forward pass
    torch::Tensor prediction = net.forward(input);

    // Compute loss
    torch::Tensor loss = torch::binary_cross_entropy(prediction, target);

    // Backpropagation
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // Update global history
    Global_History >>= 1;
    Global_History[HISTORY_LENGTH - 1] = taken;
}
