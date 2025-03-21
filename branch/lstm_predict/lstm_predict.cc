#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "ooo_cpu.h"

struct LSTMNet : torch::nn::Module {
    LSTMNet(int input_size, int hidden_size, int num_layers, float lr = 0.01)
        : lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers)))),
          fc(register_module("fc", torch::nn::Linear(hidden_size, 1))),
          learning_rate(lr) {
        hidden_state = torch::zeros({num_layers, 1, hidden_size});
        cell_state = torch::zeros({num_layers, 1, hidden_size});
        optimizer = std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(learning_rate));
    }

    torch::Tensor forward(torch::Tensor input) {
        // Pass input and hidden states into LSTM
        auto lstm_out = lstm->forward(input, std::make_tuple(hidden_state, cell_state));

        // Extract output and new states from the tuple
        torch::Tensor output = std::get<0>(lstm_out);   // LSTM output
        auto new_states = std::get<1>(lstm_out);        // Tuple (hidden_state, cell_state)
        
        hidden_state = std::get<0>(new_states).detach();  // Detach hidden state for next step
        cell_state = std::get<1>(new_states).detach();    // Detach cell state for next step

        return torch::sigmoid(fc(output));  // Pass through linear layer and apply sigmoid
    }

    void set_learning_rate(float lr) {
        learning_rate = lr;
        optimizer = std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(learning_rate));  // Reinitialize optimizer
    }

    torch::nn::LSTM lstm;
    torch::nn::Linear fc;
    torch::Tensor hidden_state;
    torch::Tensor cell_state;
    float learning_rate;
    std::unique_ptr<torch::optim::Adam> optimizer;
};

constexpr std::size_t HISTORY_LENGTH = 64;
std::bitset<HISTORY_LENGTH> Global_History;
LSTMNet lstm_net(1, 16, 1);  // Input: (IP + History), 16 hidden units, 1 layer

void O3_CPU::initialize_branch_predictor() {}

uint8_t O3_CPU::predict_branch(uint64_t ip) {
    // Convert history into Tensor
    std::vector<float> history_features;
    for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
        history_features.push_back(Global_History[i] ? 1.0f : 0.0f);
    }

    // Normalize IP and append history
    float norm_ip = static_cast<float>(ip) / static_cast<float>(UINT64_MAX);
    history_features.insert(history_features.begin(), norm_ip);
    
    torch::Tensor input = torch::tensor(history_features).view({1, 1, HISTORY_LENGTH + 1});  // [seq_len=1, batch=1, feature_dim]

    // Forward pass
    torch::Tensor output = lstm_net.forward(input);
    float prediction = output.item<float>();

    std::cout << "Prediction: " << prediction << std::endl;
    return prediction > 0.5 ? 1 : 0;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
    // Convert history into Tensor
    std::vector<float> history_features;
    for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
        history_features.push_back(Global_History[i] ? 1.0f : 0.0f);
    }

    // Normalize IP and append history
    float norm_ip = static_cast<float>(ip) / static_cast<float>(UINT64_MAX);
    history_features.insert(history_features.begin(), norm_ip);
    
    torch::Tensor input = torch::tensor(history_features).view({1, 1, HISTORY_LENGTH + 1});  // [seq_len=1, batch=1, feature_dim]

    // Convert expected output
    torch::Tensor target = torch::tensor(static_cast<float>(taken), torch::dtype(torch::kFloat32)).view({1, 1});

    // Forward pass
    torch::Tensor prediction = lstm_net.forward(input);

    // Compute loss
    torch::Tensor loss = torch::binary_cross_entropy(prediction, target);

    // Backpropagation
    lstm_net.optimizer->zero_grad();
    loss.backward();
    lstm_net.optimizer->step();

    // Update global history
    Global_History >>= 1;
    Global_History[HISTORY_LENGTH - 1] = taken;
}
