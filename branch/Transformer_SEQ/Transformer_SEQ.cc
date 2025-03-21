#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <mutex>
#include <numeric>
#include <filesystem>
#include "ooo_cpu.h"

// Debug settings
#define SAVE_INTERVAL 10000
#define TOGGLE_LEARNING 1
#define DEBUG_INTERVAL 10000

// Model parameters
#define HIDDEN_SIZE 32
#define NUM_LAYERS 1
#define LEARNING_RATE 0.0002
#define NUM_HEADS 4
#define SEQ_LEN 32
#define INPUT_SIZE 16

std::mutex optimizer_mutex;

class PositionalEncoding : public torch::nn::Module {
public:
    PositionalEncoding(int d_model, int max_len = 5000) {
        torch::Tensor pe = torch::zeros({max_len, d_model});
        torch::Tensor position = torch::arange(0, max_len, torch::kFloat32).unsqueeze(1);
        torch::Tensor div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat32) * (-std::log(10000.0) / d_model));
        pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, d_model, 2)}, torch::sin(position * div_term));
        pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, d_model, 2)}, torch::cos(position * div_term));
        pe = pe.unsqueeze(0);
        positional_encoding = register_buffer("positional_encoding", pe);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto pe_slice = positional_encoding.index({torch::indexing::Slice(), torch::indexing::Slice(0, x.size(1)), torch::indexing::Slice()}).to(x.device());
        return x + pe_slice;
    }

private:
    torch::Tensor positional_encoding;
};

class TransformerPredictor : public torch::nn::Module {
public:
    TransformerPredictor()
        : fc_in(register_module("fc_in", torch::nn::Linear(INPUT_SIZE, HIDDEN_SIZE))),
          transformer_encoder_layer(register_module("encoder_layer", torch::nn::TransformerEncoderLayer(torch::nn::TransformerEncoderLayerOptions(HIDDEN_SIZE, NUM_HEADS).dropout(0.1)))),
          transformer_encoder(register_module("encoder", torch::nn::TransformerEncoder(transformer_encoder_layer, NUM_LAYERS))),
          fc_out(register_module("fc_out", torch::nn::Linear(HIDDEN_SIZE, 1))), 
          positional_encoding(register_module("positional_encoding", std::make_shared<PositionalEncoding>(HIDDEN_SIZE, SEQ_LEN))),
          layer_norm(register_module("layer_norm", torch::nn::LayerNorm(std::vector<int64_t>{HIDDEN_SIZE}))),
          optimizer(std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions(LEARNING_RATE))),
          update_count(0), forward_count(0) { torch::set_num_threads(1); }

    void init();
    torch::Tensor predict(uint64_t ip);
    void update(uint64_t ip, uint8_t taken);

    torch::Tensor forward(torch::Tensor input) {
        forward_count++;
        if (forward_count % DEBUG_INTERVAL == 0) std::cout << "Forward pass: Input shape: " << input.sizes() << std::endl;

        input = layer_norm->forward(torch::relu(fc_in(input)));
        input = positional_encoding->forward(input);
        if (input.size(1) > SEQ_LEN) input = input.index({torch::indexing::Slice(), torch::indexing::Slice(0, SEQ_LEN), torch::indexing::Slice()});

        input = input.transpose(0, 1);
        auto memory = transformer_encoder(input).transpose(0, 1);
        auto memory_pooled = memory.mean(1);
        auto output = torch::sigmoid(fc_out(memory_pooled));

        if (forward_count % DEBUG_INTERVAL == 0) std::cout << "Output shape: " << output.sizes() << " Value: " << output << std::endl;
        return output;
    }

private:
    torch::nn::Linear fc_in, fc_out;
    torch::nn::TransformerEncoderLayer transformer_encoder_layer;
    torch::nn::TransformerEncoder transformer_encoder;
    torch::nn::ModuleHolder<PositionalEncoding> positional_encoding;
    torch::nn::LayerNorm layer_norm{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer;
    std::vector<torch::Tensor> input_sequence;
    static constexpr size_t MAX_SEQ_LEN = SEQ_LEN;
    int update_count, forward_count;
};

torch::Tensor TransformerPredictor::predict(uint64_t ip) {
    torch::Tensor features = torch::zeros({1, INPUT_SIZE}, torch::dtype(torch::kFloat32));
    for (size_t i = 0; i < INPUT_SIZE; ++i) features[0][i] = ((ip >> i) & 1) ? 1.0f : 0.0f;

    input_sequence.push_back(features);
    if (input_sequence.size() > SEQ_LEN) input_sequence.erase(input_sequence.begin());
    torch::Tensor input = torch::cat(input_sequence, 0).unsqueeze(0);

    return this->forward(input);
}

void TransformerPredictor::update(uint64_t ip, uint8_t taken) {
    torch::Tensor target = torch::tensor({static_cast<float>(taken)}).view({1, 1});  
    torch::Tensor prediction = this->predict(ip);
    torch::Tensor loss = torch::binary_cross_entropy(prediction, target);

    if (TOGGLE_LEARNING) {
        std::lock_guard<std::mutex> lock(optimizer_mutex);
        this->optimizer->zero_grad();
        loss.backward();
        this->optimizer->step();
    }
}

void TransformerPredictor::init() {}

// Transformer model per CPU core
TransformerPredictor transformer_net[NUM_CPUS];

void O3_CPU::initialize_branch_predictor() { transformer_net[cpu].init(); }

uint8_t O3_CPU::predict_branch(uint64_t ip) { return transformer_net[cpu].predict(ip).item<float>() > 0.5 ? 1 : 0; }

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) { transformer_net[cpu].update(ip, taken); }
