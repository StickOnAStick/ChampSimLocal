#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <bitset>

#include "FixedVector.hh"
#include <nlohmann/json.hpp> // Nlohmann-json dep

using json = nlohmann::json;

class TransformerBase
{
protected:

  int d_in;             // Input Dimensionality (64 bit IP)
  int d_pos;            // Positional encoding size
  int d_model;          // Embeding dimension
  int d_ff;             // Feed-Forward layer size
  int d_q;              // Query Dimension size
  int d_k;              // Key Dimension size
  int d_v;              // Value dimension size

  int num_ma_heads;     // Number of Multi-headed attention heads
  int num_mma_heads;    // Number of Masked Multi-headed Attention Heads

  int sequence_len;     // Number of previous instructions passed in as input
  float dropout_rate;   // Dropout rate

  std::string weights_file;

  FixedVector<FixedVector<float>> sequence_history; // seq_len most recent embedded inputs.
  FixedVector<FixedVector<float>> w_q;
  FixedVector<FixedVector<float>> w_k;
  FixedVector<FixedVector<float>> w_v;
  FixedVector<FixedVector<float>> w_o;
  FixedVector<FixedVector<float>> w_ff1;
  FixedVector<FixedVector<float>> w_ff2;
  FixedVector<float>              b_ff1;
  FixedVector<float>              b_ff2;
  FixedVector<float>              w_out;
  float                           b_out;

public:
  // Construct the transformer from a given input configuration file
  TransformerBase(const std::string& config_file)
  {
    json config = loadConfig(config_file);
   
    // User defined parameters
    d_in = config["d_in"];
    d_model = config["d_model"];
    d_ff = config["d_ff"];
    d_q = config["d_q"];
    d_k = config["d_k"];
    d_v = config["d_v"];
    d_pos = config["d_pos"];
    num_ma_heads = config["num_ma_heads"];
    num_mma_heads = config["num_mma_heads"];
    dropout_rate = config["dropout_rate"];
    sequence_len = config["sequence_len"]; // 24
    weights_file = config["weights_file"];
    if (d_model % num_mma_heads || d_model % num_ma_heads){
        throw std::runtime_error("Model size not compatible with number of heads!");
    }

    // Setup Sequence history matrix.
    FixedVector<FixedVector<float>> matrix(sequence_len, FixedVector<float>(d_model, 0)); // Create 2d (d_model x seq_len) matrix of 0's
    sequence_history = matrix;

    // Setup Weights
    w_q = loadWeights(weights_file, "queries", d_model, d_model);
    w_k = loadWeights(weights_file, "keys", d_model, d_model);
    w_v = loadWeights(weights_file, "values", d_model, d_model);
    w_o = loadWeights(weights_file, "output", d_model, d_model);
    w_ff1 = loadWeights(weights_file, "w_ff1", d_model, d_ff);
    w_ff2 = loadWeights(weights_file, "w_ff2", d_ff, d_model);
    b_ff1 = loadWeights(weights_file, "b_ff2", d_ff);
    b_ff2 = loadWeights(weights_file, "b_ff2", d_model);
    w_out = loadWeights(weights_file, "w_out", d_model);
    b_out = loadWeights(weights_file, "b_out");
  }

  virtual ~TransformerBase() = default;

  json loadConfig(const std::string& config_file)
  {
    std::ifstream file(config_file);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open config file.");
    }

    return json::parse(file);
  }

  FixedVector<FixedVector<float>> loadWeights(
    const std::string& file_name,
    const std::string& weight_key,
    size_t rows,
    size_t cols
  ){
    std::ifstream file(file_name);
    if(!file.is_open()) {
      throw std::runtime_error("Could not open weights file: " + file_name);
    }

    json data = json::parse(file);
    const auto& matrix_data = data[weight_key];

    if (matrix_data.size() != rows || matrix_data[0].size() != cols){
      throw std::runtime_error("Mismatch between provided matrix dimensions and loaded weights");
    }

    // Generate the matrix from json matrix
    FixedVector<FixedVector<float>> matrix(rows, FixedVector<float>(cols, 0.0f));
    for(size_t i = 0; i < rows; ++i){
      for(size_t j = 0; j < cols; ++j){
        matrix[i][j] = matrix_data[i][j];
      }
    }

    return matrix;
  }

   FixedVector<float> loadWeights(
    const std::string& file_name,
    const std::string& weight_key,
    size_t size
  ){
    std::ifstream file(file_name);
    if(!file.is_open()) {
      throw std::runtime_error("Could not open weights file: " + file_name);
    }

    json data = json::parse(file);
    const auto& matrix_data = data[weight_key];

    if (matrix_data.size() != size ){
      throw std::runtime_error("Mismatch between provided matrix dimensions and loaded weights");
    }

    // Generate the matrix from json matrix
    FixedVector<float> matrix(size, 0.0f);
    for(size_t i = 0; i < size; ++i){
        matrix[i] = matrix_data[i];
    }

    return matrix;
  }

  float loadWeights(
    const std::string& file_name,
    const std::string& weight_key
  ){
    std::ifstream file(file_name);
    if(!file.is_open()) {
      throw std::runtime_error("Could not open weights file: " + file_name);
    }

    json data = json::parse(file);

    if(!data.contains(weight_key)){
      throw std::runtime_error("Key not found in weights file: " + weight_key);
    }

    try {
      return data[weight_key].get<float>();
    } catch (const json::type_error& e){
      throw std::runtime_error("Invalid type for key: " + weight_key + ". Expected as a float.");
    }

  }

  // Returns vector of [d_in + d_pos, sequence_len] of floating point "binary-vectors" (Only binary values stored in each float)
  // [d_model * sequence_len]
  // The following needs to be updated for dynamic bitset sizing. (Should be this->sequence_len)
  virtual void hashed_posEncoding(uint64_t& input, std::bitset<24> global_history) = 0;
  virtual void fixed_posEncoding(uint64_t& ip) = 0;
  //virtual void learnable_posEncoding(uint64_t ip) = 0;

  // [seuqnece_len * d_model]  (d_model is == to 96-bit positional ecoding)
  virtual FixedVector<FixedVector<float>> MMALayer(const FixedVector<FixedVector<float>>& input) = 0;

  // [sequence_len, d_model]
  virtual FixedVector<FixedVector<float>> MALayer(bool use_mask) = 0;
      // [num_heads, sequence_len, d_(q,k,v)]

  // Input: [sequence_len, d_model]
  // Output: [sequence_len, d_model]
  virtual FixedVector<FixedVector<float>> FFLayer(FixedVector<FixedVector<float>>& input) = 0;
  virtual float layerNormalization(FixedVector<FixedVector<float>>& input) = 0;

  virtual bool predict(uint64_t input) = 0; // Final output, branch taken, or not
};