#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <bitset>
#include <fmt/chrono.h>
#include <fmt/core.h>

#include "FixedVector.hh"
#include <nlohmann/json.hpp> // Nlohmann-json dep

using json = nlohmann::json;


struct ForwardContext {
  // Memoization of forward pass for fast backward pass 
  uint64_t ip; // Instruction pointer called.
  FixedVector<FixedVector<float>> input;

  // MMA Attention Intermediate results - Q, K, V = [seq_len, d_model]
  FixedVector<FixedVector<float>> Q, K, V; // [seq_len, d_head]
  FixedVector<FixedVector<FixedVector<float>>> softmax_attn;      // after Softmax  [num_heads, [seq_len, seq_len]]
  FixedVector<FixedVector<float>> attn_out;          // After W_O        [seq_len, d_model]
  FixedVector<FixedVector<float>> attn_post_residual;// [seq_len, d_model]
  FixedVector<float> attn_mean_i;                    // The mean of each row vector during normalization
  FixedVector<float> attn_var_i;                     // The variance of each row vector during normalization

  // Feed Forward
  FixedVector<FixedVector<float>> ffnInput; // Input after attn + residual [seq_len, d_model]
  FixedVector<FixedVector<float>> ff_intermediate; 
  FixedVector<FixedVector<float>> ff_post_residual; // [seq_len, d_model]
  FixedVector<FixedVector<float>> ff_normed;        //
  FixedVector<float> ff_mean_i;                     // The mean of each row vector during normalization
  FixedVector<float> ff_var_i;                      // The variance of each row vector during normalization

  FixedVector<float> pooled; // Pooled Sequence represenation [d_model]
  float out; // Sigmoid Output

  // Meaningful default constructor -- Be considerate when changing d_model, seq_len 
  ForwardContext(size_t seq_len = 24, size_t d_model = 70, size_t num_heads = 5, size_t d_ff = 2*70) 
      : 
        ip(0),
        input(seq_len, FixedVector<float>(d_model, 0.0f)),
        Q(seq_len, FixedVector<float>(d_model, 0.0f)),
        K(seq_len, FixedVector<float>(d_model, 0.0f)),
        V(seq_len, FixedVector<float>(d_model, 0.0f)),
        softmax_attn(num_heads, FixedVector<FixedVector<float>>(seq_len, FixedVector<float>(seq_len, 0.0f))),
        attn_out(seq_len, FixedVector<float>(d_model, 0.0f)),
        attn_post_residual(seq_len, FixedVector<float>(d_model, 0.0f)),
        attn_mean_i(seq_len, 0.0f),
        attn_var_i(seq_len, 0.0f),
        ffnInput(seq_len, FixedVector<float>(d_model, 0.0f)),
        ff_intermediate(seq_len, FixedVector<float>(d_ff, 0.0f)),
        ff_post_residual(seq_len, FixedVector<float>(d_model, 0.0f)),
        ff_normed(seq_len, FixedVector<float>(d_model, 0.0f)),
        ff_mean_i(seq_len, 0.0f),
        ff_var_i(seq_len, 0.0f),
        pooled(d_model, 0.0f),
        out(0.0f){};
        
};

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

  // ADAM Moment Arrays (same shape as corresponding weights)
  FixedVector<FixedVector<float>> w_q_m; // First moment array denoted by _m
  FixedVector<FixedVector<float>> w_k_m;
  FixedVector<FixedVector<float>> w_v_m;
  FixedVector<FixedVector<float>> w_o_m;
  FixedVector<FixedVector<float>> w_ff1_m;
  FixedVector<FixedVector<float>> w_ff2_m;
  FixedVector<float>              b_ff1_m;
  FixedVector<float>              b_ff2_m;
  FixedVector<float>              w_out_m;
  float                           b_out_m;

  FixedVector<FixedVector<float>> w_q_v; // Second moment array denoted by _v
  FixedVector<FixedVector<float>> w_k_v;
  FixedVector<FixedVector<float>> w_v_v;
  FixedVector<FixedVector<float>> w_o_v;
  FixedVector<FixedVector<float>> w_ff1_v;
  FixedVector<FixedVector<float>> w_ff2_v;
  FixedVector<float>              b_ff1_v;
  FixedVector<float>              b_ff2_v;
  FixedVector<float>              w_out_v;
  float                           b_out_v;

  // Track Adam iteration
  int adam_step;

  float beta1;
  float beta2;
  float epsilon;

  

  // std::deque<state_buf>           hist_state_buf;
private:
  // Function to modify the original filename and append "-OUT" before the extension
  std::string getOutputFileName() const {
    size_t dotPos = this->weights_file.find_last_of(".");
    if (dotPos == std::string::npos) {
      return this->weights_file + "-OUT.json"; // If no extension, append directly
    }
    return this->weights_file.substr(0, dotPos) + "-OUT" + this->weights_file.substr(dotPos);
  }

  // Convert FixedVector<FixedVector<float>> to JSON format
  json convertMatrixToJson(const FixedVector<FixedVector<float>>& matrix) {
    json result = json::array();
    for (const auto& row : matrix) {
      json rowJson = json::array();
      for (const auto& val : row) {
        rowJson.push_back(val);
      }
      result.push_back(rowJson);
    }
    return result;
  }

  // Convert FixedVector<float> to JSON format
  json convertVectorToJson(const FixedVector<float>& vector) {
    json result = json::array();
    for (const auto& val : vector) {
      result.push_back(val);
    }
    return result;
  }

public:

  float lr; // learning rate                       

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
    // Add learning rate
    sequence_len = config["sequence_len"]; // 24
    weights_file = config["weights_file"];
    if (d_model % num_mma_heads || d_model % num_ma_heads){
        throw std::runtime_error("Model size not compatible with number of heads!");
    }

    // Adam hyperparameters
    beta1 = config["beta1"];
    beta2 = config["beta2"];
    epsilon = config["epsilon"];
    lr = config["learning_rate"];

    // Setup Sequence history matrix.
    sequence_history = FixedVector<FixedVector<float>>(sequence_len, FixedVector<float>(d_model, 0.0f));

    // Setup Weights
    w_q = loadWeights(weights_file, "queries", d_model, d_model);
    w_k = loadWeights(weights_file, "keys", d_model, d_model);
    w_v = loadWeights(weights_file, "values", d_model, d_model);
    w_o = loadWeights(weights_file, "output", d_model, d_model);
    w_ff1 = loadWeights(weights_file, "w_ff1", d_model, d_ff);
    w_ff2 = loadWeights(weights_file, "w_ff2", d_ff, d_model);
    b_ff1 = loadWeights(weights_file, "b_ff1", d_ff);
    b_ff2 = loadWeights(weights_file, "b_ff2", d_model);
    w_out = loadWeights(weights_file, "w_out", d_model);
    b_out = loadWeights(weights_file, "b_out");

    // Adam moment vectors m and v
    w_q_m = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f)); 
    w_k_m = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f)); 
    w_v_m = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f)); 
    w_o_m = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f)); 
    w_ff1_m = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_ff, 0.0f));
    w_ff2_m = FixedVector<FixedVector<float>>(d_ff, FixedVector<float>(d_model, 0.0f));
    b_ff1_m = FixedVector<float>(d_ff, 0.0f);
    b_ff2_m = FixedVector<float>(d_model, 0.0f);
    w_out_m = FixedVector<float>(d_model, 0.0f);
    b_out_m = 0.0f;
    w_q_v = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f)); 
    w_k_v = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f)); 
    w_v_v = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f)); 
    w_o_v = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f)); 
    w_ff1_v = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_ff, 0.0f));
    w_ff2_v = FixedVector<FixedVector<float>>(d_ff, FixedVector<float>(d_model, 0.0f));
    b_ff1_v = FixedVector<float>(d_ff, 0.0f);
    b_ff2_v = FixedVector<float>(d_model, 0.0f);
    w_out_v = FixedVector<float>(d_model, 0.0f);
    b_out_v = 0.0f;
    // Track adam iteration
    adam_step = 0;

  }

  virtual ~TransformerBase() {
    saveWeights();
  };

  void saveWeights(){
    std::string out_file_name = getOutputFileName();

    std::ofstream outFile(out_file_name);
    if(!outFile.is_open()){
      throw std::runtime_error("Could not open file for writing weights: " + out_file_name);
    }

    json data;
    
    // Store updated weights into JSON
    data["queries"] = convertMatrixToJson(w_q);
    data["keys"] = convertMatrixToJson(w_k);
    data["values"] = convertMatrixToJson(w_v);
    data["output"] = convertMatrixToJson(w_o);
    data["w_ff1"] = convertMatrixToJson(w_ff1);
    data["w_ff2"] = convertMatrixToJson(w_ff2);
    data["b_ff1"] = convertVectorToJson(b_ff1);
    data["b_ff2"] = convertVectorToJson(b_ff2);
    data["w_out"] = convertVectorToJson(w_out);
    data["b_out"] = b_out; // Assuming it's a single float

    // Write the updated JSON to the new file
    outFile << data.dump(4); // Pretty-print with 4 spaces for readability
    outFile.close();

    std::cout << "Weights saved to " << out_file_name << std::endl;
  }

  int get_seq_len(){
    return this->sequence_len;
  }
  int get_d_model(){
    return this->d_model;
  }
  int get_head_count(){
    return this->num_mma_heads;
  }

  json loadConfig(const std::string& config_file)
  {
    std::string path = __FILE__; // Path to transformer.cc
    path = path.substr(0, path.find_last_of("/\\"))+"/"+config_file; // Remove /transformer.cc from path
    
    std::ifstream file(path);
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
    std::string path = __FILE__; // Path to transformer.cc
    path = path.substr(0, path.find_last_of("/\\")) + "/" + file_name; // Remove /transformer.cc from path
    
    std::ifstream file(path);
    if(!file.is_open()) {
      throw std::runtime_error("Could not open weights file: " + file_name);
    }

    json data = json::parse(file);
    const auto& matrix_data = data[weight_key];

    if (matrix_data.size() != rows || matrix_data[0].size() != cols){
      throw std::runtime_error("Mismatch between provided matrix dimensions and loaded weights for " + weight_key);
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
    std::string path = __FILE__; // Path to transformer.cc
    path = path.substr(0, path.find_last_of("/\\")) + "/" + file_name; // Remove /transformer.cc from path
    
    std::ifstream file(path);
    if(!file.is_open()) {
      throw std::runtime_error("Could not open weights file: " + file_name);
    }

    json data = json::parse(file);
    const auto& matrix_data = data[weight_key];

    if (matrix_data.size() != size ){
      throw std::runtime_error("Mismatch between provided matrix dimensions and loaded weights for " + weight_key);
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
    std::string path = __FILE__; // Path to transformer.cc
    path = path.substr(0, path.find_last_of("/\\")) + "/" + file_name; // Remove /transformer.cc from path
    
    std::ifstream file(path);
    if(!file.is_open()) {
      throw std::runtime_error("Could not open weights file: " + file_name);
    }

    json data = json::parse(file);

    if(!data.contains(weight_key)){
      throw std::runtime_error("Key not found in weights file: " + weight_key);
    }

    try {
      if (data[weight_key].is_array()){ // numpy dumps rand floats as arry, Array check/return 
        return data[weight_key][0].get<float>();
      } else {
        return data[weight_key].get<float>();
      }
    } catch (const json::type_error& e){
      throw std::runtime_error("Invalid type for key: " + weight_key + ". Expected as a float.");
    }

  }

  // Returns vector of [d_in + d_pos, sequence_len] of floating point "binary-vectors" (Only binary values stored in each float)
  // [d_model * sequence_len]
  // The following needs to be updated for dynamic bitset sizing. (Should be this->sequence_len)
  virtual void hashed_posEncoding(uint64_t input) = 0;
  virtual void fixed_posEncoding(uint64_t ip) = 0;
  //virtual void learnable_posEncoding(uint64_t ip) = 0;

  // [seuqnece_len * d_model]  (d_model is == to 96-bit positional ecoding)
  //virtual FixedVector<FixedVector<float>> MMALayer(const FixedVector<FixedVector<float>>& input) = 0;

  // [sequence_len, d_model], inside it transforms to [seq_len, d_k] where d_k = d_model / h. h = number of heads
  virtual FixedVector<FixedVector<float>> MALayer(ForwardContext& ctx, bool use_mask = true) = 0;
      // [num_heads, sequence_len, d_(q,k,v)]

  // Input: [sequence_len, d_model]
  // Output: [sequence_len, d_model]
  virtual FixedVector<FixedVector<float>> FFLayer(ForwardContext& ctx, FixedVector<FixedVector<float>>& input) = 0;
  virtual float pooledOutput(ForwardContext& ctx, FixedVector<FixedVector<float>>& input) = 0;

  virtual bool predict(ForwardContext& ctx, uint64_t input) = 0; // Final output, branch taken, or not

  virtual void backwardsPass(ForwardContext& ctx, float y_true, float learning_rate) = 0;
};