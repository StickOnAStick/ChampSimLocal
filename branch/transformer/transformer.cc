#include "transformer.hh"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <deque>
#include <map>
#include <random>
#include <fmt/chrono.h>
#include <fmt/core.h>

#include  "FixedVectorMath.hh"


#include "msl/fwcounter.h"
#include "ooo_cpu.h"

#include "FixedVector.hh"


namespace
{
class Transformer : public TransformerBase
{
public:
  Transformer(const std::string& config_file) : TransformerBase(config_file) {}

  void hashed_posEncoding(uint64_t input) override {

    uint64_t global = 0x121212; // Use global_history
    uint64_t hashed_input = (input & 0xFFF) ^ global; // Use 12 LSBs of IP, smaller locality, reduced HW cost

    // Positionally encode based off hashed input XOR'd with recent global history
    uint8_t pos_enc = (hashed_input % static_cast<int>(pow(2, this->d_pos))); // Reduce to 5 bits.

    // Add IP bits to the constructed d_model vector
    FixedVector<float> encoded_input(this->d_model, 0.0f);
    for(int i = 0; i < this->d_in; i++){
      int bit = (input >> i) & 1;
      encoded_input[i] = bit;
    }

    // Add the positional encoding bits to the input d_model vector
    for (int i = 0; i < this->d_pos; i++) {
      int bit = (pos_enc >> i) & 1;
      encoded_input[this->d_in + i] = bit;
    }

    // Add the new input to the beginning of sequence history.
    this->sequence_history.push(encoded_input);
  }

  void fixed_posEncoding(uint64_t ip) override {
    //fmt::println("Positional Encoding: fixed"); // Know where your at when it crashes!

    FixedVector<float> encoded_input(this->d_model, 0.0f);

    for(int i = 0; i < this->d_model; i++){
      encoded_input[i] = (ip >> i) & 1;
    }

    // Push the new IP into history
    this->sequence_history.push(encoded_input);

    // Incriment all previous IP's positional encodings by 1
    for (uint8_t pos = static_cast<uint8_t>(this->sequence_history.size())-1; pos > 0; --pos) {
      for (int j = 0; j < this->d_pos; j++) {
        this->sequence_history[pos][this->d_in + j] = (pos >> j) & 1;
      }
    }
  }

  FixedVector<FixedVector<float>> MALayer(bool use_mask = false, ForwardContext& ctx) override {
    //fmt::println("MA Layer. masked: {}", use_mask); // Know where your at when it crashes!

    /*
      Attention per-head = softMax( QK^T / sqrt(d_k) + M ) V

      Q: Query Matrix (seq_len x d_k)  = XW^Q
      K: Key Matrix   (seq_len x d_k)  = XW^K
      V: Value Matrix (seq_len x d_v)  = XW^V
      M: Mask Matrix  (seq_len x seq_len)

      d_k: Dimensionality of key/query == d_model / num_heads(h)
      seq_len: Sequence length of our model

      Multi-Headed:
          MultiHead(Q,K,V) = Concat(head1, head2,...,head_h)W^O

          head_i = Attention( Q*W^Q_i, K*W^K_i, V*W^V_i )
          W^(Q,K,V)_i: Learnable weight matricies for each head (d_model x d_k)
          W^O: Output projection Weight matrix ((h * d_v) x d_model)

          Output = (seq_len x d_model)
    */

    if (this->d_model % this->num_ma_heads != 0){
      throw std::runtime_error("Model dimension (d_model) must be divisible by the number of heads");
    }

    int d_head = this->d_model / this->num_ma_heads;

    // Output matrix
    FixedVector<FixedVector<float>> attention_out(sequence_len, FixedVector<float>(d_model, 0.0f));

    FixedVector<FixedVector<float>> mask(sequence_len, FixedVector<float>(sequence_len, 0.0f));
    if (use_mask){
      FixedVectorMath::applyMask(mask);
    }

    /*
      Step 1: Compute Q, K, V
      ---------------------------------
      Using pre-loaded w_q, w_k, w_v weight matricies we construct Q, K, V vectors 

      Q, K, V = seq_len * w_q,k,v
      ---------------------------------------
      Dimensions:
      - sequence_history: [seq_len, d_model]
      - w_q, w_v, w_k: [d_model, d_q] [d_model, d_k] [d_model, d_v]
      - Q, K, V:  [seq_len, d_q] [seq_len, d_v]
    */
    // Compute Q, K, V
    FixedVector<FixedVector<float>> Q = FixedVectorMath::dotProduct(sequence_history, w_q);
    FixedVector<FixedVector<float>> K = FixedVectorMath::dotProduct(sequence_history, w_k);
    FixedVector<FixedVector<float>> V = FixedVectorMath::dotProduct(sequence_history, w_v);

    /*
      Step 2. Process Each Head
      - Slice Q, K, V for each head
      - Compute Attention scores:   Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k_head)) * V
      - Concat results into final output
    */
    for(int head = 0; head < num_ma_heads; ++head){
      
      /* Each of these is a "slice" of the original Q, K, V vectors.
        This is NOT optimal but it's hashed together quickly
        Future revisions should change this for loop to iterate through each slice without
        creating new vectors. (wasted memory and cycles)
      */

      // Slice of each head.
      FixedVector<FixedVector<float>> Q_head(sequence_len, FixedVector<float>(d_head, 0.0f));
      FixedVector<FixedVector<float>> K_head(sequence_len, FixedVector<float>(d_head, 0.0f));
      FixedVector<FixedVector<float>> V_head(sequence_len, FixedVector<float>(d_head, 0.0f));

      for (int i = 0; i < sequence_len; ++i){ // Gross copy of slice
        for (int j = 0; j < d_head; ++j){
          // To note about this, rows are the sequence history, cols are the low-rank embeddings of d_model for Q, K, V
          Q_head[i][j] = Q[i][head * d_head + j]; 
          K_head[i][j] = K[i][head * d_head + j];
          V_head[i][j] = V[i][head * d_head + j];
        }
      }

      /*
        Step 3: Scaled Dot-produc attention
        ----------------------------------------
        - Compute attention scores
        - Softmax attention scores
        - Weighted Sum output

        attention(Q,K,V) = softmax((QK^T) / sqrt(d_head) + M) 

        - Computes Attention scores for this head
      */
      FixedVector<FixedVector<float>> attention_scores(sequence_len, FixedVector<float>(sequence_len, 0.0f));
      for (int i = 0; i < sequence_len; ++i){
        for (int j = 0; j < sequence_len; ++j){
          float score = 0.0f;
          
          // QK^T
          for (int k = 0; k < d_head; ++k){
            score += Q_head[i][k] * K_head[j][k];
          }

          // QK^T / sqrt(d_head)
          attention_scores[i][j] = score / std::sqrt(static_cast<float>(d_head));

          // QK^T / sqrt(d_head) + M
          if (use_mask){
            attention_scores[i][j] += mask[i][j]; // Either adds 0 or -∞
          }
        }
      }

      // Softmax the attention scores (row-wise)
      FixedVectorMath::softmax(attention_scores);

      // Compute head_out = attention_scores * V_head
      // [seq_len, d_head]
      FixedVector<FixedVector<float>> head_out(sequence_len, FixedVector<float>(d_head, 0.0f));
      for(int i = 0; i < sequence_len; ++i){    // Implemented Mul won't work here
        for(int j = 0; j < sequence_len; ++j){
          for(int k = 0; k < d_head; ++k){
            head_out[i][k] += attention_scores[i][j] * V_head[j][k];
          }
        }
      }

      /*
        Step 4: Concat all heads
        
        We sliced the input sequence history across N heads, 
        now we stick head_out of each slice into a single output. 
      */
      for(int i = 0; i < sequence_len; ++i){
        for(int j = 0; j < d_head; ++j){
          attention_out[i][head * d_head + j] = head_out[i][j];
        }
      }
    }
    /* 
      We now have concat(head_0, head_1, ... head_h) of dim [seq_len, d_model]
      Now apply the output weight matrix
      
      output = concat(head_0...head_h)*W_O 

      where W_O is of dim [d_model, d_model]
    */

    FixedVector<FixedVector<float>> output = FixedVectorMath::dotProduct(attention_out, w_o);

    return output;
  }

  FixedVector<FixedVector<float>> FFLayer(FixedVector<FixedVector<float>>& input) override {
    //fmt::println("Feed Forward Layer"); // Know where your at when it crashes!
    /*
      FFN(x) = ReLU(0, xW_1 + b_1)W_2 + b_2

      Flow:
        1) hidden = input * w_ff1 + b_ff1
        2) hidden = ReLU(hidden)
        3) output = hidden * w_ff2 + b_ff2

      Matrix sizes:
        - in/out: [seq_len, d_model]
        - w_ff1: [d_model, d_ff]
        - b_ff1: [d_ff]
        - w_ff2: [d_ff, d_model]
        - b_ff2: [d_model]

        NOTE: The output is of size [seq_len, d_model], not the final prediction d_out [1]
    */

    // --------------------------------------------------
    // 1) hidden = input * w_ff1 + b_ff1
    //    => hidden: shape [seq_len, d_ff]
    // --------------------------------------------------
    FixedVector<FixedVector<float>> hidden = FixedVectorMath::linear(input, w_ff1, b_ff1);
    
    //---------------------------------------------------
    // 2.) Relu in place
    //---------------------------------------------------
    FixedVectorMath::relu(hidden);

    //---------------------------------------------------
    // 3.) output = hidden * w_ff2 + b_ff2
    //     => output: shapre [seq_len, d_model]
    //---------------------------------------------------
    FixedVector<FixedVector<float>> output = FixedVectorMath::linear(hidden, w_ff2, b_ff2);

    return output;
  }

  float layerNormalization(FixedVector<FixedVector<float>>& input) override {
    //fmt::println("Final Layer Normalization");
    /*
      input = [seq_len, d_model]
      out   = [1]

      pooled = 1/seq_len Σ h_i   // 1 to seq_len
      logits = w_out^T * pooled + b_out
    */

    //--------------------------------------------------------
    // 1.) Get Pooled values
    //     => [d_model]
    //--------------------------------------------------------
    FixedVector<float> pooled(input[0].size(), 0.0f);
    for(int i = 0; i < this->sequence_len; ++i){              // Σ h_i
      for(size_t j = 0; j < input[0].size(); ++j){
        pooled[j] += input[i][j];
      }
    }
    for(size_t i = 0; i < pooled.size(); ++i){        // 1/seq_len
      pooled[i] /= (float)this->sequence_len;
    }

    //--------------------------------------------------------
    // 2.) Compute logits
    //     => [1]
    //--------------------------------------------------------
    float logits = 0.0f;
    for(size_t i = 0; i < pooled.size(); ++i){
      logits += pooled[i] * w_out[i];
    }
    logits += b_out;

    //--------------------------------------------------------
    // 3.) Sigmoid activation
    //     Still don't know if this will be optimal, but we will use for the inital tests.
    //--------------------------------------------------------
    float out = 1.0f / (1.0f + std::exp(-logits));

    return out;
  }

  float predict(uint64_t ip, ForwardContext& ctx) override {

    /*
      Positional Encoding

      Dealers choice, test with correct weights
    */
    //this->hashed_pos_encoding(&ip); // We want to use this one but it relies on global history which is not yet figured out.
    this->fixed_posEncoding(ip);


    /*
      Masked Multi-Headed Attention
    */
    FixedVector<FixedVector<float>> MMA_out = this->MALayer(true);
    FixedVectorMath::add(MMA_out, this->sequence_history); // Result stored in MMA_Out
    FixedVectorMath::normalize(MMA_out); // Store mean and variance for back prop

    /*
      Feed-Forward Layer
    */
    FixedVector<FixedVector<float>> FF_out = this->FFLayer(MMA_out);
    FixedVectorMath::add(FF_out, MMA_out);
    FixedVectorMath::normalize(FF_out); // Store mean and variance for back prop

    float out = this->layerNormalization(FF_out);

    return out;
  }
};

struct Prediction {
  uint64_t  ip;
  bool   prediction;
  float  output;      // Probability given for the prediction
  std::vector<bool> history;
};

constexpr std::size_t NUM_UPDATE_ENTRIES = 100; // Size of the buffer for keeping 'perceptron_state' for update
// Arbitrarily set to the same # of entries as perceptron. Space limitations require careful consideration of this value.


/*
  Map to the O3_CPU instance
  Multi-Core tests will have their own O3_CPU instance
*/
std::map<O3_CPU*, Transformer> predictors;
/*
  Store Speculative and Global branch histories of predicitons.
  Note: Vector of bools operates similar to a dynamic bitset
*/
std::map<O3_CPU*, std::vector<bool>> global_history;          // What actually happened
std::map<O3_CPU*, std::vector<bool>> spec_global_history;     // What we think happened

/*
  Memoize the Forward pass results for faster backprop. 
*/
std::map<O3_CPU*, std::deque<ForwardContext>> prediction_state_buf;

} // namespace


void O3_CPU::initialize_branch_predictor() {
  ::predictors.emplace(this, "spec.json");
  int seq_len = ::predictors.at(this).get_seq_len();
  ::global_history.at(this) = std::vector<bool>(seq_len, 0);
  ::spec_global_history.at(this) = std::vector<bool>(seq_len, 0);
}

uint8_t O3_CPU::predict_branch(uint64_t ip) { 

  int d_model = ::predictors.at(this).get_d_model();
  int seq_len = ::predictors.at(this).get_seq_len();

  ForwardContext ctx = ForwardContext(seq_len, d_model);

  // Get the transformers prediction. 
  float output = ::predictors.at(this).predict(ip, ctx);
  bool prediction = output > 0.5; // Threshold

  // Record the prediction and current state of the transformer which led to this prediction
  ::prediction_state_buf.at(this).push_back({ip, prediction, ::spec_global_history.at(this) });
  if(::prediction_state_buf.at(this).size() > ::NUM_UPDATE_ENTRIES)
    ::prediction_state_buf.at(this).pop_front();
  
  ::spec_global_history.at(this).push_back(prediction);
  if (::spec_global_history.at(this).size() > ::predictors.at(this).get_seq_len())
    ::spec_global_history.at(this).erase(::spec_global_history.at(this).begin());


  //fmt::println("Transformer predicted: {} for ip {}\n", prediction, ip);

  return prediction;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
  
  auto state = std::find_if(
    std::begin(::prediction_state_buf.at(this)), 
    std::end(::prediction_state_buf.at(this)), 
    [ip](const ::Prediction& x) { return x.ip == ip; }
  );
  if (state == std::end(::prediction_state_buf.at(this)))
    return; // Skip update. State was lost.
  
  auto [_ip, prediction, history] = *state;
  ::prediction_state_buf.at(this).erase(state);

  ::global_history.at(this).push_back(taken);
  if (::global_history.at(this).size() > ::predictors.at(this).get_seq_len())
    ::global_history.at(this).erase(::global_history.at(this).begin());
  


  return;
}