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

#include "FixedVectorMath.hh"

#include "msl/fwcounter.h"
#include "ooo_cpu.h"

// Quick n dirty implementation
#define USE_CUDA 1 // 0 to disable

namespace
{
class Transformer : public TransformerBase
{
public:
  Transformer(const std::string& config_file) : TransformerBase(config_file) {}

  void hashed_posEncoding(uint64_t input) override {
    //fmt::println("Positional Encoding: hashed"); // Know where your at when it crashes!

    uint64_t global = 0x121212; // Use the inbuilt global history
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

  FixedVector<FixedVector<float>> MALayer(ForwardContext& ctx, bool use_mask = false) override {
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
    
    ctx.Q = USE_CUDA ? FixedVectorMath::dotProductCuda(sequence_history, w_q) : FixedVectorMath::dotProduct(sequence_history, w_q);
    ctx.K = USE_CUDA ? FixedVectorMath::dotProductCuda(sequence_history, w_k) : FixedVectorMath::dotProduct(sequence_history, w_k);
    ctx.V = USE_CUDA ? FixedVectorMath::dotProductCuda(sequence_history, w_v) : FixedVectorMath::dotProduct(sequence_history, w_v);
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
          Q_head[i][j] = ctx.Q[i][head * d_head + j]; 
          K_head[i][j] = ctx.K[i][head * d_head + j];
          V_head[i][j] = ctx.V[i][head * d_head + j];
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
      ctx.attn_scores = attention_scores;
      // Softmax the attention scores (row-wise)
      FixedVectorMath::softmax(attention_scores);
      ctx.softmax_attn = attention_scores;

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
    ctx.attn_out = USE_CUDA ? FixedVectorMath::dotProductCuda(attention_out, w_o) : FixedVectorMath::dotProduct(attention_out, w_o);
    return ctx.attn_out;
  }

  FixedVector<FixedVector<float>> FFLayer(ForwardContext& ctx, FixedVector<FixedVector<float>>& input) override {
    /*
      FFN(x) = ReLU(0, xW_1 + b_1)W_2 + b_2

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

    FixedVector<FixedVector<float>> hidden = USE_CUDA ? FixedVectorMath::linearCuda(input, w_ff1, b_ff1) : FixedVectorMath::linear(input, w_ff1, b_ff1);
    ctx.ffnIntermediate = hidden;
    //---------------------------------------------------
    // 2.) Relu in place
    //---------------------------------------------------
    FixedVectorMath::relu(hidden);
    ctx.ffnActivated = hidden;
    //---------------------------------------------------
    // 3.) output = hidden * w_ff2 + b_ff2
    //     => output: shapre [seq_len, d_model]
    //---------------------------------------------------
    FixedVector<FixedVector<float>> output = USE_CUDA ? FixedVectorMath::linearCuda(hidden, w_ff2, b_ff2) : FixedVectorMath::linear(hidden, w_ff2, b_ff2);
    ctx.ffnOut = output;
    return ctx.ffnOut;
  }

  float pooledOutput(ForwardContext& ctx, FixedVector<FixedVector<float>>& input) override {
    /*
      Reduce the entire transformer state to a single prediction. 

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
    ctx.pooled = pooled;

    //--------------------------------------------------------
    // 2.) Compute logit
    //     => [1]
    //--------------------------------------------------------
    float logit = 0.0f;
    for(size_t i = 0; i < pooled.size(); ++i){
      logit += pooled[i] * w_out[i];
    }
    logit += b_out;
    ctx.logit = logit;

    //--------------------------------------------------------
    // 3.) Sigmoid activation
    //     Still don't know if this will be optimal, but we will use for the inital tests.
    //--------------------------------------------------------
    float out = 1.0f / (1.0f + std::exp(-logit));
    ctx.out = out;

    return out;
  }

  bool predict(ForwardContext& ctx, uint64_t ip){

    /*
      Positional Encoding

      Dealers choice, test with correct weights
    */
    //this->hashed_pos_encoding(&ip); // We want to use this one but it relies on global history which is not yet figured out.
    this->fixed_posEncoding(ip);


    /*
      Masked Multi-Headed Attention
    */
    FixedVector<FixedVector<float>> MMA_out = this->MALayer(ctx, true);
    FixedVectorMath::add(MMA_out, this->sequence_history); // Result stored in MMA_Out
    FixedVectorMath::normalize(MMA_out);
    ctx.ffnInput = MMA_out;

    /*
      Feed-Forward Layer
    */
    FixedVector<FixedVector<float>> FF_out = this->FFLayer(ctx, MMA_out);
    FixedVectorMath::add(FF_out, MMA_out);
    FixedVectorMath::normalize(FF_out);
    ctx.ffnOut = FF_out;

    float out = this->pooledOutput(ctx, FF_out);

    this->spec_global_history.push(bool(out)); // Update the speculative history.

    return bool(out);
  }

  void backwardsPass(ForwardContext& ctx, float y_true, float learning_rate){
    
    float p = ctx.out;
    float logit = ctx.logit;

    /*
      Output (pooling) backwards
    */

    // Compute inital gradient from loss (dL / dLogit)
    float dL_dLogit = p - y_true;

    // Gradients for output layer weights and bias
    FixedVector<float> grad_w_out = ctx.pooled * dL_dLogit; // constribution of each hidden feature, [d_model]
    float grad_b_out = dL_dLogit;

    // Update output layer parameters (SGD update)
    for (size_t i = 0; i < w_out.size(); ++i){
      w_out[i] -= learning_rate * grad_b_out;
    }
    b_out -= learning_rate * grad_b_out;

    // Gradient of the pooled input vector (for further propagation)
    FixedVector<float> grad_pooled = FixedVector<float>(d_model, 0.0f);
    for (size_t i = 0; i < grad_pooled.size(); ++i){
      grad_pooled[i] = w_out[i] * dL_dLogit;
    }
    // Grad_pooled == dL / dh_pooled, the gradient of the loss with respect to the final hidden states


    /*
      Feed Forward Backwards
    */
    // Gradient matrix for the input sequence
    FixedVector<FixedVector<float>> grad_ffn_out = FixedVector(sequence_len, FixedVector())


    ctx.d_logits = (y_pred - y_true) / (y_pred * (1 - y_pred)); // dL/dy * dy/dz
    
    // Backprop through final pooling layer
    for (size_t i = 0; i < d_model; i++) {
        ctx.d_pooled[i] = ctx.d_logits * w_out[i];
    }
    
    // Backprop through final layer norm
    for (size_t i = 0; i < d_model; i++) {
        ctx.d_pooled[i] *= ctx.layernorm_gamma[i];
    }
    
    // Distribute gradients back to sequence level
    for (size_t i = 0; i < sequence_len; i++) {
        for (size_t j = 0; j < d_model; j++) {
            ctx.d_ff_out[i][j] = ctx.d_pooled[j] / sequence_len;
        }
    }
    
    // Backprop through final layer norm (ff_residual)
    for (size_t i = 0; i < ctx.ff_hidden.size(); i++) {
        for (size_t j = 0; j < d_model; j++) {
            ctx.d_ff_hidden[i][j] = ctx.d_ff_out[i][j];
            if (ctx.ff_hidden[i][j] <= 0) {
                ctx.d_ff_hidden[i][j] = 0; // ReLU derivative
            }
        }
    }
    
    // Backprop through first FF layer
    for (size_t i = 0; i < ctx.ff_pre_activation.size(); i++) {
        for (size_t j = 0; j < d_ff; j++) {
            ctx.d_ff_pre_activation[i][j] = ctx.d_ff_hidden[i][j] * (ctx.ff_pre_activation[i][j] > 0);
        }
    }
    
    // Compute gradients for w_ff1 and b_ff1
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            w_ff1[i][j] -= learning_rate * ctx.d_ff_pre_activation[0][j] * ctx.ff_residual[0][i];
        }
        b_ff1[i] -= learning_rate * ctx.d_ff_pre_activation[0][i];
    }
    
    // Backprop through Attention Layer
    ctx.d_attention_out = ctx.d_ff_out;
    
    // Backprop through multihead attention
    for (size_t i = 0; i < sequence_len; i++) {
        for (size_t j = 0; j < d_model; j++) {
            for (size_t k = 0; k < sequence_len; k++) {
                ctx.d_softmax_attn[i][k] += ctx.d_attention_out[i][j] * ctx.V[k][j];
            }
        }
    }
    
    // Compute d_Q, d_K, d_V
    for (size_t i = 0; i < sequence_len; i++) {
        for (size_t j = 0; j < d_model; j++) {
            for (size_t k = 0; k < sequence_len; k++) {
                ctx.d_Q[i][j] += ctx.d_softmax_attn[i][k] * ctx.K[k][j];
                ctx.d_K[i][j] += ctx.d_softmax_attn[k][i] * ctx.Q[k][j];
                ctx.d_V[i][j] += ctx.d_attention_out[i][j] * ctx.softmax_attn[i][j];
            }
        }
    }
    
    // Update Attention Weights
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < d_k; j++) {
            w_q[i][j] -= learning_rate * ctx.d_Q[0][j];
            w_k[i][j] -= learning_rate * ctx.d_K[0][j];
            w_v[i][j] -= learning_rate * ctx.d_V[0][j];
        }
    }
  }
};


//--------------------------------------------------
// Note, the actual transformer uses it's internal 
// sequence_len / d_model provided via spec.json
// 
// However, because we're using bitsets we need the length
// to be defined at compile time.
//--------------------------------------------------
/* DEPRECATED: Moved inside the transformer itself. Alleviating bitset's constantexpr requirement; however, still uncertain if this is the best approach. */
//constexpr std::size_t HISTORY_LENGTH = 24; // We can adjust. Defaulting to current perceptron's length for closer 1-to-1 comparison

//-------------------------------------------
// Map to the O3_CPU instance
//-------------------------------------------
std::map<O3_CPU*, Transformer> predictors; // One transformer for every core
//-------------------------------------------
// Save the speculative global history.
// This stores the branch taken / not taken
// This is used to compare against the actual prediction results.
// Note: This is a map because we can do Multi-Core, each O3_CPU instance being a single core with it's own history.
//-------------------------------------------
/* DEPRECATED: Moved inside the transformer itself. Alleviating bitset's constantexpr requirement; however, still uncertain if this is the best approach. */
//std::map<O3_CPU*, std::bitset<HISTORY_LENGTH>> global_history; 

} // namespace




void O3_CPU::initialize_branch_predictor() {
  ::predictors.emplace(this, "spec.json");
}

uint8_t O3_CPU::predict_branch(uint64_t ip) { 

  // Get the transformers prediction. It will handle it's own sequence history. 
  bool prediction = ::predictors.at(this).predict(ip);
  //fmt::println("Transformer predicted: {} for ip {}\n", prediction, ip);

  return prediction;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type) {
  
  // We need this, but need to rework it entirely.
  // fmt::println(
  //   "Comparing previous prediction results: ip: {}\ttype: {}\tpredicted: {}\tcorrect: {}",
  //    ip,
  //    branch_type, 
  //    ::predictors.at(this).get_prediction(0),
  //    taken
  // );
  // ::predictors.at(this).global_history.push(bool(taken)); // I hate this.

  // if(prediction != taken){
  //   // Do back prop
  // }
  return;
}