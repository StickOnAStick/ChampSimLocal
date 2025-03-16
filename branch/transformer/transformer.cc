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
#include <string.h>

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

      Q: Query Matrix (sequence_len x d_k)  = XW^Q
      K: Key Matrix   (sequence_len x d_k)  = XW^K
      V: Value Matrix (sequence_len x d_v)  = XW^V
      M: Mask Matrix  (sequence_len x sequence_len)

      d_k: Dimensionality of key/query == d_model / num_heads(h)
      sequence_len: Sequence length of our model

      Multi-Headed:
          MultiHead(Q,K,V) = Concat(head1, head2,...,head_h)W^O

          head_i = Attention( Q*W^Q_i, K*W^K_i, V*W^V_i )
          W^(Q,K,V)_i: Learnable weight matricies for each head (d_model x d_k)
          W^O: Output projection Weight matrix ((h * d_v) x d_model)

          Output = (sequence_len x d_model)
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

      Q, K, V = sequence_len * w_q,k,v
      ---------------------------------------
      Dimensions:
      - sequence_history: [sequence_len, d_model]
      - w_q, w_v, w_k: [d_model, d_model] [d_model, d_model] [d_model, d_model]
      - Q, K, V:  [sequence_len, d_q] [sequence_len, d_v]
      
    */
    
    ctx.Q = USE_CUDA ? FixedVectorMath::dotCuda(sequence_history, w_q) : FixedVectorMath::dot(sequence_history, w_q);
    ctx.K = USE_CUDA ? FixedVectorMath::dotCuda(sequence_history, w_k) : FixedVectorMath::dot(sequence_history, w_k);
    ctx.V = USE_CUDA ? FixedVectorMath::dotCuda(sequence_history, w_v) : FixedVectorMath::dot(sequence_history, w_v);
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
          float scale = 1.0f / std::sqrt(static_cast<float>(d_head)); // Enhance numerical stability.
          attention_scores[i][j] = score * scale;

          // QK^T / sqrt(d_head) + M
          if (use_mask){
            attention_scores[i][j] += mask[i][j]; // Either adds 0 or -∞
          }
        }
      }
      // Softmax the attention scores (row-wise)
      FixedVectorMath::softmax(attention_scores);
      ctx.softmax_attn[head] = attention_scores;

      // Compute head_out = attention_scores * V_head
      // [sequence_len, d_head]
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
      We now have concat(head_0, head_1, ... head_h) of dim [sequence_len, d_model]
      Now apply the output weight matrix
      
      output = concat(head_0...head_h)*W_O 

      where W_O is of dim [d_model, d_model]
    */
    ctx.attn_out = USE_CUDA ? FixedVectorMath::dotCuda(attention_out, w_o) : FixedVectorMath::dot(attention_out, w_o);
    return ctx.attn_out;
  }

  FixedVector<FixedVector<float>> FFLayer(ForwardContext& ctx, FixedVector<FixedVector<float>>& input) override {
    /*
      FFN(x) = ReLU(0, xW_1 + b_1)W_2 + b_2

      Matrix sizes:
        - in/out: [sequence_len, d_model]
        - w_ff1: [d_model, d_ff]
        - b_ff1: [d_ff]
        - w_ff2: [d_ff, d_model]
        - b_ff2: [d_model]

        NOTE: The output is of size [sequence_len, d_model], not the final prediction d_out [1]
    */

    // --------------------------------------------------
    // 1) hidden = input * w_ff1 + b_ff1
    //    => hidden: shape [sequence_len, d_ff]
    // --------------------------------------------------

    FixedVector<FixedVector<float>> hidden = USE_CUDA ? FixedVectorMath::linearCuda(input, w_ff1, b_ff1) : FixedVectorMath::linear(input, w_ff1, b_ff1);
    //---------------------------------------------------
    // 2.) Relu in place
    //---------------------------------------------------
    FixedVectorMath::relu(hidden);
    ctx.ff_intermediate = hidden;
    //---------------------------------------------------
    // 3.) output = hidden * w_ff2 + b_ff2
    //     => output: shapre [sequence_len, d_model]
    //---------------------------------------------------
    FixedVector<FixedVector<float>> output = USE_CUDA ? FixedVectorMath::linearCuda(hidden, w_ff2, b_ff2) : FixedVectorMath::linear(hidden, w_ff2, b_ff2);
    return output;
  }

  float pooledOutput(ForwardContext& ctx, FixedVector<FixedVector<float>>& input) override {
    /*
      Reduce the entire transformer state to a single prediction. 

      input = [sequence_len, d_model]
      out   = [1]

      pooled = 1/sequence_len Σ h_i   // 1 to sequence_len
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
    for(size_t i = 0; i < pooled.size(); ++i){        // 1/sequence_len
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
    ctx.ip = ip;
    this->fixed_posEncoding(ip);
    ctx.input = this->sequence_history; // Input is after we added the new instruction.
    /*
      Masked Multi-Headed Attention
    */
    //std::cout << "Begin attn" << std::endl;
    FixedVector<FixedVector<float>> MMA_out = this->MALayer(ctx, true);
    FixedVectorMath::add(MMA_out, this->sequence_history); // Result stored in MMA_Out
    ctx.attn_post_residual = MMA_out;
    USE_CUDA ? FixedVectorMath::normalizeCuda(MMA_out, ctx.attn_mean_i, ctx.attn_var_i, this->epsilon) : FixedVectorMath::normalize(MMA_out, ctx.attn_mean_i, ctx.attn_var_i, this->epsilon);
    ctx.ffnInput = MMA_out;
    //std::cout << "End attn" << std::endl;

    /*
      Feed-Forward Layer
    */
    //std::cout << "Begin ff" << std::endl;
    FixedVector<FixedVector<float>> FF_out = this->FFLayer(ctx, MMA_out);
    FixedVectorMath::add(FF_out, MMA_out);
    ctx.ff_post_residual = FF_out;
    USE_CUDA ? FixedVectorMath::normalizeCuda(FF_out, ctx.ff_mean_i, ctx.ff_var_i, this->epsilon) : FixedVectorMath::normalize(FF_out, ctx.ff_mean_i, ctx.ff_var_i, this->epsilon);
    ctx.ff_normed = FF_out;
    //std::cout << "End ff" << std::endl;


    float out = this->pooledOutput(ctx, FF_out);

    return out > 0.5;
  }

  void backwardsPass(ForwardContext& ctx, float y_true, float learning_rate = 0.0001f){
    
    if(std::isnan(learning_rate) || std::isinf(learning_rate))
      throw std::runtime_error("Invalid learning rate");

    // Layer Norm. Backwards pass helper lambda function
    auto layerNormBackward = [&](  // *[&] is a lambda function which captures all variables out-of-scope by reference
      const FixedVector<FixedVector<float>>& x,  // LN Input
      const FixedVector<FixedVector<float>>& y,  // LN Output
      const FixedVector<float>& mean_vec,        // The mean of the i_th row of the sequence [d_model]
      const FixedVector<float>& var_vec,         // Variance of the i_th row
      const FixedVector<FixedVector<float>>& dL_dy, // Grad wrt LN Out
      FixedVector<FixedVector<float>>& dL_dx,       // to fill
      float e = 1e-5f
    ){
      //std::cout<< "Layer Norm Back Prop beginning...\n" << std::endl;
      for(int i = 0; i < this->sequence_len; i++){
        float mean_i = mean_vec[i];
        float var_i = var_vec[i];
        float inv_std = 1.0f / std::sqrt(var_i + e);     
        float inv_var = inv_std * inv_std;          
        //fmt::print("inv_var {:.6f}, inv_std: {:.6f}\n", inv_var, inv_std);


        float sum_dL_dy = 0.0f;
        float sum_dL_dy_times_xm = 0.0f; // sum( dL/dy_k * (x_k - mean) )
        for (int k = 0; k < d_model; k++) {
            float x_mean_diff = x[i][k] - mean_i;
            sum_dL_dy += dL_dy[i][k];
            sum_dL_dy_times_xm += dL_dy[i][k] * x_mean_diff;
            //fmt::print("sum_dL_dy_times_xm {:.6f}, sum_dL_dy: {:.6f}\n", sum_dL_dy_times_xm, sum_dL_dy);
        }
        for (int k = 0; k < d_model; k++) {
            float x_hat = (x[i][k] - mean_i) * inv_std;            
            dL_dx[i][k] = inv_std * (dL_dy[i][k] - sum_dL_dy / d_model - x_hat * sum_dL_dy_times_xm / d_model);
        }
      }
      //std::cout<< "Layer norm backprop complete!\n" << std::endl;
    };
    /****************************************************
     * 0) Derivative of BCE Loss wrt logit
     ****************************************************/
    float out_clamped = std::max(std::min(ctx.out, 1.0f - 1e-6f), 1e-6f);
    float dL_dlogit = (out_clamped - y_true);  // out = sigmoid(logit)

    /****************************************************
     * 1) Backprop from logit -> W_logit, b_logit, pooled
     ****************************************************/
    // Suppose we have class-member: W_logit (size d_model), b_logit (scalar)
    // We'll accumulate grads in local arrays:
    //std::cout<< "1. Begin Logit backprop\n" << std::endl;
    FixedVector<float> W_logit_grad = FixedVector<float>(d_model, 0.0f); 
    float b_logit_grad = 0.0f;

    // dL/dW_logit[k] = dL/dlogit * pooled[k]
    // dL/db_logit    = dL/dlogit
    // dL/dpooled[k]  = dL/dlogit * W_logit[k]
    FixedVector<float> dL_dpooled(d_model, 0.0f);
    for(int k = 0; k < d_model; k++){
        // Actually we must correct the above line:
        //   The gradient wrt W_logit is (dL/dlogit * pooled[k])
        //   So:
        W_logit_grad[k] = dL_dlogit * ctx.pooled[k];
    }
    b_logit_grad = dL_dlogit;

    for(int k = 0; k < d_model; k++){
        dL_dpooled[k] = dL_dlogit * this->w_out[k];
    }
    //std::cout<< "1. End Logit backprop\n" << std::endl;

    /****************************************************
     * 2) Backprop from pooled -> ff_normed
     *    pooled[k] = average of ff_normed[i][k] over sequence_len
     ****************************************************/
    // So dL/dff_normed[i][k] += dL/dpooled[k] / sequence_len
    //std::cout<< "2. Begin Pooled backprop\n" << std::endl;
    FixedVector<FixedVector<float>> dL_dFF_normed(sequence_len, FixedVector<float>(d_model, 0.0f));
    for(int i = 0; i < sequence_len; i++){
        for(int k = 0; k < d_model; k++){
            dL_dFF_normed[i][k] = dL_dpooled[k] / (float)sequence_len;
        }
    }
    //std::cout<< "2. End Pooled backprop\n" << std::endl;
    
    /****************************************************
     * 3) LayerNorm backward on the FF block
     *    FF block output is ctx.ff_normed => LN output
     *    LN input was ctx.ff_post_residual => (ff_out + attn_normed)
     ****************************************************/
    //std::cout<< "3. Begin FF Layer Norm backprop\n" << std::endl;
    FixedVector<FixedVector<float>> dL_dFF_post_resid(sequence_len, FixedVector<float>(d_model, 0.0f));
    // LN backward:
    layerNormBackward(
        ctx.ff_post_residual,   // LN input
        ctx.ff_normed,          // LN output
        ctx.ff_mean_i,          // means
        ctx.ff_var_i,           // vars
        dL_dFF_normed,          // gradient from above
        dL_dFF_post_resid,      // result => dL/d LN input
        this->epsilon
    );

    /****************************************************
     * 3a) Split gradient among (ff_out) and (attn_normed) residual
     ****************************************************/
    // forward did: ff_post_residual = ff_out + attn_normed
    // so dL/dff_out_raw[i][k] += dL_dFF_post_resid[i][k]
    //    dL/dattn_normed[i][k] += dL_dFF_post_resid[i][k]
    // We haven't defined dL_dff_out_raw or dL_dAttn_normed yet, so let's do it:
    FixedVector<FixedVector<float>> dL_dFF_out_raw(sequence_len, FixedVector<float>(d_model, 0.0f));
    FixedVector<FixedVector<float>> dL_dAttn_normed(sequence_len, FixedVector<float>(d_model, 0.0f));

    for(int i = 0; i < sequence_len; i++){
        for(int k = 0; k < d_model; k++){
            float g = dL_dFF_post_resid[i][k];
            dL_dFF_out_raw[i][k]    += g;  // to feed-forward sublayer out
            dL_dAttn_normed[i][k]   += g;  // to MHA LN output
        }
    }
    //std::cout<< "3. End FF Layer Norm backprop\n" << std::endl;

    /****************************************************
     * 4) Backprop the feed-forward sublayer
     *    From dL_dFF_out_raw => (linear -> ReLU -> linear).
     *    We have w_ff2, b_ff2, then w_ff1, b_ff1, etc.
     *    We also have ff_hidden in ctx (pre-RELU).
     ****************************************************/
    // We'll keep local gradients for w_ff2, b_ff2, w_ff1, b_ff1
    //std::cout<< "4. Begin FF backprop\n" << std::endl;

    FixedVector<FixedVector<float>> w_ff1_grad = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_ff, 0.0f));
    FixedVector<float> b_ff1_grad = FixedVector<float>(d_ff, 0.0f);
    FixedVector<FixedVector<float>> w_ff2_grad = FixedVector<FixedVector<float>>(d_ff, FixedVector<float>(d_model, 0.0f));
    FixedVector<float> b_ff2_grad = FixedVector<float>(d_model, 0.0f);

    // We'll also need gradient wrt ff_hidden (post-relu):
    FixedVector<FixedVector<float>> dL_dff_hidden(sequence_len, FixedVector<float>(d_ff, 0.0f));

    // The final FF layer was: ff_out = ff_hidden * w_ff2 + b_ff2
    // dL/dff_out_raw = dL_dFF_out_raw
    for(int i = 0; i < sequence_len; i++){
        for(int j = 0; j < d_model; j++){
            float grad_out_ij = dL_dFF_out_raw[i][j];
            // b_ff2
            b_ff2_grad[j] += grad_out_ij;

            // w_ff2 => shape [d_model, d_model]
            // ff_hidden[i][k] * w_ff2[k][j]
            for(int ff_dim = 0; ff_dim < d_ff; ff_dim++){
                w_ff2_grad[ff_dim][j] += grad_out_ij * ctx.ff_intermediate[i][ff_dim];
            }
        }
    }

    // Hidden Layer gradient
    // dL/dff_hidden = sum_j( dL/dff_out[i][j] * w_ff2[k][j] ) for each k
    // Then ReLU( ff_hidden ), so we must gate with ReLU derivative
    for(int i = 0; i < sequence_len; i++){
        for(int ff_dim = 0; ff_dim < d_ff; ff_dim++){
            float sum_grad = 0.0f;
            for(int j = 0; j < d_model; j++){
                sum_grad += dL_dFF_out_raw[i][j] * this->w_ff2[ff_dim][j];
            }
            // Now apply ReLU gate. In forward pass, we did in-place ReLU on ff_hidden
            // If ff_hidden[i][k] <= 0 => gradient is 0
            // We stored the post-activation in ctx.ff_intermediate too, so let's check that:
            if(ctx.ff_intermediate[i][ff_dim] <= 0.0f) {
                sum_grad = 0.0f;
            }
            dL_dff_hidden[i][ff_dim] = sum_grad;
        }
    }

    // Now backprop the first FF layer: ff_hidden = ffnInput * w_ff1 + b_ff1
    // with ffnInput = ctx.ffnInput ( = attn_normed in forward code).
    FixedVector<FixedVector<float>> dL_dFFN_input(sequence_len, FixedVector<float>(d_model, 0.0f));

    for(int i = 0; i < sequence_len; i++){
        for(int ff_dim = 0; ff_dim < d_ff; ff_dim++){
            float grad_hidden_ij = dL_dff_hidden[i][ff_dim];
            b_ff1_grad[ff_dim] += grad_hidden_ij;

            for(int k = 0; k < d_model; k++){
                w_ff1_grad[k][ff_dim] += grad_hidden_ij * ctx.ffnInput[i][k];
                dL_dFFN_input[i][k] += grad_hidden_ij * this->w_ff1[k][ff_dim];
            }
        }
    }

    // That gradient flows into dL_dAttn_normed as well (the FF input).
    // But we’ve already been adding to dL_dAttn_normed from the residual,
    // so we now combine:
    for(int i = 0; i < sequence_len; i++){
        for(int k = 0; k < d_model; k++){
            dL_dAttn_normed[i][k] += dL_dFFN_input[i][k];
        }
    }
    //std::cout<< "4. End FF backprop\n" << std::endl;

    /****************************************************
     * 5) Now handle LN backward for the MHA output
     *    attn_normed = LN( attn_post_residual )
     *    attn_post_residual = attn_out + original_input_seq
     *    The "original_input_seq" for a single-layer decoder might be
     *    the embedding or the input to the block. We'll call it dL_dInput.
     ****************************************************/
    //std::cout<< "5. Begin Layer Norm MHA backprop\n" << std::endl;

    FixedVector<FixedVector<float>> dL_dAttn_post_resid(sequence_len, FixedVector<float>(d_model, 0.0f));

    layerNormBackward(
        ctx.attn_post_residual,  // LN input
        ctx.ffnInput,         // LN output
        ctx.attn_mean_i,
        ctx.attn_var_i,
        dL_dAttn_normed,         // grad from feed-forward residual
        dL_dAttn_post_resid,     // result => gradient w.r.t. LN input
        this->epsilon
    );

    // Now that splits into attn_out + input_sequence
    // We'll define dL_dAttn_out and dL_dInputSeq
    // If your original input is "ctx.input" or "sequence_history," adapt accordingly:
    FixedVector<FixedVector<float>> dL_dAttn_out(sequence_len, FixedVector<float>(d_model, 0.0f));
    FixedVector<FixedVector<float>> dL_dInputSeq(sequence_len, FixedVector<float>(d_model, 0.0f));

    for(int i = 0; i < sequence_len; i++){
        for(int k = 0; k < d_model; k++){
            float g = dL_dAttn_post_resid[i][k];
            dL_dAttn_out[i][k] += g;
            dL_dInputSeq[i][k] += g;  // gradient w.r.t. the original input to MHA sub-layer
        }
    }
    //std::cout<< "5. End Layer Norm MHA backprop\n" << std::endl;


    /****************************************************
     * 6) Backprop multi-head attention
     *    final MHA out: attention_out * w_o => attn_out
     *    So we do the matrix multiply backward first:
    ****************************************************/
    //std::cout<< "6. Begin MHA backprop\n" << std::endl;
    FixedVector<FixedVector<float>> w_o_grad(d_model, FixedVector<float>(d_model, 0.0f));

    // dL_dAttentionOut is shape [sequence_len, d_model]
    FixedVector<FixedVector<float>> dL_dAttentionOut(sequence_len, FixedVector<float>(d_model, 0.0f));

    // attn_out[i][j] = sum_{k} attention_out[i][k] * w_o[k][j]
    for(int i = 0; i < sequence_len; i++){
        for(int j = 0; j < d_model; j++){
            float grad_attn_out_ij = dL_dAttn_out[i][j];
            // accumulate w_o grad
            for(int k = 0; k < d_model; k++){
                w_o_grad[k][j] += grad_attn_out_ij * ctx.attn_out[i][k];
            }
            // pass back to attention_out
            for(int k = 0; k < d_model; k++){
                dL_dAttentionOut[i][k] += grad_attn_out_ij * this->w_o[k][j];
            }
        }
    }
    //std::cout<< "6. End MHA backprop\n" << std::endl;

    /****************************************************
     * 7) Now handle each MHA head:
     *    We have Q,K,V in ctx.Q, ctx.K, ctx.V
     *    The final was attention_out = concat(head_0..head_n)
    ****************************************************/
    //std::cout<< "7. Begin MHA heads backprop\n" << std::endl;
    
    FixedVector<FixedVector<float>> dL_dQ(sequence_len, FixedVector<float>(d_model, 0.0f));
    FixedVector<FixedVector<float>> dL_dK(sequence_len, FixedVector<float>(d_model, 0.0f));
    FixedVector<FixedVector<float>> dL_dV(sequence_len, FixedVector<float>(d_model, 0.0f));

    int d_head = d_model / this->num_ma_heads;

    for(int head = 0; head < this->num_ma_heads; head++){
        // slice dL_dHead_out from dL_dAttentionOut
        FixedVector<FixedVector<float>> dL_dHead_out(sequence_len, FixedVector<float>(d_head, 0.0f));
        for(int i = 0; i < sequence_len; i++){
            for(int d = 0; d < d_head; d++){
                dL_dHead_out[i][d] = dL_dAttentionOut[i][head*d_head + d];
            }
        }

        // We had head_out[i][d] = sum_j( softmax_attn[i][j] * V_head[j][d] )
        // => dL/dsoftmax_attn[i][j], dL/dV_head[j][d]

        // Build local dL_dV_head, dL_dSoftmax
        FixedVector<FixedVector<float>> dL_dV_head(sequence_len, FixedVector<float>(d_head, 0.0f));
        FixedVector<FixedVector<float>> dL_dSoftmax(sequence_len, FixedVector<float>(sequence_len, 0.0f));

        for(int i = 0; i < sequence_len; i++){
            for(int d = 0; d < d_head; d++){
                float grad_out = dL_dHead_out[i][d];
                for(int j = 0; j < sequence_len; j++){
                    float attn_ij = ctx.softmax_attn[head][i][j];
                    // accumulate dL/dV_head
                    dL_dV_head[j][d] += grad_out * attn_ij;
                }
            }
        }
        // also accumulate dL/dSoftmax from the product with V_head
        for(int i = 0; i < sequence_len; i++){
            for(int j = 0; j < sequence_len; j++){
                float sum_d = 0.0f;
                for(int d = 0; d < d_head; d++){
                    float grad_out = dL_dHead_out[i][d];
                    float v_jd     = ctx.V[j][head*d_head + d];
                    sum_d += grad_out * v_jd;
                }
                dL_dSoftmax[i][j] += sum_d;
            }
        }

        // Now backprop softmax => scores. We have a row-wise softmax.
        // standard formula:
        //   dL/dscores[i][j] = softmax_attn[i][j] * ( dL/dSoftmax[i][j] 
        //     - sum_{p} dL/dSoftmax[i][p] * softmax_attn[i][p] )
        FixedVector<FixedVector<float>> dL_dScores(sequence_len, FixedVector<float>(sequence_len, 0.0f));
        for(int i = 0; i < sequence_len; i++){
            // row sum
            float rowDot = 0.0f;
            for(int p = 0; p < sequence_len; p++){
                rowDot += dL_dSoftmax[i][p] * ctx.softmax_attn[head][i][p];
            }
            for(int j = 0; j < sequence_len; j++){
                float grad_softmax_ij = dL_dSoftmax[i][j];
                float sm_ij           = ctx.softmax_attn[head][i][j];
                // If masked, forward pass had sm_ij = 0 for that j => gradient ~ 0
                dL_dScores[i][j] = sm_ij * (grad_softmax_ij - rowDot);
            }
        }

        // scores[i][j] = (Q_head[i] dot K_head[j]) / sqrt(d_head) + mask
        // => backprop to Q_head, K_head
        float scale = 1.0f / std::sqrt((float)d_head);
        FixedVector<FixedVector<float>> dL_dQ_head(sequence_len, FixedVector<float>(d_head, 0.0f));
        FixedVector<FixedVector<float>> dL_dK_head(sequence_len, FixedVector<float>(d_head, 0.0f));

        for(int i = 0; i < sequence_len; i++){
            for(int j = 0; j < sequence_len; j++){
                float grad_score_ij = dL_dScores[i][j];
                // If masked, forward pass effectively sets grad to 0
                // We'll trust that softmax + mask => no gradient flows
                // so we won't do an explicit "if mask" check here.
                for(int d = 0; d < d_head; d++){
                    float q_id = ctx.Q[i][head*d_head + d];
                    float k_jd = ctx.K[j][head*d_head + d];
                    dL_dQ_head[i][d] += grad_score_ij * scale * k_jd;
                    dL_dK_head[j][d] += grad_score_ij * scale * q_id;
                }
            }
        }

        // We already have dL_dV_head from above. Now integrate them back:
        for(int i = 0; i < sequence_len; i++){
            for(int d = 0; d < d_head; d++){
                dL_dQ[i][head*d_head + d] += dL_dQ_head[i][d];
                dL_dV[i][head*d_head + d] += dL_dV_head[i][d];
            }
        }
        for(int j = 0; j < sequence_len; j++){
            for(int d = 0; d < d_head; d++){
                dL_dK[j][head*d_head + d] += dL_dK_head[j][d];
            }
        }
    }
    //std::cout<< "7. End MHA heads backprop\n" << std::endl;

    /****************************************************
     * 8) Backprop Q,K,V => seq_history => w_q, w_k, w_v
     ****************************************************/
    //std::cout<< "8. Begin Q,K,V backprop\n" << std::endl;

    FixedVector<FixedVector<float>> w_q_grad(d_model, FixedVector<float>(d_model, 0.0f));
    FixedVector<FixedVector<float>> w_k_grad(d_model, FixedVector<float>(d_model, 0.0f));
    FixedVector<FixedVector<float>> w_v_grad(d_model, FixedVector<float>(d_model, 0.0f));

    // We'll also accumulate gradient wrt the original input (which we might add to dL_dInputSeq)
    // but let's define local:
    FixedVector<FixedVector<float>> dL_dSeqHistory(sequence_len, FixedVector<float>(d_model, 0.0f));

    // Q[i][j] = sum_p( input_seq[i][p] * w_q[p][j] )
    // => w_q_grad[p][j] += sum_i( dL_dQ[i][j] * input_seq[i][p] )
    // => dL_dSeqHistory[i][p] += dL_dQ[i][j] * w_q[p][j]
    for(int i = 0; i < sequence_len; i++){
        for(int j = 0; j < d_model; j++){
            float grad_Qij = dL_dQ[i][j];
            for(int p = 0; p < d_model; p++){
                w_q_grad[p][j] += grad_Qij * ctx.input[i][p]; // assuming ctx.input is the "sequence_history"
                dL_dSeqHistory[i][p] += grad_Qij * this->w_q[p][j];
            }
        }
    }
    // K
    for(int i = 0; i < sequence_len; i++){
        for(int j = 0; j < d_model; j++){
            float grad_Kij = dL_dK[i][j];
            for(int p = 0; p < d_model; p++){
                w_k_grad[p][j] += grad_Kij * ctx.input[i][p];
                dL_dSeqHistory[i][p] += grad_Kij * this->w_k[p][j];
            }
        }
    }
    // V
    for(int i = 0; i < sequence_len; i++){
        for(int j = 0; j < d_model; j++){
            float grad_Vij = dL_dV[i][j];
            for(int p = 0; p < d_model; p++){
                w_v_grad[p][j] += grad_Vij * ctx.input[i][p];
                dL_dSeqHistory[i][p] += grad_Vij * this->w_v[p][j];
            }
        }
    }

    // Finally, we also have dL_dInputSeq from the LN residual side,
    // so let's add dL_dSeqHistory into that:
    // i.e. total gradient wrt the actual input is dL_dInputSeq + dL_dSeqHistory.
    for(int i = 0; i < sequence_len; i++){
        for(int p = 0; p < d_model; p++){
            dL_dInputSeq[i][p] += dL_dSeqHistory[i][p];
        }
    }
    //std::cout<< "8. End Q,K,V backprop\n" << std::endl;

    /****************************************************
     * 9) Now we apply vanilla SGD updates to:
     *    w_q, w_k, w_v, w_o,
     *    w_ff1, b_ff1, w_ff2, b_ff2,
     *    W_logit, b_logit,
     *    (and if LN had gamma,beta, we’d update them, but we don't here).
     ****************************************************/
    //std::cout<< "9. Begin Weight / Bias update\n" << std::endl;

    this->adam_step += 1;
    float alpha_t = this->lr
                    * std::sqrt(1.0f - std::pow(beta2, this->adam_step))
                    / (1.0f - std::pow(beta1, this->adam_step));
    
    // ADAM update for w_q, w_k, w_v, w_o
    for(int i = 0; i < d_model; i++){
      for(int j = 0; j < d_model; j++){
        // old moments
        float q_m_old = w_q_m[i][j];
        float q_v_old = w_q_v[i][j];
        float k_m_old = w_k_m[i][j];
        float k_v_old = w_k_v[i][j];        
        float v_m_old = w_v_m[i][j];
        float v_v_old = w_v_v[i][j];
        float o_m_old = w_o_m[i][j];
        float o_v_old = w_o_v[i][j];

        float q_g = w_q_grad[i][j];
        float k_g = w_k_grad[i][j];
        float v_g = w_v_grad[i][j];
        float o_g = w_o_grad[i][j];
        
        // new moments
        float q_m_new = beta1 * q_m_old + (1.0f - beta1) * q_g;
        float q_v_new = beta2 * q_v_old + (1.0f - beta2) * (q_g*q_g);
        float k_m_new = beta1 * k_m_old + (1.0f - beta1) * k_g;
        float k_v_new = beta2 * k_v_old + (1.0f - beta2) * (k_g*k_g);
        float v_m_new = beta1 * v_m_old + (1.0f - beta1) * v_g;
        float v_v_new = beta2 * v_v_old + (1.0f - beta2) * (v_g*v_g);
        float o_m_new = beta1 * o_m_old + (1.0f - beta1) * o_g;
        float o_v_new = beta2 * o_v_old + (1.0f - beta2) * (o_g * o_g);


        w_q_m[i][j] = q_m_new;
        w_q_v[i][j] = q_v_new;
        w_k_m[i][j] = k_m_new;
        w_k_v[i][j] = k_v_new;
        w_v_m[i][j] = v_m_new;
        w_v_v[i][j] = v_v_new;
        w_o_m[i][j] = o_m_new;
        w_o_v[i][j] = o_v_new;

        w_q[i][j] -= alpha_t * (q_m_new / (std::sqrt(q_v_new) + this->epsilon));
        w_k[i][j] -= alpha_t * (k_m_new / (std::sqrt(k_v_new) + this->epsilon));
        w_v[i][j] -= alpha_t * (v_m_new / (std::sqrt(v_v_new) + this->epsilon));
        w_o[i][j] -= alpha_t * (o_m_new / (std::sqrt(o_v_new) + this->epsilon));
      }
    }           
    
    // ADAM update for w_ff1
    for(int i = 0; i < d_model; i++){
      for(int j = 0; j < d_ff; j++){
            float m_old = w_ff1_m[i][j];
            float v_old = w_ff1_v[i][j];

            float g = w_ff1_grad[i][j];

            float m_new = beta1 * m_old + (1.0f - beta1) * g;
            float v_new = beta2 * v_old + (1.0f - beta2) * (g*g);

            w_ff1_m[i][j] = m_new;
            w_ff1_v[i][j] = v_new;

            w_ff1[i][j] -= alpha_t * (m_new / (std::sqrt(v_new) + epsilon));
      }
    }

    // Adam update for w_ff2
    for(int i = 0; i < d_ff; i++){
      for(int j = 0; j < d_model; j++){
        float m_old = w_ff2_m[i][j];
            float v_old = w_ff2_v[i][j];

            float g = w_ff2_grad[i][j];

            float m_new = beta1 * m_old + (1.0f - beta1) * g;
            float v_new = beta2 * v_old + (1.0f - beta2) * (g*g);

            w_ff2_m[i][j] = m_new;
            w_ff2_v[i][j] = v_new;

            w_ff2[i][j] -= alpha_t * (m_new / (std::sqrt(v_new) + epsilon));
      }
    }

    // Adam update for b_ff1
    for(int i = 0; i < d_ff; i++){
      float m_old = b_ff1_m[i];
      float v_old = b_ff1_v[i];

      float g = b_ff1_grad[i];

      float m_new = beta1 * m_old + (1.0f - beta1) * g;
      float v_new = beta2 * v_old + (1.0f - beta2) * (g*g);

      b_ff1_m[i] = m_new;
      b_ff1_v[i] = v_new;

      b_ff1[i] -= alpha_t * (m_new / (std::sqrt(v_new) + epsilon));
    }

    // Adam update for b_ff2
    for(int j = 0; j < d_model; j++){
      float m_old = b_ff2_m[j];
      float v_old = b_ff2_v[j];

      float g = b_ff2_grad[j];

      float m_new = beta1 * m_old + (1.0f - beta1) * g;
      float v_new = beta2 * v_old + (1.0f - beta2) * (g*g);

      b_ff2_m[j] = m_new;
      b_ff2_v[j] = v_new;

      b_ff2[j] -= alpha_t * (m_new / (std::sqrt(v_new) + epsilon));
    }

    // Adam update for w_out (1D)
    for(int k = 0; k < d_model; k++){
      float m_old = w_out_m[k];
      float v_old = w_out_v[k];

      float g = W_logit_grad[k];

      float m_new = beta1 * m_old + (1.0f - beta1) * g;
      float v_new = beta2 * v_old + (1.0f - beta2) * (g*g);

      w_out_m[k] = m_new;
      w_out_v[k] = v_new;

      w_out[k] -= alpha_t * (m_new / (std::sqrt(v_new) + epsilon));
    }

    // Adam update for b_out
    {
      float m_old = b_out_m;
      float v_old = b_out_v;

      float g = b_logit_grad;

      float m_new = beta1 * m_old + (1.0f - beta1) * g;
      float v_new = beta2 * v_old + (1.0f - beta2) * (g*g);

      b_out_m = m_new;
      b_out_v = v_new;

      b_out -= alpha_t * (m_new / (std::sqrt(v_new) + epsilon));
    }
    //std::cout<< "9. End Weight / Bias update\n" << std::endl;
    
    // Print Gradients for testing
    // w_logit, b_logit
    // print_gradients(
    //   w_q_grad,
    //   w_k_grad,
    //   w_v_grad,
    //   w_o_grad,
    //   w_ff1_grad,
    //   w_ff2_grad,
    //   b_ff1_grad,
    //   b_ff2_grad,
    //   W_logit_grad,
    //   b_logit_grad
    // );

  }

  void print_gradients(
    const FixedVector<FixedVector<float>>& w_q_grad,
    const FixedVector<FixedVector<float>>& w_k_grad,
    const FixedVector<FixedVector<float>>& w_v_grad,
    const FixedVector<FixedVector<float>>& w_o_grad,
    const FixedVector<FixedVector<float>>& w_ff1_grad,
    const FixedVector<FixedVector<float>>& w_ff2_grad,
    const FixedVector<float>& b_ff1_grad,
    const FixedVector<float>& b_ff2_grad,
    const FixedVector<float>& W_logit_grad,
    float b_logit_grad
  ){
    auto print_matrix = [](const FixedVector<FixedVector<float>>& mat, const std::string& name){
      if (mat.empty()) {
        fmt::print("{}: [Empty]\n\n", name);
        return;
      }

      fmt::print("{}\n", name);
      for(const auto& row: mat){
        for(float val : row){
          fmt::print("{} ", std::isnan(val) ? "NaN" : fmt::format("{:.4f}", val));
        }
        fmt::print("\n");
      }
      fmt::print("\n");
    };

    auto print_vector = [](const FixedVector<float>& vec, const std::string& name){
      
      if(vec.empty()) {
        fmt::print("{}: [Empty]\n\n", name);
        return;
      }

      fmt::print("{}\n", name);
      for(float val : vec){
        fmt::print("{} ", std::isnan(val) ? "NaN" : fmt::format("{:.4f}", val));
      }
      fmt::print("\n\n");
    };

    fmt::print("==== Gradients ====\n");
    print_matrix(w_q_grad, "w_q_grad");
    print_matrix(w_k_grad, "w_k_grad");
    print_matrix(w_v_grad, "w_v_grad");
    print_matrix(w_o_grad, "w_o_grad");
    print_matrix(w_ff1_grad, "w_ff1_grad");
    print_matrix(w_ff2_grad, "w_ff2_grad");
    print_vector(b_ff1_grad, "b_ff1_grad");
    print_vector(b_ff2_grad, "b_ff2_grad");
    print_vector(W_logit_grad, "W_logit_grad");

    fmt::print("b_logit_grad: {:8.4f}\n", b_logit_grad);
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

  std::map<O3_CPU*, std::deque<ForwardContext>> transformer_state_buf;

} // namespace




void O3_CPU::initialize_branch_predictor() {
  ::predictors.emplace(this, "spec.json");
  ::transformer_state_buf.emplace(this, std::deque<ForwardContext>());
}

uint8_t O3_CPU::predict_branch(uint64_t ip) { 


  int d_model = ::predictors.at(this).get_d_model();
  int seq_len = ::predictors.at(this).get_seq_len();
  int num_heads = ::predictors.at(this).get_head_count();


  ForwardContext ctx = ForwardContext(seq_len, d_model, num_heads);
  // Get the transformers prediction. It will handle it's own sequence history. 
  bool prediction = ::predictors.at(this).predict(ctx, ip);

  ::transformer_state_buf.at(this).push_back(ctx);
  if (::transformer_state_buf.at(this).size() > 1000)
    ::transformer_state_buf.at(this).pop_front();
  
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

  auto state = std::find_if(
    std::begin(::transformer_state_buf.at(this)), 
    std::end(::transformer_state_buf.at(this)),
    [ip](auto x) { return x.ip == ip; }
  );
  if(state == std::end(::transformer_state_buf.at(this))){
    //std::cout<< "State lost, no backwards pass!\n" << std::endl; 
    return; // State was lost, skip training.
  }

  ForwardContext ctx = *state;
  //fmt::print("TAKEN: {}\n", taken);
  if((ctx.out < 0.5 && taken) || (ctx.out >= 0.5 && !taken)) {
    //fmt::print("Running Backwards Pass\n");
    ::predictors.at(this).backwardsPass(ctx, taken ? 1.0 : 0.0, ::predictors.at(this).lr);
  }

  ::transformer_state_buf.at(this).erase(state);

  return;
}