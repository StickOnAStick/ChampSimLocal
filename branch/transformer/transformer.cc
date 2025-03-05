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
      - w_q, w_v, w_k: [d_model, d_q] [d_model, d_k] [d_model, d_v]
      - Q, K, V:  [sequence_len, d_q] [sequence_len, d_v]
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
    ctx.attn_out = USE_CUDA ? FixedVectorMath::dotProductCuda(attention_out, w_o) : FixedVectorMath::dotProduct(attention_out, w_o);
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
    ctx.ffnIntermediate = hidden;
    //---------------------------------------------------
    // 2.) Relu in place
    //---------------------------------------------------
    FixedVectorMath::relu(hidden);
    ctx.ffnActivated = hidden;
    //---------------------------------------------------
    // 3.) output = hidden * w_ff2 + b_ff2
    //     => output: shapre [sequence_len, d_model]
    //---------------------------------------------------
    FixedVector<FixedVector<float>> output = USE_CUDA ? FixedVectorMath::linearCuda(hidden, w_ff2, b_ff2) : FixedVectorMath::linear(hidden, w_ff2, b_ff2);
    ctx.ffnOut = output;
    return ctx.ffnOut;
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

  void backwardsPass(ForwardContext& ctx, float y_true, float learning_rate = 0.001f){
    
    // Layer Norm. Backwards pass helper lambda function
    auto layerNormBackward = [&](  // *[&] is a lambda function which captures all variables out-of-scope by reference
      const FixedVector<FixedVector<float>>& x,  // LN Input
      const FixedVector<FixedVector<float>>& y,  // LN Output
      const FixedVector<float>& mean_vec,        // The mean of the i_th row of the sequence [d_model]
      const FixedVector<float>& var_vec,         // Variance of the i_th row
      const FixedVector<FixedVector<float>>& dL_dy, // Grad wrt LN Out
      FixedVector<FixedVector<float>>& dL_dx,       // to fill
      float epsilon
    ){
      for(int i = 0; i < this->sequence_len; i++){
        float mean_i = mean_vec[i];
        float var_i  = var_vec[i];
        float inv_std = 1.0f / std::sqrt(var_i + epsilon);

        // We'll need sums across the feature dimension.
        float sum_dL_dy        = 0.0f;
        float sum_dL_dy_times_y= 0.0f;

        for(int k = 0; k < d_model; k++){
            sum_dL_dy         += dL_dy[i][k];
            sum_dL_dy_times_y += dL_dy[i][k] * y[i][k]; 
        }

        // Now compute dL/dx for each feature in row i
        for(int k = 0; k < d_model; k++){
            float grad_yk = dL_dy[i][k];   // dL/dy[i][k]
            float yk      = y[i][k];       // LN output
            // LN backprop formula:
            // dL/dx = (1 / inv_std) * [ grad_yk
            //    - (1/d_model)*sum_dL_dy
            //    - yk*(1/d_model)*sum_dL_dy_times_y ]
            float term = grad_yk
                         - (sum_dL_dy / (float)d_model)
                         - (yk * sum_dL_dy_times_y / (float)d_model);
            dL_dx[i][k] = inv_std * term;
          }
        }
    };

    /****************************************************
     * 0) Derivative of BCE Loss wrt logit
     ****************************************************/
    float dL_dlogit = (ctx.out - y_true);  // out = sigmoid(logit)

    /****************************************************
     * 1) Backprop from logit -> W_logit, b_logit, pooled
     ****************************************************/
    // Suppose we have class-member: W_logit (size d_model), b_logit (scalar)
    // We'll accumulate grads in local arrays:
    static FixedVector<float> W_logit_grad = FixedVector<float>(d_model, 0.0f); 
    static float b_logit_grad = 0.0f;

    // dL/dW_logit[k] = dL/dlogit * pooled[k]
    // dL/db_logit    = dL/dlogit
    // dL/dpooled[k]  = dL/dlogit * W_logit[k]
    FixedVector<float> dL_dpooled(d_model, 0.0f);
    for(int k = 0; k < d_model; k++){
        W_logit_grad[k] = dL_dlogit * this->w_out[k];
        // Actually we must correct the above line:
        //   The gradient wrt W_logit is (dL/dlogit * pooled[k])
        //   So:
        W_logit_grad[k] = dL_dlogit * ctx.pooled[k];
    }
    b_logit_grad = dL_dlogit;

    for(int k = 0; k < d_model; k++){
        dL_dpooled[k] = dL_dlogit * this->w_out[k];
    }

    /****************************************************
     * 2) Backprop from pooled -> ff_normed
     *    pooled[k] = average of ff_normed[i][k] over sequence_len
     ****************************************************/
    // So dL/dff_normed[i][k] += dL/dpooled[k] / sequence_len
    FixedVector<FixedVector<float>> dL_dFF_normed(sequence_len, FixedVector<float>(d_model, 0.0f));
    for(int i = 0; i < sequence_len; i++){
        for(int k = 0; k < d_model; k++){
            dL_dFF_normed[i][k] = dL_dpooled[k] / (float)sequence_len;
        }
    }

    /****************************************************
     * 3) LayerNorm backward on the FF block
     *    FF block output is ctx.ff_normed => LN output
     *    LN input was ctx.ff_post_residual => (ff_out + attn_normed)
     ****************************************************/
    FixedVector<FixedVector<float>> dL_dFF_post_resid(sequence_len, FixedVector<float>(d_model, 0.0f));
    // LN backward:
    layerNormBackward(
        ctx.ff_post_residual,   // LN input
        ctx.ff_normed,          // LN output
        ctx.ff_mean_i,          // means
        ctx.ff_var_i,           // vars
        dL_dFF_normed,          // gradient from above
        dL_dFF_post_resid,      // result => dL/d LN input
        1e-5f
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

    /****************************************************
     * 4) Backprop the feed-forward sublayer
     *    From dL_dFF_out_raw => (linear -> ReLU -> linear).
     *    We have w_ff2, b_ff2, then w_ff1, b_ff1, etc.
     *    We also have ff_hidden in ctx (pre-RELU).
     ****************************************************/
    // We'll keep local gradients for w_ff2, b_ff2, w_ff1, b_ff1
    static FixedVector<FixedVector<float>> w_ff2_grad = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f));
    static FixedVector<float> b_ff2_grad = FixedVector<float>(d_model, 0.0f);
    static FixedVector<FixedVector<float>> w_ff1_grad = FixedVector<FixedVector<float>>(d_model, FixedVector<float>(d_model, 0.0f));
    static FixedVector<float> b_ff1_grad = FixedVector<float>(d_model, 0.0f);

    // We'll also need gradient wrt ff_hidden (post-relu):
    FixedVector<FixedVector<float>> dL_dff_hidden(sequence_len, FixedVector<float>(d_model, 0.0f));

    // The final FF layer was: ff_out = ff_hidden * w_ff2 + b_ff2
    // dL/dff_out_raw = dL_dFF_out_raw
    for(int i = 0; i < sequence_len; i++){
        for(int j = 0; j < d_model; j++){
            float grad_out_ij = dL_dFF_out_raw[i][j];
            // b_ff2
            b_ff2_grad[j] += grad_out_ij;

            // w_ff2 => shape [d_model, d_model]
            // ff_hidden[i][k] * w_ff2[k][j]
            for(int k = 0; k < d_model; k++){
                w_ff2_grad[k][j] += grad_out_ij * ctx.ff_hidden[i][k];
            }
        }
    }

    // dL/dff_hidden = sum_j( dL/dff_out[i][j] * w_ff2[k][j] ) for each k
    // Then ReLU( ff_hidden ), so we must gate with ReLU derivative
    for(int i = 0; i < sequence_len; i++){
        for(int k = 0; k < d_model; k++){
            float sum_grad = 0.0f;
            for(int j = 0; j < d_model; j++){
                sum_grad += dL_dFF_out_raw[i][j] * this->w_ff2[k][j];
            }
            // Now apply ReLU gate. In forward pass, we did in-place ReLU on ff_hidden
            // If ff_hidden[i][k] <= 0 => gradient is 0
            // We stored the post-activation in ctx.ff_hidden too, so let's check that:
            if(ctx.ff_hidden[i][k] <= 0.0f) {
                sum_grad = 0.0f;
            }
            dL_dff_hidden[i][k] = sum_grad;
        }
    }

    // Now backprop the first FF layer: ff_hidden = ffnInput * w_ff1 + b_ff1
    // with ffnInput = ctx.ffnInput ( = attn_normed in forward code).
    FixedVector<FixedVector<float>> dL_dFFN_input(sequence_len, FixedVector<float>(d_model, 0.0f));

    for(int i = 0; i < sequence_len; i++){
        for(int j = 0; j < d_model; j++){
            float grad_hidden_ij = dL_dff_hidden[i][j];
            b_ff1_grad[j] += grad_hidden_ij;
            for(int k = 0; k < d_model; k++){
                w_ff1_grad[k][j] += grad_hidden_ij * ctx.ffnInput[i][k];
                dL_dFFN_input[i][k] += grad_hidden_ij * this->w_ff1[k][j];
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

    /****************************************************
     * 5) Now handle LN backward for the MHA output
     *    attn_normed = LN( attn_post_residual )
     *    attn_post_residual = attn_out + original_input_seq
     *    The "original_input_seq" for a single-layer decoder might be
     *    the embedding or the input to the block. We'll call it dL_dInput.
     ****************************************************/
    FixedVector<FixedVector<float>> dL_dAttn_post_resid(sequence_len, FixedVector<float>(d_model, 0.0f));

    layerNormBackward(
        ctx.attn_post_residual,  // LN input
        ctx.attn_normed,         // LN output
        ctx.attn_mean_i,
        ctx.attn_var_i,
        dL_dAttn_normed,         // grad from feed-forward residual
        dL_dAttn_post_resid,     // result => gradient w.r.t. LN input
        1e-5f
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

    /****************************************************
     * 6) Backprop multi-head attention
     *    final MHA out: attention_out * w_o => attn_out
     *    So we do the matrix multiply backward first:
     ****************************************************/
    static FixedVector<FixedVector<float>> w_o_grad(d_model, FixedVector<float>(d_model, 0.0f));

    // dL_dAttentionOut is shape [sequence_len, d_model]
    FixedVector<FixedVector<float>> dL_dAttentionOut(sequence_len, FixedVector<float>(d_model, 0.0f));

    // attn_out[i][j] = sum_{k} attention_out[i][k] * w_o[k][j]
    for(int i = 0; i < sequence_len; i++){
        for(int j = 0; j < d_model; j++){
            float grad_attn_out_ij = dL_dAttn_out[i][j];
            // accumulate w_o grad
            for(int k = 0; k < d_model; k++){
                w_o_grad[k][j] += grad_attn_out_ij * ctx.attention_out[i][k];
            }
            // pass back to attention_out
            for(int k = 0; k < d_model; k++){
                dL_dAttentionOut[i][k] += grad_attn_out_ij * this->w_o[k][j];
            }
        }
    }

    /****************************************************
     * 7) Now handle each MHA head:
     *    We have Q,K,V in ctx.Q, ctx.K, ctx.V
     *    The final was attention_out = concat(head_0..head_n)
     ****************************************************/
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
                    float attn_ij = ctx.softmax_attn[i][j];
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
                rowDot += dL_dSoftmax[i][p] * ctx.softmax_attn[i][p];
            }
            for(int j = 0; j < sequence_len; j++){
                float grad_softmax_ij = dL_dSoftmax[i][j];
                float sm_ij           = ctx.softmax_attn[i][j];
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

    /****************************************************
     * 8) Backprop Q,K,V => seq_history => w_q, w_k, w_v
     ****************************************************/
    static FixedVector<FixedVector<float>> w_q_grad(d_model, FixedVector<float>(d_model, 0.0f));
    static FixedVector<FixedVector<float>> w_k_grad(d_model, FixedVector<float>(d_model, 0.0f));
    static FixedVector<FixedVector<float>> w_v_grad(d_model, FixedVector<float>(d_model, 0.0f));

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

    /****************************************************
     * 9) Now we apply vanilla SGD updates to:
     *    w_q, w_k, w_v, w_o,
     *    w_ff1, b_ff1, w_ff2, b_ff2,
     *    W_logit, b_logit,
     *    (and if LN had gamma,beta, we’d update them, but we don't here).
     ****************************************************/
    // w_q
    for(int p = 0; p < d_model; p++){
        for(int j = 0; j < d_model; j++){
            this->w_q[p][j] -= learning_rate * w_q_grad[p][j];
            this->w_k[p][j] -= learning_rate * w_k_grad[p][j];
            this->w_v[p][j] -= learning_rate * w_v_grad[p][j];
            this->w_ff1[p][j] -= learning_rate * w_ff1_grad[p][j];
            this->w_ff2[p][j] -= learning_rate * w_ff2_grad[p][j];
        }
    }
    // w_o
    for(int p = 0; p < d_model; p++){
        for(int j = 0; j < d_model; j++){
            this->w_o[p][j] -= learning_rate * w_o_grad[p][j];
        }
    }
    // b_ff1, b_ff2
    for(int j = 0; j < d_model; j++){
        this->b_ff1[j] -= learning_rate * b_ff1_grad[j];
        this->b_ff2[j] -= learning_rate * b_ff2_grad[j];
    }
    // W_logit, b_logit
    for(int k = 0; k < d_model; k++){
        this->w_out[k] -= learning_rate * W_logit_grad[k];
    }
    this->b_out -= learning_rate * b_logit_grad;
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