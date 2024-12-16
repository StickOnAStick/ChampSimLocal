#include "transformer.hh"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <deque>
#include <map>
#include <random>

#include "msl/fwcounter.h"
#include "ooo_cpu.h"
#include "FixedVector.hh"
#include "FixedVectorMath.hh"

constexpr std::size_t WEIGHT_BITS = 8;     // We can quantize down to 4 later
constexpr std::size_t HISTORY_LENGTH = 24; // We can adjust. Defaulting to current perceptron's length for closer 1-to-1 comparison


namespace
{
template <std::size_t HISTLEN, std::size_t BITS> // We set the history length and size of weights
class Transformer : public TransformerBase
{
public:
  Transformer(const std::string& config_file) : TransformerBase(config_file) {}

  void hashed_posEncoding(uint64_t& input, std::bitset<HISTLEN> global_history) override {
    uint64_t hashed_input = (*input & 0xFFF) ^ global_history; // Use 12 LSBs of IP, smaller locality, reduced HW cost
    // Positionally encode based off hashed input XOR'd with recent global history
    uint8_t pos_enc = (hashed_input % static_cast<int>(pow(2, this->d_pos))); // Reduce to 5 bits.

    // Add IP bits to the constructed d_model vector
    FixedVector<float> encoded_input(this->d_model);
    for(int i = 0; i < this->d_in; i++){
      int bit = (*input >> i) & 1;
      encoded_input[i] = bit;
    }

    // Add the positional encoding bits to the input d_model vector
    for(int i = 0; i < this->d_pos; i++){
      int bit = (pos_enc >> i) & 1;
      encoded_input[this->d_in + i] = bit;
    }

    // Add the new input to the beginning of sequence history.
    this->sequence_history.push(encoded_input); 
  }

  void fixed_posEncoding(uint64_t& ip) override {
    FixedVector<float> encoded_input(this->d_model);

    for(int i = 0; i < this->d_model; i++){
      encoded_input[i] = (*ip >> i) & 1;
    }

    // Push the new IP into history
    this->sequence_history.push(encoded_input);

    // Incriment all previous IP's positional encodings by 1
    for(uint8_t pos = static_cast<uint8_t>(this->sequence_history.size()); pos > 0; --pos){
      for(int j = 0; j < this->d_pos; j++){
        this->sequence_history[pos][this->d_in + j] = (pos >> j) & 1;
      }
    }
  }

  FixedVector<FixedVector<float>> MALayer() override {
    if (this->d_model % this->num_ma_heads != 0){
      throw std::runtime_error("Model dimension (d_model) must be divisible by the number of heads");
    }

    int d_head = this->d_model / this->num_ma_heads;

    // Output matrix
    FixedVector<FixedVector<float>> attention_out(sequence_len, FixedVector<float>(d_model, 0.0f));

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

    FixedVector<FixedVector<float>> Q(sequence_len, FixedVector<float>(d_q, 0.0f));
    FixedVector<FixedVector<float>> K(sequence_len, FixedVector<float>(d_k, 0.0f));
    FixedVector<FixedVector<float>> V(sequence_len, FixedVector<float>(d_v, 0.0f));

    // Compute Q, K, V
    Q = FixedVectorMath::dotProduct(&sequence_history, &w_q);
    K = FixedVectorMath::dotProduct(&sequence_history, &w_k);
    V = FixedVectorMath::dotProduct(&sequence_history, &w_v);

    /*
      Step 2. Process Each Head
      - Slice Q, K, V for each head
      - Compute Attention scores:   Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k_head)) * V
      - Concat results into final output
    */
    for(int head = 0; head < num_ma_heads; ++head){
      

    }


  }
  
  bool predict(uint64_t ip, std::bitset<HISTLEN> global_history){

    /*
      Positional Encoding

      Dealers choice, test with correct weights
    */    
    this->hashed_posEncoding(&ip, global_history);
    // fixed_posEncoding(&ip);

    /*
      Multi-Headed attention
    */
    this->MALayer()
  }

};


} // namespace


void O3_CPU::initialize_branch_predictor(){
}

uint8_t O3_CPU::predict_branch(uint64_t ip){
  return 1;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type){
}