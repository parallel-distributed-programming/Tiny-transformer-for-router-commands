// TinyGPT: causal character-level transformer with pre-LayerNorm blocks.
//
//   x = E[token] + P[position]                                   (N, C)
//   repeat n_layer times:
//     x = x + Wo · MultiHeadCausalAttention(LayerNorm1(x) · Wqkv)
//     x = x + W2 · ReLU(W1 · LayerNorm2(x) + b1) + b2
//   logits = Wcls · LayerNormF(x) + bcls                         (N, V)
//
// All parameters live in ONE contiguous float array (ParamSet::data), and the
// gradients in another array with the same layout. Each tensor is a pointer
// into it (Weights). This keeps "zero all gradients", "gradient norm" and the
// optimizer step as single loops, and the whole model as one memcpy to a GPU.
#pragma once

#include <vector>

#include "data.h"
#include "ops.h"

struct ModelDims {
  int V;        // vocabulary
  int C;        // d_model
  int H;        // heads
  int F;        // d_ff
  int L;        // layers
  int max_B;    // largest mini-batch (sizes the activation buffers)
};

// Pointers to every tensor inside one flat array (weights OR gradients).
struct LayerWeights {
  float *ln1_g, *ln1_b;  // (C), (C)
  float *Wqkv;           // (3C, C)  rows: q(0..C), k(C..2C), v(2C..3C)
  float *Wo;             // (C, C)
  float *ln2_g, *ln2_b;  // (C), (C)
  float *W1, *b1;        // (F, C), (F)
  float *W2, *b2;        // (C, F), (C)
};
struct Weights {
  float *E;              // (V, C)   character embedding
  float *P;              // (kMaxLen, C) position embedding
  std::vector<LayerWeights> layers;
  float *lnf_g, *lnf_b;  // (C), (C)
  float *Wcls, *bcls;    // (V, C), (V)
};

struct ParamSet {
  std::vector<float> data;
  Weights w;
  explicit ParamSet(const ModelDims &d);
  size_t size() const { return data.size(); }
};

// Activations saved by forward() for backward().
struct LayerActs {
  std::vector<float> x;              // (N, C)   input of the layer (residual stream)
  std::vector<float> ln1, ln1_mean, ln1_rstd;
  std::vector<float> qkv;            // (N, 3C)
  std::vector<float> att_p;          // (B, H, T, T)
  std::vector<float> att_out;        // (N, C)   heads concatenated, before Wo
  std::vector<float> x_mid;          // (N, C)   after the attention residual
  std::vector<float> ln2, ln2_mean, ln2_rstd;
  std::vector<float> ff_pre, ff_act; // (N, F)   before / after ReLU
};

// Per-sequence key/value cache for generation: (layer, position, C).
struct KVCache {
  std::vector<float> k, v;
};

class Model {
 public:
  Model(const ModelDims &dims, int seed);

  // Forward pass over a mini-batch; returns loss statistics over the trained
  // positions and keeps the activations for backward().
  LossStats forward(const Batch &batch);
  // Accumulates d(mean loss)/d(param) into grads. Call after forward().
  void backward(const Batch &batch);

  // Generation: feed one token at position pos (0, 1, 2, ...) and get the
  // logits for the next one. K/V of earlier positions come from the cache.
  KVCache new_cache() const;
  void decode_step(int token, int pos, KVCache *cache, float *logits);

  const ModelDims &dims() const { return d_; }
  const float *logits() const { return logits_.data(); }  // (B*T, V) of the last forward()
  ParamSet params, grads;

 private:
  ModelDims d_;
  // forward activations (sized for max_B * kMaxLen rows)
  std::vector<LayerActs> acts_;
  std::vector<float> x_final_, lnf_, lnf_mean_, lnf_rstd_, logits_, probs_;
  // backward scratch
  std::vector<float> dx_, dln_, dqkv_, datt_, dff_, dlogits_, ds_, tmp_;
  // generation scratch (one row)
  std::vector<float> g_x_, g_h_, g_qkv_, g_att_, g_tmp_, g_ff_, g_p_;
};
