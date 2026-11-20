// Building blocks of the transformer: forward and backward passes.
//
// All tensors are flat row-major float arrays. Notation:
//   N = B*T rows (one row per character position of the mini-batch)
//   C = d_model, H = heads, Dh = C/H, F = d_ff, V = vocab
//
// Every function is plain serial loops. The comment "PARALLEL:" names the
// loop(s) that can be split across threads / GPU threads without write
// conflicts, and "REDUCTION:" marks sums into shared memory that need care
// (atomics, per-thread copies, or a different loop order) when parallelized.
//
// Backward conventions: parameter gradients (dW, db, dg, ...) are ACCUMULATED
// (+=); input gradients are OVERWRITTEN unless the name says "_accum".
#pragma once

#include <cstddef>

struct LossStats {
  double loss_sum = 0.0;  // sum of -log p(target) over trained positions
  int n_targets = 0;
  int n_correct = 0;      // argmax == target
};

// ---- embedding: x[b,t,:] = E[tokens[b,t], :] + P[t, :] ----
void embedding_forward(const int *tokens, const float *E, const float *P, float *x,
                       int B, int T, int C);
void embedding_backward(const int *tokens, const float *dx, float *dE, float *dP,
                        int B, int T, int C);

// ---- linear: y[n,o] = sum_i x[n,i] * W[o,i] + b[o]   (W is out x in, b may be null) ----
void linear_forward(const float *x, const float *W, const float *b, float *y,
                    int N, int in, int out);
void linear_backward(const float *x, const float *W, const float *dy, float *dx,
                     float *dW, float *db, int N, int in, int out);

// ---- layer normalization over the last dimension (eps = 1e-5) ----
void layernorm_forward(const float *x, const float *g, const float *b, float *y,
                       float *mean, float *rstd, int N, int C);
// dx is accumulated (dx += ...), because the input of every LayerNorm here is
// also the residual stream, whose gradient already holds the skip path.
void layernorm_backward_accum(const float *x, const float *g, const float *mean,
                              const float *rstd, const float *dy, float *dx, float *dg,
                              float *db, int N, int C);

// ---- ReLU ----
void relu_forward(const float *x, float *y, size_t n);
void relu_backward(const float *x, const float *dy, float *dx, size_t n);  // x = pre-activation

// ---- y = a + b (residual connection) ----
void add(const float *a, const float *b, float *y, size_t n);

// ---- causal multi-head self-attention ----
// qkv: (B, T, 3C), each row = [q (C) | k (C) | v (C)], head h uses columns h*Dh..(h+1)*Dh.
// p:   (B, H, T, T) attention weights (only j <= i is used), saved for backward.
// out: (B, T, C) heads concatenated.
void attention_forward(const float *qkv, float *p, float *out, int B, int T, int C, int H);
// ds: (B, H, T, T) scratch. dqkv is overwritten.
void attention_backward(const float *qkv, const float *p, const float *dout, float *dqkv,
                        float *ds, int B, int T, int C, int H);
// One query attending to n_keys keys/values whose rows are `stride` floats apart.
// Used by attention_forward and by generation with a KV cache.
void attend_one_query(const float *q, const float *k, const float *v, size_t stride,
                      int n_keys, int Dh, float *p, float *out);

// ---- softmax + cross-entropy over rows whose target >= 0 ----
LossStats softmax_cross_entropy(const float *logits, const int *targets, float *probs,
                                int N, int V);
// dlogits = (probs - onehot(target)) * scale for trained rows, 0 for the rest.
void softmax_cross_entropy_backward(const float *probs, const int *targets, float scale,
                                    float *dlogits, int N, int V);
