// Gradient clipping, SGD with momentum, and the learning-rate schedule.
// All of them are single loops over the flat parameter / gradient arrays.
#pragma once

#include <cstddef>
#include <vector>

#include "config.h"

// Global L2 norm of the gradient.   PARALLEL: i (REDUCTION: the sum)
double grad_norm(const float *g, size_t n);
// g *= s.                            PARALLEL: i
void scale_grad(float *g, size_t n, float s);

// v = momentum * v + g ;  w -= lr * v   (same as torch.optim.SGD with momentum)
class MomentumSGD {
 public:
  MomentumSGD(size_t n, float momentum) : v_(n, 0.0f), momentum_(momentum) {}
  void step(float *w, const float *g, float lr);  // PARALLEL: i
 private:
  std::vector<float> v_;
  float momentum_;
};

// Linear warm-up for cfg.warmup updates, then cosine decay to 0 (if enabled).
float learning_rate(const Config &cfg, int update, int n_updates);
