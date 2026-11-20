#include "optimizer.h"

#include <algorithm>
#include <cmath>

double grad_norm(const float *g, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += static_cast<double>(g[i]) * g[i];
  return std::sqrt(s);
}

void scale_grad(float *g, size_t n, float s) {
  for (size_t i = 0; i < n; ++i) g[i] *= s;
}

void MomentumSGD::step(float *w, const float *g, float lr) {
  for (size_t i = 0; i < v_.size(); ++i) {
    v_[i] = momentum_ * v_[i] + g[i];
    w[i] -= lr * v_[i];
  }
}

float learning_rate(const Config &cfg, int update, int n_updates) {
  double f = 1.0;
  if (cfg.warmup > 0) f = std::min(1.0, (update + 1.0) / cfg.warmup);
  if (cfg.cosine) f *= 0.5 * (1.0 + std::cos(M_PI * update / n_updates));
  return static_cast<float>(cfg.lr * f);
}
