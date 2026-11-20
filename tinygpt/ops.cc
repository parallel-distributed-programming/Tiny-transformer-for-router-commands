#include "ops.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr float kLnEps = 1e-5f;
inline size_t idx(size_t row, size_t cols, size_t col) { return row * cols + col; }
}  // namespace

// ============================================================ embedding

void embedding_forward(const int *tokens, const float *E, const float *P, float *x,
                       int B, int T, int C) {
  // PARALLEL: b, t, c (all independent)
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t) {
      const size_t n = static_cast<size_t>(b) * T + t;
      const float *e = E + static_cast<size_t>(tokens[n]) * C;
      const float *p = P + static_cast<size_t>(t) * C;
      for (int c = 0; c < C; ++c) x[idx(n, C, c)] = e[c] + p[c];
    }
}

void embedding_backward(const int *tokens, const float *dx, float *dE, float *dP,
                        int B, int T, int C) {
  // PARALLEL: c. REDUCTION: the same token row of dE (and the same position
  // row of dP, across b) receives many updates when parallelizing over b or t.
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t) {
      const size_t n = static_cast<size_t>(b) * T + t;
      float *de = dE + static_cast<size_t>(tokens[n]) * C;
      float *dp = dP + static_cast<size_t>(t) * C;
      for (int c = 0; c < C; ++c) {
        de[c] += dx[idx(n, C, c)];
        dp[c] += dx[idx(n, C, c)];
      }
    }
}

// ============================================================ linear

void linear_forward(const float *x, const float *W, const float *b, float *y,
                    int N, int in, int out) {
  // PARALLEL: n and o (each y[n,o] is an independent dot product)
  for (int n = 0; n < N; ++n) {
    const float *xr = x + static_cast<size_t>(n) * in;
    for (int o = 0; o < out; ++o) {
      const float *wr = W + static_cast<size_t>(o) * in;
      float s = b ? b[o] : 0.0f;
      for (int i = 0; i < in; ++i) s += wr[i] * xr[i];
      y[idx(n, out, o)] = s;
    }
  }
}

void linear_backward(const float *x, const float *W, const float *dy, float *dx,
                     float *dW, float *db, int N, int in, int out) {
  // dx[n,:] = sum_o dy[n,o] * W[o,:]          PARALLEL: n
  for (int n = 0; n < N; ++n) {
    float *dxr = dx + static_cast<size_t>(n) * in;
    std::fill(dxr, dxr + in, 0.0f);
    for (int o = 0; o < out; ++o) {
      const float g = dy[idx(n, out, o)];
      const float *wr = W + static_cast<size_t>(o) * in;
      for (int i = 0; i < in; ++i) dxr[i] += g * wr[i];
    }
  }
  // dW[o,:] += sum_n dy[n,o] * x[n,:]          PARALLEL: o (REDUCTION over n)
  for (int o = 0; o < out; ++o) {
    float *dwr = dW + static_cast<size_t>(o) * in;
    float gb = 0.0f;
    for (int n = 0; n < N; ++n) {
      const float g = dy[idx(n, out, o)];
      const float *xr = x + static_cast<size_t>(n) * in;
      for (int i = 0; i < in; ++i) dwr[i] += g * xr[i];
      gb += g;
    }
    if (db) db[o] += gb;
  }
}

// ============================================================ layernorm

void layernorm_forward(const float *x, const float *g, const float *b, float *y,
                       float *mean, float *rstd, int N, int C) {
  // PARALLEL: n (REDUCTION over c inside a row)
  for (int n = 0; n < N; ++n) {
    const float *xr = x + static_cast<size_t>(n) * C;
    float m = 0.0f;
    for (int c = 0; c < C; ++c) m += xr[c];
    m /= C;
    float var = 0.0f;
    for (int c = 0; c < C; ++c) var += (xr[c] - m) * (xr[c] - m);
    var /= C;
    const float r = 1.0f / std::sqrt(var + kLnEps);
    mean[n] = m;
    rstd[n] = r;
    for (int c = 0; c < C; ++c) y[idx(n, C, c)] = (xr[c] - m) * r * g[c] + b[c];
  }
}

void layernorm_backward_accum(const float *x, const float *g, const float *mean,
                              const float *rstd, const float *dy, float *dx, float *dg,
                              float *db, int N, int C) {
  // With xhat = (x - mean) * rstd and dxhat = dy * g:
  //   dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
  // PARALLEL: n (REDUCTION over c inside a row)
  for (int n = 0; n < N; ++n) {
    const float *xr = x + static_cast<size_t>(n) * C;
    const float *dyr = dy + static_cast<size_t>(n) * C;
    float sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float xhat = (xr[c] - mean[n]) * rstd[n];
      const float dxhat = dyr[c] * g[c];
      sum_dxhat += dxhat;
      sum_dxhat_xhat += dxhat * xhat;
    }
    for (int c = 0; c < C; ++c) {
      const float xhat = (xr[c] - mean[n]) * rstd[n];
      const float dxhat = dyr[c] * g[c];
      dx[idx(n, C, c)] += rstd[n] * (dxhat - sum_dxhat / C - xhat * sum_dxhat_xhat / C);
    }
  }
  // dg[c] += sum_n dy * xhat, db[c] += sum_n dy     PARALLEL: c (REDUCTION over n)
  for (int c = 0; c < C; ++c) {
    float sg = 0.0f, sb = 0.0f;
    for (int n = 0; n < N; ++n) {
      const float xhat = (x[idx(n, C, c)] - mean[n]) * rstd[n];
      sg += dy[idx(n, C, c)] * xhat;
      sb += dy[idx(n, C, c)];
    }
    dg[c] += sg;
    db[c] += sb;
  }
}

// ============================================================ elementwise

void relu_forward(const float *x, float *y, size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;  // PARALLEL: i
}

void relu_backward(const float *x, const float *dy, float *dx, size_t n) {
  for (size_t i = 0; i < n; ++i) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;  // PARALLEL: i
}

void add(const float *a, const float *b, float *y, size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] = a[i] + b[i];  // PARALLEL: i
}

// ============================================================ attention

void attend_one_query(const float *q, const float *k, const float *v, size_t stride,
                      int n_keys, int Dh, float *p, float *out) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  float max_s = -INFINITY;
  for (int j = 0; j < n_keys; ++j) {
    const float *kj = k + j * stride;
    float s = 0.0f;
    for (int d = 0; d < Dh; ++d) s += q[d] * kj[d];
    p[j] = s * scale;
    max_s = std::max(max_s, p[j]);
  }
  float z = 0.0f;
  for (int j = 0; j < n_keys; ++j) {
    p[j] = std::exp(p[j] - max_s);
    z += p[j];
  }
  for (int j = 0; j < n_keys; ++j) p[j] /= z;
  std::fill(out, out + Dh, 0.0f);
  for (int j = 0; j < n_keys; ++j) {
    const float *vj = v + j * stride;
    for (int d = 0; d < Dh; ++d) out[d] += p[j] * vj[d];
  }
}

void attention_forward(const float *qkv, float *p, float *out, int B, int T, int C, int H) {
  const int Dh = C / H;
  const size_t row = 3 * static_cast<size_t>(C);
  // PARALLEL: b, h, i (every query is independent)
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < T; ++i) {
        const float *base = qkv + static_cast<size_t>(b) * T * row;
        const float *q = base + i * row + h * Dh;
        const float *k = base + C + h * Dh;
        const float *v = base + 2 * C + h * Dh;
        float *p_row = p + ((static_cast<size_t>(b) * H + h) * T + i) * T;
        float *o = out + (static_cast<size_t>(b) * T + i) * C + h * Dh;
        attend_one_query(q, k, v, row, i + 1, Dh, p_row, o);  // causal: keys 0..i
      }
}

void attention_backward(const float *qkv, const float *p, const float *dout, float *dqkv,
                        float *ds, int B, int T, int C, int H) {
  const int Dh = C / H;
  const size_t row = 3 * static_cast<size_t>(C);
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  // PARALLEL: b, h. Inside, the three passes are ordered so that each pass
  // writes disjoint rows (no atomics needed): pass 1 and 2 by query i,
  // pass 3 by key j.
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const float *base = qkv + static_cast<size_t>(b) * T * row;
      float *dbase = dqkv + static_cast<size_t>(b) * T * row;
      const float *P = p + (static_cast<size_t>(b) * H + h) * T * T;
      float *S = ds + (static_cast<size_t>(b) * H + h) * T * T;
      auto q = [&](int t) { return base + t * row + h * Dh; };
      auto k = [&](int t) { return base + t * row + C + h * Dh; };
      auto v = [&](int t) { return base + t * row + 2 * C + h * Dh; };
      auto dO = [&](int t) { return dout + (static_cast<size_t>(b) * T + t) * C + h * Dh; };

      // Pass 1: gradient of the scores through softmax.   PARALLEL: i
      //   dp_ij = dout_i . v_j ;  ds_ij = p_ij * (dp_ij - sum_k p_ik dp_ik)
      for (int i = 0; i < T; ++i) {
        float *Si = S + static_cast<size_t>(i) * T;
        const float *Pi = P + static_cast<size_t>(i) * T;
        float weighted = 0.0f;
        for (int j = 0; j <= i; ++j) {
          float dp = 0.0f;
          for (int d = 0; d < Dh; ++d) dp += dO(i)[d] * v(j)[d];
          Si[j] = dp;
          weighted += Pi[j] * dp;
        }
        for (int j = 0; j <= i; ++j) Si[j] = Pi[j] * (Si[j] - weighted);
      }
      // Pass 2: dq_i = scale * sum_{j<=i} ds_ij k_j.      PARALLEL: i
      for (int i = 0; i < T; ++i) {
        float *dq = dbase + i * row + h * Dh;
        std::fill(dq, dq + Dh, 0.0f);
        for (int j = 0; j <= i; ++j) {
          const float s = S[static_cast<size_t>(i) * T + j] * scale;
          for (int d = 0; d < Dh; ++d) dq[d] += s * k(j)[d];
        }
      }
      // Pass 3: dk_j = scale * sum_{i>=j} ds_ij q_i,
      //         dv_j =         sum_{i>=j} p_ij dout_i.     PARALLEL: j
      for (int j = 0; j < T; ++j) {
        float *dk = dbase + j * row + C + h * Dh;
        float *dv = dbase + j * row + 2 * C + h * Dh;
        std::fill(dk, dk + Dh, 0.0f);
        std::fill(dv, dv + Dh, 0.0f);
        for (int i = j; i < T; ++i) {
          const float s = S[static_cast<size_t>(i) * T + j] * scale;
          const float pij = P[static_cast<size_t>(i) * T + j];
          for (int d = 0; d < Dh; ++d) {
            dk[d] += s * q(i)[d];
            dv[d] += pij * dO(i)[d];
          }
        }
      }
    }
}

// ============================================================ loss

LossStats softmax_cross_entropy(const float *logits, const int *targets, float *probs,
                                int N, int V) {
  // PARALLEL: n (REDUCTION: the three totals)
  LossStats st;
  for (int n = 0; n < N; ++n) {
    if (targets[n] < 0) continue;
    const float *l = logits + static_cast<size_t>(n) * V;
    float *pr = probs + static_cast<size_t>(n) * V;
    int best = 0;
    for (int o = 1; o < V; ++o)
      if (l[o] > l[best]) best = o;
    float z = 0.0f;
    for (int o = 0; o < V; ++o) {
      pr[o] = std::exp(l[o] - l[best]);
      z += pr[o];
    }
    for (int o = 0; o < V; ++o) pr[o] /= z;
    st.loss_sum += -std::log(std::max(pr[targets[n]], 1e-12f));
    st.n_correct += (best == targets[n]);
    st.n_targets += 1;
  }
  return st;
}

void softmax_cross_entropy_backward(const float *probs, const int *targets, float scale,
                                    float *dlogits, int N, int V) {
  // PARALLEL: n
  for (int n = 0; n < N; ++n) {
    float *d = dlogits + static_cast<size_t>(n) * V;
    if (targets[n] < 0) {
      std::fill(d, d + V, 0.0f);
      continue;
    }
    const float *pr = probs + static_cast<size_t>(n) * V;
    for (int o = 0; o < V; ++o) d[o] = pr[o] * scale;
    d[targets[n]] -= scale;
  }
}
