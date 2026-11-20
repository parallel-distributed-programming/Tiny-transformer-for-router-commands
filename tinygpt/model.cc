#include "model.h"

#include <algorithm>
#include <random>

namespace {

// Lays out all tensors one after another in `base` (or only counts them when
// base == nullptr). Returns the total number of floats.
size_t layout(const ModelDims &d, float *base, Weights *w) {
  size_t off = 0;
  auto take = [&](size_t n) {
    float *p = base ? base + off : nullptr;
    off += n;
    return p;
  };
  const size_t C = d.C, V = d.V, F = d.F;
  w->E = take(V * C);
  w->P = take(kMaxLen * C);
  w->layers.resize(d.L);
  for (LayerWeights &l : w->layers) {
    l.ln1_g = take(C);
    l.ln1_b = take(C);
    l.Wqkv = take(3 * C * C);
    l.Wo = take(C * C);
    l.ln2_g = take(C);
    l.ln2_b = take(C);
    l.W1 = take(F * C);
    l.b1 = take(F);
    l.W2 = take(C * F);
    l.b2 = take(C);
  }
  w->lnf_g = take(C);
  w->lnf_b = take(C);
  w->Wcls = take(V * C);
  w->bcls = take(V);
  return off;
}

}  // namespace

ParamSet::ParamSet(const ModelDims &d) {
  Weights tmp;
  data.assign(layout(d, nullptr, &tmp), 0.0f);
  layout(d, data.data(), &w);
}

Model::Model(const ModelDims &dims, int seed) : params(dims), grads(dims), d_(dims) {
  // Initialization as in app.cc: weights ~ U(-0.06, 0.06), biases 0,
  // LayerNorm gains 1 and shifts 0.
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uf(-0.06f, 0.06f);
  auto uniform = [&](float *p, size_t n) { for (size_t i = 0; i < n; ++i) p[i] = uf(rng); };
  auto ones = [](float *p, size_t n) { std::fill(p, p + n, 1.0f); };
  const size_t C = d_.C, V = d_.V, F = d_.F;
  Weights &w = params.w;
  uniform(w.E, V * C);
  uniform(w.P, kMaxLen * C);
  for (LayerWeights &l : w.layers) {
    uniform(l.Wqkv, 3 * C * C);
    uniform(l.Wo, C * C);
    uniform(l.W1, F * C);
    uniform(l.W2, C * F);
    ones(l.ln1_g, C);
    ones(l.ln2_g, C);
  }
  ones(w.lnf_g, C);
  uniform(w.Wcls, V * C);

  // Buffers for the largest batch.
  const size_t N = static_cast<size_t>(d_.max_B) * kMaxLen;
  const size_t att = static_cast<size_t>(d_.max_B) * d_.H * kMaxLen * kMaxLen;
  acts_.resize(d_.L);
  for (LayerActs &a : acts_) {
    a.x.resize(N * C);
    a.ln1.resize(N * C); a.ln1_mean.resize(N); a.ln1_rstd.resize(N);
    a.qkv.resize(N * 3 * C);
    a.att_p.resize(att);
    a.att_out.resize(N * C);
    a.x_mid.resize(N * C);
    a.ln2.resize(N * C); a.ln2_mean.resize(N); a.ln2_rstd.resize(N);
    a.ff_pre.resize(N * F);
    a.ff_act.resize(N * F);
  }
  x_final_.resize(N * C);
  lnf_.resize(N * C); lnf_mean_.resize(N); lnf_rstd_.resize(N);
  logits_.resize(N * V);
  probs_.resize(N * V);
  dx_.resize(N * C); dln_.resize(N * C); dqkv_.resize(N * 3 * C); datt_.resize(N * C);
  dff_.resize(N * F); dlogits_.resize(N * V); ds_.resize(att); tmp_.resize(N * C);

  g_x_.resize(C); g_h_.resize(C); g_qkv_.resize(3 * C); g_att_.resize(C);
  g_tmp_.resize(C); g_ff_.resize(F); g_p_.resize(kMaxLen);
}

// ------------------------------------------------------------------ forward

LossStats Model::forward(const Batch &bt) {
  const int B = bt.B, T = bt.T, N = B * T, C = d_.C, F = d_.F, H = d_.H, V = d_.V;
  const Weights &w = params.w;

  embedding_forward(bt.tokens.data(), w.E, w.P, acts_[0].x.data(), B, T, C);

  for (int l = 0; l < d_.L; ++l) {
    const LayerWeights &lw = w.layers[l];
    LayerActs &a = acts_[l];
    float *x_out = (l + 1 < d_.L) ? acts_[l + 1].x.data() : x_final_.data();

    // attention sub-block: x_mid = x + Wo · Attn(LN1(x) · Wqkv)
    layernorm_forward(a.x.data(), lw.ln1_g, lw.ln1_b, a.ln1.data(), a.ln1_mean.data(),
                      a.ln1_rstd.data(), N, C);
    linear_forward(a.ln1.data(), lw.Wqkv, nullptr, a.qkv.data(), N, C, 3 * C);
    attention_forward(a.qkv.data(), a.att_p.data(), a.att_out.data(), B, T, C, H);
    linear_forward(a.att_out.data(), lw.Wo, nullptr, tmp_.data(), N, C, C);
    add(a.x.data(), tmp_.data(), a.x_mid.data(), static_cast<size_t>(N) * C);

    // feed-forward sub-block: x_out = x_mid + W2 · ReLU(W1 · LN2(x_mid) + b1) + b2
    layernorm_forward(a.x_mid.data(), lw.ln2_g, lw.ln2_b, a.ln2.data(), a.ln2_mean.data(),
                      a.ln2_rstd.data(), N, C);
    linear_forward(a.ln2.data(), lw.W1, lw.b1, a.ff_pre.data(), N, C, F);
    relu_forward(a.ff_pre.data(), a.ff_act.data(), static_cast<size_t>(N) * F);
    linear_forward(a.ff_act.data(), lw.W2, lw.b2, tmp_.data(), N, F, C);
    add(a.x_mid.data(), tmp_.data(), x_out, static_cast<size_t>(N) * C);
  }

  layernorm_forward(x_final_.data(), w.lnf_g, w.lnf_b, lnf_.data(), lnf_mean_.data(),
                    lnf_rstd_.data(), N, C);
  linear_forward(lnf_.data(), w.Wcls, w.bcls, logits_.data(), N, C, V);
  return softmax_cross_entropy(logits_.data(), bt.targets.data(), probs_.data(), N, V);
}

// ------------------------------------------------------------------ backward

void Model::backward(const Batch &bt) {
  const int B = bt.B, T = bt.T, N = B * T, C = d_.C, F = d_.F, H = d_.H, V = d_.V;
  const Weights &w = params.w;
  Weights &g = grads.w;
  const size_t NC = static_cast<size_t>(N) * C;

  // loss = mean over trained positions
  softmax_cross_entropy_backward(probs_.data(), bt.targets.data(), 1.0f / bt.n_targets,
                                 dlogits_.data(), N, V);
  linear_backward(lnf_.data(), w.Wcls, dlogits_.data(), dln_.data(), g.Wcls, g.bcls, N, C, V);
  std::fill(dx_.begin(), dx_.begin() + NC, 0.0f);
  layernorm_backward_accum(x_final_.data(), w.lnf_g, lnf_mean_.data(), lnf_rstd_.data(),
                           dln_.data(), dx_.data(), g.lnf_g, g.lnf_b, N, C);

  // dx_ always holds the gradient of the residual stream at the current point.
  for (int l = d_.L - 1; l >= 0; --l) {
    const LayerWeights &lw = w.layers[l];
    LayerWeights &lg = g.layers[l];
    const LayerActs &a = acts_[l];

    // feed-forward sub-block (dx_ = d x_out  ->  d x_mid)
    linear_backward(a.ff_act.data(), lw.W2, dx_.data(), dff_.data(), lg.W2, lg.b2, N, F, C);
    relu_backward(a.ff_pre.data(), dff_.data(), dff_.data(), static_cast<size_t>(N) * F);
    linear_backward(a.ln2.data(), lw.W1, dff_.data(), dln_.data(), lg.W1, lg.b1, N, C, F);
    layernorm_backward_accum(a.x_mid.data(), lw.ln2_g, a.ln2_mean.data(), a.ln2_rstd.data(),
                             dln_.data(), dx_.data(), lg.ln2_g, lg.ln2_b, N, C);

    // attention sub-block (dx_ = d x_mid  ->  d x)
    linear_backward(a.att_out.data(), lw.Wo, dx_.data(), datt_.data(), lg.Wo, nullptr, N, C, C);
    attention_backward(a.qkv.data(), a.att_p.data(), datt_.data(), dqkv_.data(), ds_.data(),
                       B, T, C, H);
    linear_backward(a.ln1.data(), lw.Wqkv, dqkv_.data(), dln_.data(), lg.Wqkv, nullptr,
                    N, C, 3 * C);
    layernorm_backward_accum(a.x.data(), lw.ln1_g, a.ln1_mean.data(), a.ln1_rstd.data(),
                             dln_.data(), dx_.data(), lg.ln1_g, lg.ln1_b, N, C);
  }

  embedding_backward(bt.tokens.data(), dx_.data(), g.E, g.P, B, T, C);
}

// ------------------------------------------------------------------ generation

KVCache Model::new_cache() const {
  KVCache c;
  c.k.assign(static_cast<size_t>(d_.L) * kMaxLen * d_.C, 0.0f);
  c.v.assign(static_cast<size_t>(d_.L) * kMaxLen * d_.C, 0.0f);
  return c;
}

// Same computation as forward() for a single row, reusing K/V of positions
// 0..pos-1 from the cache instead of recomputing them.
void Model::decode_step(int token, int pos, KVCache *cache, float *logits) {
  const int C = d_.C, F = d_.F, H = d_.H, V = d_.V, Dh = C / H;
  const Weights &w = params.w;
  float mean, rstd;

  embedding_forward(&token, w.E, w.P + static_cast<size_t>(pos) * C, g_x_.data(), 1, 1, C);
  for (int l = 0; l < d_.L; ++l) {
    const LayerWeights &lw = w.layers[l];
    float *K = cache->k.data() + static_cast<size_t>(l) * kMaxLen * C;
    float *Vc = cache->v.data() + static_cast<size_t>(l) * kMaxLen * C;

    layernorm_forward(g_x_.data(), lw.ln1_g, lw.ln1_b, g_h_.data(), &mean, &rstd, 1, C);
    linear_forward(g_h_.data(), lw.Wqkv, nullptr, g_qkv_.data(), 1, C, 3 * C);
    std::copy(g_qkv_.begin() + C, g_qkv_.begin() + 2 * C, K + static_cast<size_t>(pos) * C);
    std::copy(g_qkv_.begin() + 2 * C, g_qkv_.end(), Vc + static_cast<size_t>(pos) * C);
    for (int h = 0; h < H; ++h)  // PARALLEL: h
      attend_one_query(g_qkv_.data() + h * Dh, K + h * Dh, Vc + h * Dh, C, pos + 1, Dh,
                       g_p_.data(), g_att_.data() + h * Dh);
    linear_forward(g_att_.data(), lw.Wo, nullptr, g_tmp_.data(), 1, C, C);
    add(g_x_.data(), g_tmp_.data(), g_x_.data(), C);

    layernorm_forward(g_x_.data(), lw.ln2_g, lw.ln2_b, g_h_.data(), &mean, &rstd, 1, C);
    linear_forward(g_h_.data(), lw.W1, lw.b1, g_ff_.data(), 1, C, F);
    relu_forward(g_ff_.data(), g_ff_.data(), F);
    linear_forward(g_ff_.data(), lw.W2, lw.b2, g_tmp_.data(), 1, F, C);
    add(g_x_.data(), g_tmp_.data(), g_x_.data(), C);
  }
  layernorm_forward(g_x_.data(), w.lnf_g, w.lnf_b, g_h_.data(), &mean, &rstd, 1, C);
  linear_forward(g_h_.data(), w.Wcls, w.bcls, logits, 1, C, V);
}
