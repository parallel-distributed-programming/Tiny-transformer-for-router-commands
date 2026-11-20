// Correctness checks for the model code (run them again after parallelizing):
//   1. backward() against central finite differences of forward();
//   2. decode_step() (KV cache) against forward() on the same line;
//   3. optional: --dump FILE writes weights, batch, loss and gradients so that
//      tools/compare_with_torch.py can compare them with PyTorch autograd.
//
// About 1: in float precision and with ReLU kinks, finite differences are only
// approximate, so the check passes when most sampled parameters agree.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "data.h"
#include "model.h"

namespace {

void dump(const char *path, const Model &m, const Batch &bt, double loss) {
  FILE *f = std::fopen(path, "wb");
  const ModelDims &d = m.dims();
  const int32_t hdr[8] = {d.V, d.C, d.H, d.F, d.L, bt.B, bt.T,
                          static_cast<int32_t>(m.params.size())};
  std::fwrite(hdr, sizeof(int32_t), 8, f);
  std::vector<int32_t> tok(bt.tokens.begin(), bt.tokens.end());
  std::vector<int32_t> tgt(bt.targets.begin(), bt.targets.end());
  std::fwrite(tok.data(), sizeof(int32_t), tok.size(), f);
  std::fwrite(tgt.data(), sizeof(int32_t), tgt.size(), f);
  std::fwrite(m.params.data.data(), sizeof(float), m.params.size(), f);
  std::fwrite(m.grads.data.data(), sizeof(float), m.grads.size(), f);
  std::fwrite(&loss, sizeof(double), 1, f);
  std::fclose(f);
  std::printf("wrote %s\n", path);
}

}  // namespace

int main(int argc, char **argv) {
  const char *dump_path = (argc == 3 && std::strcmp(argv[1], "--dump") == 0) ? argv[2] : nullptr;

  const std::vector<Sample> samples = make_dataset(3, 42, false);
  const Vocab vocab = build_vocab(samples);
  const std::vector<Example> ex = encode_dataset(samples, vocab);
  std::vector<const Example *> group{&ex[0], &ex[1], &ex[2]};
  const Batch batch = make_batch(group);  // 3 lines of different lengths (padding is tested)

  // small model: 2 layers, 2 heads of 4 dims, d_ff 16
  const ModelDims dims{vocab.size(), 8, 2, 16, 2, batch.B};
  Model model(dims, 1);
  auto loss = [&]() {
    const LossStats st = model.forward(batch);
    return st.loss_sum / st.n_targets;
  };

  // ---- 1. gradient check ----
  std::fill(model.grads.data.begin(), model.grads.data.end(), 0.0f);
  const double loss0 = loss();
  model.backward(batch);
  if (dump_path) dump(dump_path, model, batch, loss0);
  const std::vector<float> analytic = model.grads.data;

  std::mt19937 rng(0);
  std::uniform_int_distribution<size_t> pick(0, model.params.size() - 1);
  const float eps = 1e-2f;
  int good = 0, n_checked = 0;
  for (int k = 0; k < 400; ++k) {
    const size_t i = pick(rng);
    float &w = model.params.data[i];
    const float w0 = w;
    w = w0 + eps; const double lp = loss();
    w = w0 - eps; const double lm = loss();
    w = w0;
    const double numeric = (lp - lm) / (2 * eps);
    if (std::fabs(numeric) < 1e-6 && std::fabs(analytic[i]) < 1e-6) continue;  // unused param
    const double err = std::fabs(numeric - analytic[i]) /
                       std::max(1e-3, std::fabs(numeric) + std::fabs(analytic[i]));
    good += err < 5e-2;
    ++n_checked;
  }
  const bool grad_ok = good >= 0.9 * n_checked;
  std::printf("gradient check: %d/%d sampled params within 5%% of finite differences -> %s\n",
              good, n_checked, grad_ok ? "OK" : "FAILED");

  // ---- 2. KV-cache generation path == batched forward ----
  model.forward(batch);
  std::vector<float> logits(dims.V);
  KVCache cache = model.new_cache();
  double max_diff = 0.0;
  for (int t = 0; t < static_cast<int>(ex[0].ids.size()); ++t) {  // line 0 = rows 0..len-1
    model.decode_step(ex[0].ids[t], t, &cache, logits.data());
    for (int o = 0; o < dims.V; ++o)
      max_diff = std::max(max_diff, static_cast<double>(std::fabs(
                                        logits[o] - model.logits()[t * dims.V + o])));
  }
  const bool decode_ok = max_diff < 1e-4;
  std::printf("decode_step vs forward: max |logit difference| %.2e -> %s\n", max_diff,
              decode_ok ? "OK" : "FAILED");
  return (grad_ok && decode_ok) ? 0 : 1;
}
