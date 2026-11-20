// TinyGPT: serial baseline. Train a small causal transformer to turn a router
// "intent" sentence into a command line, then evaluate on unseen settings.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

#include "config.h"
#include "data.h"
#include "evaluate.h"
#include "model.h"
#include "optimizer.h"

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

void train(const Config &cfg, Model *model, const std::vector<Example> &train_set) {
  const int n_updates = (cfg.steps + cfg.batch - 1) / cfg.batch;
  MomentumSGD opt(model->params.size(), cfg.momentum);
  std::mt19937 rng(cfg.seed + 123);
  std::uniform_int_distribution<int> pick(0, static_cast<int>(train_set.size()) - 1);

  double run_loss = 0.0;
  long run_targets = 0, total_targets = 0;
  int clipped = 0, done = 0;
  const auto t0 = Clock::now();
  for (int u = 0; u < n_updates; ++u) {
    // 1. draw a mini-batch of lines
    const int B = std::min(cfg.batch, cfg.steps - done);
    std::vector<const Example *> group;
    for (int b = 0; b < B; ++b) group.push_back(&train_set[pick(rng)]);
    const Batch batch = make_batch(group);

    // 2. forward + backward (gradient of the mean loss over the batch)
    std::fill(model->grads.data.begin(), model->grads.data.end(), 0.0f);
    const LossStats st = model->forward(batch);
    model->backward(batch);

    // 3. clip and update
    float *g = model->grads.data.data();
    const size_t n = model->grads.size();
    const double norm = grad_norm(g, n);
    if (!std::isfinite(norm)) throw std::runtime_error("non-finite gradient");
    if (norm > cfg.clip_norm) {
      scale_grad(g, n, static_cast<float>(cfg.clip_norm / norm));
      ++clipped;
    }
    opt.step(model->params.data.data(), g, learning_rate(cfg, u, n_updates));

    // 4. report
    run_loss += st.loss_sum;
    run_targets += st.n_targets;
    total_targets += st.n_targets;
    const int prev = done;
    done += B;
    if (done / cfg.print_every != prev / cfg.print_every || done == cfg.steps) {
      std::cout << "step " << done << " avg_train_loss=" << run_loss / run_targets
                << " lr=" << learning_rate(cfg, u, n_updates)
                << " time_s=" << seconds_since(t0) << std::endl;
      run_loss = 0.0;
      run_targets = 0;
    }
  }
  const double sec = seconds_since(t0);
  std::cout << "training_time_ms=" << sec * 1000.0 << "\n"
            << "weight_updates=" << n_updates << " clipped_updates=" << clipped << "\n"
            << "training_characters_per_second=" << total_targets / sec << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  Config cfg;
  if (!parse_args(argc, argv, &cfg)) {
    print_usage(argv[0]);
    return 1;
  }
  std::cout << std::fixed << std::setprecision(4);
  print_config(cfg);

  const std::vector<Sample> train_samples = make_dataset(cfg.train_samples, cfg.seed, false);
  const std::vector<Sample> test_samples = make_dataset(cfg.test_samples, cfg.seed, true);
  const Vocab vocab = build_vocab(train_samples);
  const std::vector<Example> train_set = encode_dataset(train_samples, vocab);
  const std::vector<Example> test_set = encode_dataset(test_samples, vocab);

  const ModelDims dims{vocab.size(), cfg.d_model, cfg.n_head, cfg.d_ff, cfg.n_layer, cfg.batch};
  Model model(dims, cfg.seed + 7);
  std::cout << "vocab_size=" << vocab.size() << " parameters=" << model.params.size() << "\n";

  train(cfg, &model, train_set);

  auto t0 = Clock::now();
  const LossStats st = teacher_forced_eval(&model, test_set, cfg.batch);
  std::cout << "test_command_teacher_forced_loss=" << st.loss_sum / st.n_targets << "\n"
            << "test_command_next_char_accuracy="
            << static_cast<double>(st.n_correct) / st.n_targets << "\n"
            << "teacher_forced_eval_time_ms=" << seconds_since(t0) * 1000.0 << "\n";

  t0 = Clock::now();
  const int n_eval = std::min(cfg.eval_exact_n, static_cast<int>(test_samples.size()));
  int exact = 0;
  for (int i = 0; i < n_eval; ++i) {
    const std::string pred = greedy_generate(&model, test_samples[i], vocab, 220);
    if (pred == test_samples[i].command) ++exact;
    if (i < 3)
      std::cout << "example_intent=" << test_samples[i].intent
                << "\nexample_expected=" << test_samples[i].command
                << "\nexample_generated=" << pred << "\n";
  }
  std::cout << "exact_command_match_rate=" << static_cast<double>(exact) / n_eval << " ("
            << exact << "/" << n_eval << ")\n"
            << "generation_time_ms=" << seconds_since(t0) * 1000.0 << "\n";
  return 0;
}
