#include "config.h"

#include <cstring>
#include <iostream>
#include <string>

namespace {

bool parse_int(const char *s, int *out) {
  try { *out = std::stoi(s); return true; } catch (...) { return false; }
}
bool parse_float(const char *s, float *out) {
  try { *out = std::stof(s); return true; } catch (...) { return false; }
}

struct IntOpt { const char *name; int Config::*field; };
struct FloatOpt { const char *name; float Config::*field; };

const IntOpt kIntOpts[] = {
    {"--train-samples", &Config::train_samples}, {"--test-samples", &Config::test_samples},
    {"--eval-exact-n", &Config::eval_exact_n},   {"--steps", &Config::steps},
    {"--batch", &Config::batch},                 {"--warmup", &Config::warmup},
    {"--n-layer", &Config::n_layer},             {"--n-head", &Config::n_head},
    {"--d-model", &Config::d_model},             {"--d-ff", &Config::d_ff},
    {"--seed", &Config::seed},                   {"--print-every", &Config::print_every},
};
const FloatOpt kFloatOpts[] = {
    {"--lr", &Config::lr}, {"--momentum", &Config::momentum}, {"--clip-norm", &Config::clip_norm},
};

}  // namespace

bool parse_args(int argc, char **argv, Config *cfg) {
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (std::strcmp(a, "--no-cosine") == 0) { cfg->cosine = false; continue; }
    if (i + 1 >= argc) return false;
    bool ok = false;
    for (const auto &o : kIntOpts)
      if (std::strcmp(a, o.name) == 0) ok = parse_int(argv[++i], &(cfg->*o.field));
    for (const auto &o : kFloatOpts)
      if (std::strcmp(a, o.name) == 0) ok = parse_float(argv[++i], &(cfg->*o.field));
    if (!ok) return false;
  }
  return cfg->train_samples > 0 && cfg->test_samples > 0 && cfg->eval_exact_n > 0 &&
         cfg->steps > 0 && cfg->batch > 0 && cfg->lr > 0 && cfg->n_layer > 0 &&
         cfg->n_head > 0 && cfg->d_model % cfg->n_head == 0 && cfg->d_ff > 0 &&
         cfg->print_every > 0 && cfg->warmup >= 0;
}

void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog << " [options]\n";
  for (const auto &o : kIntOpts) std::cerr << "  " << o.name << " N\n";
  for (const auto &o : kFloatOpts) std::cerr << "  " << o.name << " X\n";
  std::cerr << "  --no-cosine\n";
}

void print_config(const Config &c) {
  std::cout << "Model=TinyGPT(causal, pre-LN)"
            << " n_layer=" << c.n_layer << " n_head=" << c.n_head << " d_model=" << c.d_model
            << " d_ff=" << c.d_ff << "\n"
            << "steps=" << c.steps << " batch=" << c.batch << " lr=" << c.lr
            << " momentum=" << c.momentum << " warmup=" << c.warmup
            << " cosine=" << c.cosine << " clip_norm=" << c.clip_norm
            << " train_samples=" << c.train_samples << " test_samples=" << c.test_samples
            << " eval_exact_n=" << c.eval_exact_n << " seed=" << c.seed << "\n";
}
