// Run-time configuration (command-line options).
#pragma once

struct Config {
  // ---- data (same meaning as app.cc) ----
  int train_samples = 12000;  // training lines drawn (with replacement) from 460 settings
  int test_samples = 600;     // test lines drawn from the 116 held-out settings
  int eval_exact_n = 600;     // test lines used for greedy generation (exact match)

  // ---- training ----
  int steps = 12000;          // number of training lines (one line = one step, as in app.cc)
  int batch = 8;              // lines per weight update
  float lr = 0.3f;            // peak learning rate
  float momentum = 0.9f;      // SGD momentum
  int warmup = 100;           // linear warm-up, in weight updates
  bool cosine = true;         // cosine decay to 0 after warm-up
  float clip_norm = 5.0f;     // global gradient-norm clipping

  // ---- model ----
  int n_layer = 2;
  int n_head = 4;
  int d_model = 64;
  int d_ff = 256;

  int seed = 42;
  int print_every = 1000;     // in training lines
};

bool parse_args(int argc, char **argv, Config *cfg);
void print_usage(const char *prog);
void print_config(const Config &cfg);
