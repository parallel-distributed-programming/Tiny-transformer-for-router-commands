// Test metrics (the same three numbers as app.cc).
#pragma once

#include <string>
#include <vector>

#include "data.h"
#include "model.h"

// Mean loss and next-character accuracy over the command characters,
// with the correct previous characters as input (teacher forcing).
LossStats teacher_forced_eval(Model *model, const std::vector<Example> &test, int batch);

// Greedy decoding from "[INTENT] ... [CMD] " until '\n' (or max_new characters).
// PARALLEL: different samples are independent (each has its own KV cache).
std::string greedy_generate(Model *model, const Sample &s, const Vocab &v, int max_new);
