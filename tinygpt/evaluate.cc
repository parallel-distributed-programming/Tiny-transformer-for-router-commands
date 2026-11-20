#include "evaluate.h"

#include <algorithm>

LossStats teacher_forced_eval(Model *model, const std::vector<Example> &test, int batch) {
  LossStats total;
  for (size_t i = 0; i < test.size(); i += batch) {
    std::vector<const Example *> group;
    for (size_t j = i; j < std::min(test.size(), i + batch); ++j) group.push_back(&test[j]);
    LossStats st = model->forward(make_batch(group));
    total.loss_sum += st.loss_sum;
    total.n_targets += st.n_targets;
    total.n_correct += st.n_correct;
  }
  return total;
}

std::string greedy_generate(Model *model, const Sample &s, const Vocab &v, int max_new) {
  const int V = model->dims().V;
  std::vector<float> logits(V);
  KVCache cache = model->new_cache();
  std::vector<int> prompt = encode(prompt_of(s), v);

  // Feed the prompt (prefill), then feed back each predicted character.
  int pos = 0;
  for (int tok : prompt) model->decode_step(tok, pos++, &cache, logits.data());
  std::string out;
  for (int step = 0; step < max_new && pos < kMaxLen; ++step) {
    const int next = static_cast<int>(std::max_element(logits.begin(), logits.end()) -
                                      logits.begin());
    const char c = v.id_to_char[next];
    if (c == '\n') break;
    out.push_back(c);
    model->decode_step(next, pos++, &cache, logits.data());
  }
  return out;
}
