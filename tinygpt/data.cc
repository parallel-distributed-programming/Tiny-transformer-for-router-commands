#include "data.h"

#include <algorithm>
#include <random>

namespace {

const char *kCmdMark = "[CMD] ";

// Unchanged from app.cc.
Sample make_sample(std::mt19937 &rng) {
  std::uniform_int_distribution<int> b01(0, 1);

  const int channels[3] = {1, 6, 11};
  const char *dnses[4] = {"1.1.1.1", "8.8.8.8", "9.9.9.9", "8.8.4.4"};
  const char *firewall_modes[2] = {"normal", "strict"};
  const int powers[3] = {40, 70, 100};
  const int clients[4] = {8, 16, 32, 64};

  int guest = b01(rng);
  int channel = channels[rng() % 3];
  const char *dns = dnses[rng() % 4];
  const char *fw = firewall_modes[rng() % 2];
  int tx_power = powers[rng() % 3];
  int max_client = clients[rng() % 4];

  std::string intent =
      std::string("set guest wifi ") + (guest ? "on" : "off") +
      ", set channel " + std::to_string(channel) + ", set dns " + dns + ", firewall " + fw +
      ", tx power " + std::to_string(tx_power) + ", max clients " + std::to_string(max_client);

  std::string command =
      std::string("set wifi.guest=") + (guest ? "1" : "0") + "; " +
      "set wifi.channel=" + std::to_string(channel) + "; " + "set wan.dns=" + dns + "; " +
      "set firewall.mode=" + fw + "; " + "set wifi.tx_power=" + std::to_string(tx_power) +
      "; " + "set wifi.max_clients=" + std::to_string(max_client) + "; " + "commit;";

  Sample s;
  s.intent = intent;
  s.command = command;
  s.line = "[INTENT] " + intent + " " + kCmdMark + command + "\n";
  return s;
}

}  // namespace

// Unchanged from app.cc: all 576 settings, 460 for training and 116 for testing.
std::vector<Sample> make_dataset(int n, int seed, bool test_split) {
  std::mt19937 generator(12345);
  std::vector<Sample> unique;
  std::unordered_map<std::string, bool> seen;
  while (unique.size() < 576) {
    Sample sample = make_sample(generator);
    if (seen.emplace(sample.intent, true).second) unique.push_back(sample);
  }
  std::mt19937 splitter(seed);
  std::shuffle(unique.begin(), unique.end(), splitter);
  const int boundary = 460;
  std::mt19937 sampler(seed + (test_split ? 1 : 0));
  std::uniform_int_distribution<int> pick(test_split ? boundary : 0,
                                          test_split ? 575 : boundary - 1);
  std::vector<Sample> samples;
  samples.reserve(n);
  for (int i = 0; i < n; ++i) samples.push_back(unique[pick(sampler)]);
  return samples;
}

Vocab build_vocab(const std::vector<Sample> &samples) {
  Vocab v;
  auto add = [&](char c) {
    if (v.char_to_id.count(c)) return;
    v.char_to_id[c] = v.size();
    v.id_to_char.push_back(c);
  };
  v.id_to_char.push_back('\0');  // padding
  add('\n');
  add('[');
  add(']');
  for (const auto &s : samples)
    for (char c : s.line) add(c);
  return v;
}

std::vector<int> encode(const std::string &s, const Vocab &v) {
  std::vector<int> ids;
  ids.reserve(s.size());
  for (char c : s) {
    auto it = v.char_to_id.find(c);
    if (it != v.char_to_id.end()) ids.push_back(it->second);
  }
  return ids;
}

std::vector<Example> encode_dataset(const std::vector<Sample> &samples, const Vocab &v) {
  std::vector<Example> out;
  out.reserve(samples.size());
  for (const auto &s : samples) {
    Example e;
    e.ids = encode(s.line, v);
    e.cmd_start = static_cast<int>(s.line.find(kCmdMark) + std::string(kCmdMark).size());
    out.push_back(std::move(e));
  }
  return out;
}

Batch make_batch(const std::vector<const Example *> &examples) {
  Batch bt;
  bt.B = static_cast<int>(examples.size());
  for (const Example *e : examples) bt.T = std::max(bt.T, static_cast<int>(e->ids.size()));
  bt.tokens.assign(static_cast<size_t>(bt.B) * bt.T, kPadId);
  bt.targets.assign(static_cast<size_t>(bt.B) * bt.T, -1);
  for (int b = 0; b < bt.B; ++b) {
    const std::vector<int> &ids = examples[b]->ids;
    const int len = static_cast<int>(ids.size());
    for (int t = 0; t < len; ++t) bt.tokens[static_cast<size_t>(b) * bt.T + t] = ids[t];
    // Position t predicts character t+1; only command characters are trained.
    for (int t = examples[b]->cmd_start - 1; t + 1 < len; ++t) {
      bt.targets[static_cast<size_t>(b) * bt.T + t] = ids[t + 1];
      ++bt.n_targets;
    }
  }
  return bt;
}

std::string prompt_of(const Sample &s) { return "[INTENT] " + s.intent + " " + kCmdMark; }
