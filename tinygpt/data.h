// Synthetic "router intent -> command" dataset (same generator as app.cc),
// character vocabulary, and padded mini-batches.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

constexpr int kPadId = 0;
constexpr int kMaxLen = 256;  // longest line, including the final '\n'

struct Sample {
  std::string intent;
  std::string command;
  std::string line;  // "[INTENT] <intent> [CMD] <command>\n"
};

struct Vocab {
  std::vector<char> id_to_char;            // id 0 = padding
  std::unordered_map<char, int> char_to_id;
  int size() const { return static_cast<int>(id_to_char.size()); }
};

// One encoded line. Characters ids[cmd_start .. end) are the ones we learn to predict.
struct Example {
  std::vector<int> ids;
  int cmd_start;
};

// A mini-batch of B lines, right-padded to the longest line (T characters).
//   tokens [b*T + t]  : character t of line b (kPadId after the end)
//   targets[b*T + t]  : the character that should follow position t, or -1
//                       when position t is not trained (prompt part / padding)
struct Batch {
  int B = 0, T = 0;
  std::vector<int> tokens;
  std::vector<int> targets;
  int n_targets = 0;
};

std::vector<Sample> make_dataset(int n, int seed, bool test_split);
Vocab build_vocab(const std::vector<Sample> &samples);
std::vector<int> encode(const std::string &s, const Vocab &v);
std::vector<Example> encode_dataset(const std::vector<Sample> &samples, const Vocab &v);
Batch make_batch(const std::vector<const Example *> &examples);
std::string prompt_of(const Sample &s);  // "[INTENT] <intent> [CMD] "
