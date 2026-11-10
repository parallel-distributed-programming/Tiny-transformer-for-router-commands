# Tiny Transformer for Router Commands

Team 5A project for the **[Parallel and Distributed Programming-Kenjiro Taura](https://taura.github.io/parallel-distributed-programming/)**. The goal is to
parallelize the existing C++17 implementation and measure the speedup while
preserving its model, inputs, training procedure, and evaluation protocol.

## Task and data

The program generates all examples internally. The command syntax is a teaching
format for this experiment. The program neither connects to nor configures routers.

| Setting | Possible values | Target field |
| --- | --- | --- |
| Guest Wi-Fi | on, off | `wifi.guest` (on becomes `1`, off becomes `0`) |
| Wi-Fi channel | 1, 6, 11 | `wifi.channel` |
| DNS server | 1.1.1.1, 8.8.8.8, 9.9.9.9, 8.8.4.4 | `wan.dns` |
| Firewall mode | normal, strict | `firewall.mode` |
| Transmit power | 40, 70, 100 | `wifi.tx_power` |
| Maximum clients | 8, 16, 32, 64 | `wifi.max_clients` |

The generator splits the 576 possible configurations into 460 training and 116
test configurations. It draws samples with replacement from each disjoint split.
For example, the benchmark's 116 test samples can repeat configurations from the
test split. The serial baseline uses records of the form
`[INTENT] request [CMD] command` followed by a newline. The model processes one
character per token.

## Build and run

Use a C++17 compiler. The program uses only the C++ standard library.

```sh
c++ -O3 -std=c++17 app.cc -o app
./app --train-samples 12000 --test-samples 116 --steps 1000 \
  --ctx 256 --d-model 48 --d-ff 96 --lr 0.005 --clip-norm 5 \
  --seed 42 --print-every 100 --eval-exact-n 116
```

These flags match the reported Miyabi run. With no flags, the program draws
1,200 test samples and caps exact-match evaluation at 200 samples.

| Option | Meaning |
| --- | --- |
| `--train-samples`, `--test-samples` | Number of sampled examples in each split |
| `--steps` | Number of training sample selections |
| `--ctx` | Maximum number of preceding characters in the context |
| `--d-model`, `--d-ff` | Model and feed-forward hidden widths |
| `--lr`, `--clip-norm` | SGD learning rate and global gradient norm limit |
| `--seed` | Random seed |
| `--print-every` | Training log interval in steps |
| `--eval-exact-n` | Maximum number of test samples used for greedy exact-match evaluation |

## Model and training

- **Embeddings** — learned character and position embeddings
- **Attention** — 1 head, query at the last position, padding mask
- **Transformer block** — 1 block, ReLU feed-forward network, 2 residual connections
- **Layer normalization** — none
- **Vocabulary** — 45 tokens, including padding
- **Model width** — 48
- **Feed-forward width** — 96
- **Parameters** — 35,229 at 256-character context · 24,477 at 32-character context

Each training step selects one sample with replacement. Using teacher forcing,
the model receives the correct preceding characters and predicts each command
character and the terminating newline in turn. Backpropagation computes
gradients, global norm clipping limits their magnitude, and serial SGD updates
the weights after each target character.

## Reported serial baseline on Miyabi

| Measurement | Result |
| --- | ---: |
| Training time | 166,173.283 ms (166.173283 s) |
| Character-level weight updates | 140,431 |
| Updates with gradient clipping | 77,629 |
| Character updates per second | 845.088 |
| Test command teacher-forced loss | 1.286 |
| Test command next-character accuracy (teacher-forced) | 50.6% |
| Exact command match (greedy generation) | 0/116 (0%) |
| Teacher-forced evaluation time | 7,247.528 ms |
| Generation time | 11,332.144 ms |

The training timer covers context construction, forward and backward passes,
gradient clipping, SGD updates, and training logs. Compilation, data and model
setup, evaluation, and generation are outside that timer. Average training loss
fell from about 2.81 over the first 100 steps to 1.38 over the last 100.
The reported 50.6% is next-character accuracy with the request and the correct
preceding command characters in the input. In the separate greedy generation
evaluation, the model produced repetitions such as `set set set ...` and matched
0 of 116 complete commands.

## Team 5A parallelization plan

Parallelize the computations within the existing training procedure. Preserve
the per-character SGD update schedule and the original evaluation inputs.

- **Matrix operations**
- **Backpropagation**
- **Batching (optional)** — group independent computations within an update
