# TinyGPT — 直列ベースライン（C++17, 標準ライブラリのみ）

app.cc と同じ課題（ルータ設定の意図文 → コマンド文字列）を、予備実験で決めた
causal transformer で解く直列版。予備実験（PyTorch）はこのブランチには含めていない。OpenMP / CUDA で並列化する出発点として書いている。

## ビルドと実行

```bash
module purge && module load nvidia/25.9 cuda/12.9 && PATH=/work/gt81/share/env/llvm/bin:$PATH
make            # tinygpt（本体）と check（正しさの検査）
./check         # 勾配と生成経路の検査
./tinygpt       # 推奨設定で学習 → 評価（計算ノードで: qsub run_job.sh）
```

主なオプション（既定値 = 推奨設定）: `--steps 12000 --batch 8 --lr 0.3 --momentum 0.9 --warmup 100
--n-layer 2 --n-head 4 --d-model 64 --d-ff 256 --test-samples 600 --eval-exact-n 600 --seed 42`
（`--no-cosine` で cosine 減衰なし）。

## 結果（Miyabi 計算ノード, Grace 1 コア, clang++ -O3）

| 項目 | 値 |
|---|---|
| 学習 | 12,000 steps（1,500 回の重み更新）で **254 s**、6,640 文字/s |
| teacher-forced 評価（600 件） | 7.7 s、次文字正解率 1.0000 |
| 生成（600 件, KV キャッシュ） | 7.6 s、**完全一致 600/600** |
| 全体 | 269 s |

（参考: app.cc のデフォルトは全体 204 s で完全一致 0/200）

## ファイル

| ファイル | 内容 |
|---|---|
| `config.h/.cc` | コマンドライン引数 |
| `data.h/.cc` | データ生成（app.cc と同一）、語彙、パディング付きミニバッチ `Batch` |
| `ops.h/.cc` | 演算カーネル: embedding, linear, LayerNorm, ReLU, causal attention, softmax+交差エントロピー（各 forward / backward） |
| `model.h/.cc` | パラメータ配置、forward / backward、生成用 `decode_step`（KV キャッシュ） |
| `optimizer.h/.cc` | 勾配ノルムのクリップ、モーメンタム SGD、学習率（warm-up + cosine） |
| `evaluate.h/.cc` | teacher-forced 評価、貪欲生成 |
| `main.cc` | 学習ループと評価の呼び出し、時間計測 |
| `check.cc` | 検査: 有限差分との勾配比較、`decode_step` と `forward` の一致、`--dump` |
| `tools/compare_with_torch.py` | `./check --dump f` の重み・勾配を PyTorch autograd（float64）と比較 |

## 計算の流れ

1 回の重み更新（`main.cc` の `train`）:

1. 学習用 460 設定から 8 行を引いて `Batch` を作る（B=8, T=最長行 ≤ 256, N=B·T 行）
2. `Model::forward`: 下表の順に計算し、逆伝播に必要な中間値を保存
3. `Model::backward`: 逆順に勾配を計算（パラメータ勾配は `+=` で蓄積）
4. 勾配ノルムが 5 を超えたら縮小 → モーメンタム SGD で更新

| 段 | 関数 | 形 | 1 行あたり FLOP の目安 |
|---|---|---|---|
| 埋め込み | `embedding_forward` | (N, 64) | – |
| ×2 層: LN1 → QKV | `layernorm_forward`, `linear_forward` | (N,64)→(N,192) | 2·64·192 |
| 　 attention（4 ヘッド × 16 次元, 因果） | `attention_forward` | (B,4,T,T) | 4·64·(t+1) |
| 　 出力射影 + 残差 | `linear_forward`, `add` | (N,64) | 2·64·64 |
| 　 LN2 → FFN（64→256→64, ReLU）+ 残差 | | (N,256) | 2·2·64·256 |
| 最終 LN → 出力 | `linear_forward` | (N,45) | 2·64·45 |
| 損失 | `softmax_cross_entropy` | コマンド部分の位置だけ | – |

backward は forward の約 2 倍。学習時間の大半は linear（行列積）と attention。

パラメータは `ParamSet::data`（float 1 本, 121,773 要素）にまとめて置き、各テンソルはそこへの
ポインタ（`Weights`）。勾配 `grads` も同じ配置なので、勾配のゼロ化・ノルム・更新は 1 重ループ。
GPU へは 1 回の転送で済む。

## 並列化の手がかり

`ops.cc` の各ループに `PARALLEL:`（書き込みが衝突しない軸）と `REDUCTION:`（和をとるので
並列化に注意が要る箇所）を書いてある。主なもの:

| 箇所 | 並列軸 | 注意 |
|---|---|---|
| `linear_forward` | 行 n × 出力 o | 行列積そのもの |
| `linear_backward` の dx | 行 n | – |
| `linear_backward` の dW, db | 出力 o | n についての和（o で分ければ衝突しない） |
| `attention_forward` | バッチ b × ヘッド h × クエリ i | i ごとに仕事量が違う（因果マスクで i+1 個） |
| `attention_backward` | b × h、内側は 3 パスに分けてある | パス 1・2 はクエリ i、パス 3 はキー j で分割すれば衝突なし |
| `layernorm_*` | 行 n（dg, db は列 c） | 行内の和 |
| `embedding_backward` | 列 c | 同じ文字・同じ位置への加算が衝突する |
| `grad_norm` | 要素 i | 全体の和 |
| `MomentumSGD::step`, `scale_grad` | 要素 i | – |
| 評価・生成 | テスト行ごと | 生成は行ごとに KV キャッシュが要る |

重みの更新は毎回の更新が前の更新に依存するので逐次。並列化するのは 1 回の更新の中
（バッチ・位置・ヘッド・行列積）と、評価・生成。

## 正しさの確認

- `./check`:
  1. 勾配と中心差分の比較（float と ReLU の折れ目のため近似。90% 以上が 5% 以内なら OK）
  2. KV キャッシュを使う `decode_step` と、まとめて計算する `forward` の logits の一致
- `./check --dump f && python tools/compare_with_torch.py f`: PyTorch autograd との比較。
  現在の実装は全テンソルで相対誤差 < 1e-6。

並列化したら、まずこの 2 つと「学習ロスの推移・完全一致率が直列版とほぼ同じか」を確認する。
（並列化で和の順序が変わるので、ビット単位では一致しない。）
