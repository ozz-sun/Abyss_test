# Abyss マニュアル

Abyss のビルド、DeepSeek V4 Flash（と V4.1 Flash）の準備、起動と調整の手順。English: [MANUAL.md](MANUAL.md)

Abyss は [Project Maya](https://github.com/mw00/project-maya)（MIT）をコピーして作ったフォークで、Maya は
[Strata](https://github.com/Niko1221/Strata)（MIT）から生まれたプロジェクト。エンジン、エキスパートの配置の仕組み、
サーバー、ダッシュボードは Maya のもので、Abyss はそこに DeepSeek V4 と V4.1 への対応を足した。Maya の元のドキュメント
（GLM-5.3-Flash、AMD、複数 GPU、インストーラ）は [docs/maya/](maya/README.md) にそのまま残している。

- [必要なもの](#必要なもの)
- [まとめてセットアップ](#まとめてセットアップ)
- [ビルド](#ビルド)
- [モデルを取る](#モデルを取る)
- [pack を作る](#pack-を作る)
- [設定と起動](#設定と起動)
- [使い方](#使い方)
- [DeepSeek V4.1 Flash](#deepseek-v41-flash)
- [Ornith-1.5-122B](#ornith-15-122b)
- [EXL3 量子化（GLM-5.3-Flash）](#exl3-量子化glm-53-flash)
- [メモリと速度](#メモリと速度)
- [DSpark のドラフトモデル（DeepSeek V4 Flash）](#dspark-のドラフトモデルdeepseek-v4-flash)
- [設定項目](#設定項目)
- [llama.cpp との照合](#llamacpp-との照合)
- [うまくいかないとき](#うまくいかないとき)
- [制限](#制限)

## 必要なもの

| | 確認した環境 | 補足 |
| --- | --- | --- |
| OS | Ubuntu (Linux) | Windows は DeepSeek V4 では未確認 |
| GPU | RTX 5090 (32 GB)、RTX 3090 | GPU のアーキテクチャに合わせてビルドする（RTX 50 は `120a`、RTX 30 は `86`。分割するときは複数まとめて）。Blackwell より前の GPU では MXFP4 のエキスパートは FP4 MMA ではなく q8_1 の経路で計算する。Ampere（sm_80）より前では indexer と prompt の attention は従来の FP32 カーネルを使う。V100（sm_70）は CUDA 12 でビルドできるが実機は未確認。AMD Radeon は「AMD の GPU」（試験的） |
| RAM | 94 GB | モデルは 96.8 GB。VRAM + RAM が約 100 GB を超えれば全エキスパートがメモリに載る。足りない分は SSD から読む（動くが遅い） |
| ディスク | NVMe SSD、空き約 100 GB | GGUF ファイル。pack は 0.2 GB |
| CUDA | 13.3 | `nvcc` を含む。V100 は 12.9（`--nvcc`） |
| コンパイラ | g++ 13 | CUDA が受け付けるバージョン |
| Python | 3.12 | Abyss のフォルダ内の `.venv` |
| ROCm（任意） | 7（`hipcc`） | AMD APU の内蔵 GPU が RAM 層のエキスパートを計算する（下の「iGPU レーン」）。なければ CPU が計算する |

`./abyss.sh`（Maya のインストーラの名前を変えたもの）は、今も Maya の GLM-5.3-Flash を入れるためのもの。
DeepSeek V4 は `./setup-deepseek.sh`（次の節）か、その後の手順を手で行って準備する。

## まとめてセットアップ

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --download-model
```

`setup-deepseek.sh` は `.venv` を作ってから `tools/abyss_setup.py` を実行する。済んだ手順は飛ばすので、何度実行してもよい:

1. PC の確認（NVIDIA GPU、RAM）
2. Python パッケージ、固定コミットの llama.cpp のソース
3. GPU に合わせてエンジンをビルド（RTX 50 は `120a`）
4. モデルを `../Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/<量子化>` にダウンロード（`--download-model` が
   なければ先に確認する）。全ファイルを Hugging Face の SHA-256 と照合する
5. シャードと同じフォルダに pack を作る
6. Abyss のフォルダに設定 `abyss-<名前>.json` と起動スクリプト `run-abyss-<名前>.sh` を書く
7. DSpark のドラフトモデル（v1.3.43、任意）: 設定が新しいときは、DeepSeek の DSpark ドラフトモデル（6.9 GB）をダウンロードして
   使うかを確認する（`--dspark` なら確認なし）。SHA-256 を照合し、設定の `env` に書く
   （[DSpark のドラフトモデル](#dspark-のドラフトモデルdeepseek-v4-flash)）
8. Abyss の隣に **Abyss-tools** フォルダを作る（`abyss.sh start | stop | restart | status | update | build | log | models`。
   雛形は本体の `ops/abyss-tools/`）。セットアップを再実行すると `abyss.sh` / `common.sh` は雛形に合わせて更新し
   （前のものは `.bak` に残す）、自分で書いた `README.md` はそのまま残す

```sh
../Abyss-tools/abyss.sh start      # 起動後 http://127.0.0.1:8082/
../Abyss-tools/abyss.sh stop
```

| オプション | |
| --- | --- |
| `--quant UD-IQ3_XXS` | unsloth/DeepSeek-V4-Flash-0731-GGUF のほかの量子化（既定 `UD-Q2_K_XL`。確認済みはこれだけ） |
| `--name 名前` | `abyss-<名前>.json` の名前（既定 `dsv4-<量子化>`。例 `dsv4-q2_k_xl`） |
| `--context N`, `--port N` | コンテキスト長（既定 65536）とポート（既定 8082） |
| `--data-dir DIR`, `--tools-dir DIR` | Abyss-data と Abyss-tools を別の場所にする |
| `--gpu N`, `--arch A` | 使う GPU、コンパイルする CUDA アーキテクチャ |
| `--nvcc PATH` | ビルドに使う CUDA コンパイラ（V100 には CUDA 12 のもの） |
| `--gpus 0,1`, `--gpus all` | 複数の GPU に層を分けて載せる（下の「複数の GPU」）。全 GPU のアーキテクチャをまとめてビルドする |
| `--download-model`, `--yes` | 確認せずにダウンロード、すべての質問に既定値で答える |
| `--rewrite-config` | `abyss-<名前>.json` を書き直す（なければ既存のものを残す。書いてあった DSpark ドラフトモデルはそのまま使う） |
| `--dspark`, `--no-dspark` | DeepSeek の DSpark ドラフトモデル（6.9 GB、`../Abyss-data/models/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF` へ）をダウンロードして設定でオンにする / 設定でオフにする（ファイルは残す）。GPU 1 枚のときだけ |
| `--tools-only` | Abyss-tools だけを作る / 更新する |

以下の節は、同じことを手で行う手順。

## ビルド

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss

# このフォルダ専用の Python 環境
python3 -m venv .venv
.venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests cmake ninja pillow psutil

# エンジンが固定している llama.cpp のソース（setup.py の LLAMA_CPP_COMMIT）: ggml、MMQ カーネル、gguf-py
C=3cf03257f219afbe7334045ff7c6a06ac68c627d
curl -L -o /tmp/llama.zip https://github.com/ggml-org/llama.cpp/archive/$C.zip
unzip -q /tmp/llama.zip -d third_party && mv third_party/llama.cpp-$C third_party/llama.cpp

# エンジン: build/strata
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16
```

ROCm の `hipcc` が入っていれば、`build/libabyss_igpu.so`（下の「iGPU レーン」）も一緒にビルドされる
（対象は既定で `-DSTRATA_IGPU_ARCH=gfx1151`。`-DSTRATA_IGPU_LANE=OFF` で作らない）。

初回のビルドは数分。変更後は変わった部分だけを再コンパイルする。RTX 50 では `120` ではなく `120a` を使う
（MXFP4 のカーネルが Blackwell 専用の FP4 MMA 命令を使い、`120` ではコンパイルできない）。

## モデルを取る

unsloth の [DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) の `UD-Q2_K_XL`
（3 ファイル、96.8 GB）。Hugging Face の CLI（`pip install huggingface_hub` で `hf` が入る）で取る:

```sh
M=~/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF   # Abyss の隣の Abyss-data（NVMe ならどこでもよい）
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf --local-dir $M
sha256sum $M/UD-Q2_K_XL/*.gguf
```

| ファイル | SHA-256 |
| --- | --- |
| `-00001-of-00003.gguf` (5 MB) | `de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b` |
| `-00002-of-00003.gguf` (49.4 GB) | `c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327` |
| `-00003-of-00003.gguf` (47.4 GB) | `a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4` |

ダウンロードが途中で止まって再開しないときは、
`curl -L -C - -o <ファイル> https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-Q2_K_XL/<ファイル>`
で続きから取れる。

同じリポジトリのほかの量子化も、エキスパートの型が decode 用カーネルの対応範囲（IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_K〜Q6_K, MXFP4。IQ1_M はまとめ処理のカーネルがないので prompt は 1 トークンずつ）
なら動くはず。確認したのは UD-Q2_K_XL だけ。

ルーターや小さい射影が量子化されずに F16 で入っている DeepSeek V4 の GGUF（antirez の deepseek-v4-gguf、huihui-ai の
abliterated 版）も pack して動く。`--compat-bf16` がそれらをエンジンの GEMV が読む BF16 に丸める。huihui-ai の Q2 は
同じファイルの llama.cpp と照合した（3,000 トークンで KL 0.0075）。

## pack を作る

pack はエンジンが使う GGUF の索引（小さいテンソルの dense.bin、エキスパートの表、チャットテンプレート付きの
トークナイザー）。シャードと同じフォルダに作る:

```sh
cd ~/llm/apps/Abyss
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL/pack-dsv4 --compat-bf16
```

`--compat-bf16` は、GGUF に F32 で入っているハイパーコネクションの行列を、エンジンが読む BF16 に丸める。数秒で終わる。
エキスパートは GGUF のまま使う。

### 軽い dense pack（attention の射影を Q6_K か Q5_K に）

decode は 1 トークンごとに dense の重み全部（UD-Q2_K_XL で 6.6 GB）を読み、そのうち 4.6 GB が各層の Q8_0 の 3 つ
`attn_q_b`、`attn_output_a`、`attn_output_b`。`tools/requant_dense.py` はこの 3 つだけを再量子化し（llama.cpp のビルドの
`ggml_quantize_chunk` を使う。llama-quantize は i-quant のモデルを imatrix なしでは受け付けない）、シャードの他のバイトは
そのまま写す。できた GGUF を `iq_pack.py` で元と同じように pack にする:

```sh
cd ~/llm/apps/Abyss
GGUF_PY=~/llm/apps/llama.cpp/gguf-py LIBGGML=~/llm/apps/llama.cpp/build-cuda/bin/libggml-base.so \
  .venv/bin/python -I tools/requant_dense.py $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  $M/UD-Q2_K_XL-attn-q6k q6_k 'blk\.\d+\.attn_q_b\.weight' 'blk\.\d+\.attn_output_[ab]\.weight'
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL-attn-q6k/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL-attn-q6k/pack-dsv4 --compat-bf16
```

90 GB のシャードを書き直すので（NVMe で 10〜25 分）、サーバーを止めてから実行する。できたら設定の pack と tokenizer の
パスを新しいフォルダに向ける。3 つの型の品質と得るもの（2026-10-15、v1.3.18。KL はエンジン自身の Q8_0 の基準 logits
との比較、4,000 トークンの最後の 1,000 トークン、CPU レーン、3 つとも同じ条件）:

| attention の射影 | 3 テンソルの大きさ | KL（基準比） | max KL | top token 一致 | nll | VRAM の expert 枠（RTX 5090） |
|---|---|---|---|---|---|---|
| Q8_0（配布された GGUF のまま） | 4.28 GiB | 0.0060 | 0.11 | 97.0% | 1.3071 | 3,010 |
| Q6_K | 3.31 GiB | 0.0082 | 0.31 | 95.8% | 1.3116 | 3,139 |
| Q5_K | 2.77 GiB | 0.0119 | 0.57 | 95.9% | 1.3135 | 3,225 |
| Q4_K | 2.27 GiB | 0.0246 | 0.84 | 92.9% | 1.3224 | 3,268 |

- Q6_K: GEMV が小さくなる分（v1.3.18 の分割カーネル）で decode +3.5%、VRAM の枠が 129 増える。swe-bench の L7（16 問 × 3、
  RTX 5090 + Radeon 8060S）では 91.3 → 93.9 tok/s で、resolved 率（81.2%）と品質スコア（81.0 → 81.2）は同じ。KL の増分は
  実行ごとのばらつきと同じ大きさ
- Q5_K: KL は Q8_0 の 2 倍（1 トークンの最大のずれ 0.57）。RTX 5090 では Q6_K より decode が 4% 速い（1,024 トークンの
  答え: 単発 12.13 → 11.65 ms、検証窓 15.52 → 14.86、86〜89 → 93 tok/s。窓も単発も同じだけ縮む）。帯域の低いカードほど
  効く: Radeon AI PRO R9700（640 GB/s）では 1 トークンの読みが 1.5 GB 減る。同じ swe-bench で 99.0 tok/s、resolved 87.5%
  （14/16）、成功率 83.3%、品質 82.0 で、KL が Q6_K の 2 倍でも劣化は見えない。**作者が常用しているのはこの pack で、
  「メモリと速度」の生成の数値もこれで測っている**（v1.3.20）
- Q4_K: 割に合わなくなる所。KL がさらに 2 倍、最頻トークンの一致が 3 ポイント落ちる（Q8_0 → Q5_K は 1.1）のに、
  単発は Q5_K より 2% しか速くならない（11.67 → 11.40 ms、枠 +43）。swe-bench は回していない
- 触る価値がないもの: `attn_q_a`、kv / compressor / indexer の射影、norm は小さく、共有エキスパート・埋め込み・出力
  ヘッドはもう Q5_K / Q6_K / Q4_K

## 設定と起動

設定ファイル（名前は自由。ここでは `strata-dsv4.json`）を Abyss のフォルダに置く:

```json
{
 "exe": "/home/you/llm/apps/Abyss/build/strata",
 "args": ["--glm-pack", "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4",
          "--max-context", "65536", "--prefill", "auto"],
 "cwd": "/home/you/llm/apps/Abyss",
 "tokenizer": "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4/tokenizer",
 "model_name": "deepseek-v4-flash",
 "sampling": {"temperature": 1.0, "top_p": 0.95},
 "port": 8082,
 "log": "/home/you/llm/apps/Abyss/strata-dsv4.log"
}
```

| 項目 | |
| --- | --- |
| `exe` | エンジン（`build/strata`） |
| `args` | エンジンの引数。`--glm-pack` に pack（DeepSeek V4 も GLM と同じ入口で動く）、`--max-context` にコンテキスト長、`--prefill auto` は prompt のまとめ処理の大きさ |
| `tokenizer` | pack の `tokenizer` フォルダ（チャットテンプレートは DeepSeek のもの） |
| `model_name` | API が返すモデル名 |
| `sampling` | リクエストが指定しないときの既定値（DeepSeek の推奨: temperature 1.0, top_p 0.95） |
| `reasoning_effort` | 省略可。思考の既定レベル（`none`, `low`, `high`, `max`） |
| `thinking_budget` | 省略可。思考の上限トークン数（既定 32768、0 で上限なし） |
| `env` | 省略可。エンジンの設定 `{"STRATA_GLM_...": "..."}`（[設定項目](#設定項目)） |
| `log` | エンジンのログ |

サーバーを起動する:

```sh
.venv/bin/python serve/server.py --engine strata --config strata-dsv4.json --port 8082
```

読み込みが終わると `ready: http://127.0.0.1:8082/v1` と表示される。起動には 1 分ほどかかる（密な重みを GPU に載せ、
全エキスパートを VRAM と RAM に読み込む。NVMe から 4 GB/s で約 20 秒）。Ctrl+C で止まる。

サーバーのオプション: `--host 0.0.0.0` でほかのマシンからの接続を受け付ける（必ず `--api-key <secret>` も付ける）、
`--open` でブラウザを開く、`--gpu 0` で GPU を選ぶ。

## 使い方

- **ブラウザ:** `http://127.0.0.1:8082/` でチャット、エキスパートの配置と GPU のモニター、設定。
- **OpenAI 互換 API:** ベース URL `http://127.0.0.1:8082/v1`、モデル名は何でもよい。

  ```sh
  curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
  ```

  回答は `content`、思考は `reasoning_content` に入る。
- **Anthropic 互換 API:** `http://127.0.0.1:8082/v1/messages`（Claude Code なら `ANTHROPIC_BASE_URL=http://127.0.0.1:8082`）。
- **思考:** DeepSeek V4 は指定しなければ考えてから答える。`reasoning_effort`（またはチャットの設定）で変えられる:

  | レベル | DeepSeek に渡るもの |
  | --- | --- |
  | 指定なし | 思考あり |
  | `none` | 思考なし（prompt の末尾が `</think>`） |
  | `low` | 思考あり |
  | `high` | 思考あり + DeepSeek の「Reasoning Effort: Absolute maximum」の指示 |
  | `max` | 思考あり + 「Beyond maximum」の指示 |

  サーバーは回答に最低 1024 トークン（または `max_tokens` の 4 分の 1）を必ず残す。`max_tokens` が小さい（数百）と、
  思考はほぼすぐに打ち切られる。思考させるときは `max_tokens` を数千にする。
- **ツール呼び出し:** OpenAI の `tools`、Anthropic の `tools` が使える。1 回の応答で複数の呼び出しも可。モデルは
  DeepSeek の DSML 形式で書き、サーバーがそれを `tool_calls` に変換する。過去の呼び出しと `tool` の結果は
  prompt に描画し直す。
- **会話の再利用:** 前のリクエストの続き（同じメッセージ + 新しいメッセージ）は、新しいトークンだけを読む。
- **リクエストは 1 つずつ**処理する。ほかは順番待ち。

### Strata-tools の strata.sh を使う場合

`strata.sh`（start / stop / log）を使うなら、Abyss のフォルダに起動スクリプトと設定を `run-<名前>.sh`,
`strata-<名前>.json` の名前で置いて、`STRATA_DIR=~/llm/apps/Abyss ./strata.sh start <名前>` で起動する。
起動スクリプトの中身は上の起動コマンド（`exec .venv/bin/python serve/server.py --engine strata "--config" "<設定>" "--port" "8082" "$@"`）。

## 複数の GPU

NVIDIA の GPU が 2 枚以上あれば、層を分けて載せられる（layer split）。各 GPU は自分の層のエキスパートだけを VRAM に置く。
2 枚目の VRAM の分だけ、VRAM に載るエキスパートが増える。

- **セットアップ:** `./setup-deepseek.sh --gpus 0,1` または `--gpus all`。設定に `"gpu": [0, 1]` が入り、サーバーが
  `--layer-split auto` を付けて起動する。世代の違う GPU（例: RTX 5090 + RTX 3090）は、両方のアーキテクチャ
  （`86;120a`）でビルドする。
- **`auto` の選び方（v1.3.25）:** すべての境目と「各カード 1 枚だけ」を、予測した decode 時間で比べる（各カードの読む
  バイト ÷ メモリ帯域、CPU / iGPU レーンが隠しきれないミスの分、境目 1 つごとの固定コスト。RTX 5090 + RTX 3090 で
  合わせた係数で、実測 10.9〜18.1 ms との差は 0.35 ms 以内）。1 枚のほうが速ければ 1 枚で動かす。decode の 1 トークンは
  part を順番に通るので、遅い 2 枚目の層は、増えた VRAM で減るミスより高くつくことがある。RTX 5090 + RTX 3090（どちらも
  PCIe Gen4 x4）では 5090 だけで 94〜95 tok/s（以前選んでいた 26 / 17 の分割は 84 tok/s、最良の分割 33 / 10 でも
  88 tok/s）。`STRATA_GLM_SPLIT_KEEP=1` で分割を続ける（長い文脈や prompt 用の VRAM が欲しいとき）。
  `STRATA_GLM_SPLIT_LOG=1` で予測をすべて出す。
- **境目を手で決める:** 設定の `"layer_split": "22"`（2 枚目が担当する最初の層）。
- **iGPU レーン:** 分けたときも使える。各 GPU の RAM 層を 1 つの iGPU が順番に計算する。
- **確認済みの範囲:** 1 枚の GPU の上で 2 つに分けた形（`STRATA_GLM_SPLIT_SAME=1 STRATA_GLM_SPLIT=22
  STRATA_GLM_DEVS=0,0`、試験用）で、prompt と decode の結果が分けない場合と一致すること（KL がぶれの範囲内）。
  RTX 5090（PCIe Gen4 x4）+ RTX 3090（USB4 ケース、約 3.8 GB/s）を 27 / 16 層に分けた実測は、iGPU レーンありで
  prompt 8K / 32K が 1,506 / 1,985 tok/s、decode が 54.9 / 56.7 tok/s（RTX 5090 1 枚 + iGPU: 1,683 / 2,876 と
  58.5 / 65.9）。2 枚目で VRAM の当たり率は 82 → 91% に上がるが、カード間の受け渡しと遅いリンクのコストのほうが
  大きかった。PCIe に直接挿した速い 2 枚目ではまだ測っていない。
- **投機デコード**（v1.3.26）: 検証窓を part ごとに順に通し、各 part の列を 1 トークンのときと同じように次の part に
  渡す。RTX 5090 + RTX 3090 を 33 / 10 に分けて 87〜90 tok/s（なしで約 80〜88）、候補の採用率は 1 枚のときと同じ
  （コードで 57%）。
- **prestage**（v1.3.27）: 各 part が、次の層で使いそうな RAM 層のエキスパートを mixer の間に VRAM へ送る（1 枚の
  ときと同じ）。分割では chunk の大きさを先に決め（最大 8192）、残った分を prestage に使う。RTX 5090 + RTX 3090 の
  33 / 10 分割で、32K の prompt が 2,177〜2,186 → 2,274〜2,350 tok/s、8K はほぼ同じ。`STRATA_GLM_PRESTAGE=0` でオフ。
- AMD の GPU は下の「AMD の GPU」を参照。
- **V100（sm_70）:** CUDA 13 は V100 向けにコンパイルできないので、CUDA 12（12.9）を入れて
  `./setup-deepseek.sh --nvcc /usr/local/cuda-12.9/bin/nvcc` でビルドする（CUDA 12.9 は RTX 50 の `120a` も作れる）。
  sm_70 でのビルドは通ることを確かめた。V100 の実機ではまだ動かしていない。tensor core を使う新しいカーネル
  （indexer、prompt の attention）は sm_80 以上が条件で、V100 では従来のカーネルを使う。

## AMD の GPU（HIP、試験的）

AMD の Radeon でも動く。対象は RX 7900 XT / XTX（gfx1100）、RX 9070 / AI PRO R9700（gfx1201）、Ryzen AI Max の
Radeon 8060S（gfx1151）。

- **セットアップ:** `./setup-deepseek.sh --backend hip`（NVIDIA の GPU がなければ自動で HIP を選ぶ）。Linux では
  `/opt/rocm`（または `/opt/rocm/core-10.0`）の ROCm 10 を使う。HIP 7.14 以降が要り、ROCm 7.0〜7.2 では止まる。この
  2 つはいまのドライバと組み合わせると失敗するため（issue #1、#8。`ABYSS_OLD_ROCM=1` でそれでもビルドできる）。
  エンジンは `build-hip/strata` にビルドされ、設定に `"backend": "hip"` が入る。別の ROCm に切り替えたら、
  `ROCM_PATH` を付けてもう一度実行する。その ROCm でエンジンを最初からビルドし直す（v1.3.44）。
- **ドライバと ROCm:** 入れ方は [docs/amd/README.ja.md](amd/README.ja.md) にまとめた（amdgpu、ROCm 10 の
  アーキテクチャ別のパッケージ、`HSA_OVERRIDE_GFX_VERSION` を設定してはいけない理由）。動作確認とトラブル対応も
  そこにある。R9700 では `amdrocm10.0-gfx1201` が要る。
- **確認済みの範囲:** Radeon 8060S（gfx1151）単体と、AI PRO R9700（gfx1201）で 8060S を iGPU レーンにした構成。
  R9700 では DeepSeek V4 が swe-bench の問題で 36〜42 tok/s（v1.3.39）、51.7 tok/s（v1.3.41。レーンなしで 49.6）、
  1 トークンずつで 52 tok/s（v1.3.40。レーンなしは v1.3.41 で 46）で、CUDA 版との KL はぶれの範囲内（0.005〜0.006）
  だった。DSpark のドラフトモデル（v1.3.43）を使うと、同じ swe-bench の問題がレーンなしで 61.9 tok/s、レーンありで
  67.3 tok/s（[DSpark のドラフトモデル](#dspark-のドラフトモデルdeepseek-v4-flash)）。
- **ROCm の HW キュー:** `GPU_MAX_HW_QUEUES` は 2 のままにする（セットアップが書く）。R9700 向けの手順にある 1 では、
  prompt が 3 分の 1 ほど遅くなる（[docs/amd](amd/README.ja.md#3-環境変数)）。
- **Ryzen AI Max に単体の Radeon をつないだ場合:** NVIDIA のカードのときと同じく、iGPU が RAM 層のエキスパートを
  計算する（R9700 で生成 +12〜46%）。設定の `"gpu_pci"` があれば、HIP の番号が変わっても同じカードを使う。
- **速くなる部分:** indexer のスコア計算と prompt の attention は RDNA の WMMA を使う（gfx11 / gfx12）。
  `STRATA_DS4_IDX=1` で indexer を、`STRATA_DS4_ATTN=1` で attention を従来のカーネルに戻せる。
- **制限:** NVIDIA との組み合わせ（1 つのプロセスで CUDA と HIP の両方）には対応していない。

## DeepSeek V4.1 Flash

v1.3.29 から、同じエンジンで DeepSeek-V4.1-Flash-1017 が動く。準備は手作業（まとめてセットアップが入れるのは V4 だけ）。
読める GGUF、pack の作り方、設定例、測定結果は [docs/deepseek41/README.md](deepseek41/README.md)、V4 から変わった計算は
[docs/deepseek41/SPEC.md](deepseek41/SPEC.md)（どちらも英語）。要点:

- **ディスクとメモリ。** nazeshinjite の Q2 GGUF は 366 GB。エキスパートが 153 GB（層ごとに 384 個、1 個 9.95 MB）、Engram
  の表が約 200 GB、残りが密な重み。RTX 5090 + RAM 94 GB では、エキスパートの約 57% が常駐する（VRAM に約 2,300 個、60 GB
  の RAM 層に約 6,500 個）。残りと Engram の行（1 トークン 48 行、1 行 264 バイト）は GGUF から読む。いちばん速い NVMe に
  置き、RAM は約 25 GB をページキャッシュに残す（94 GB の機械なら `STRATA_GLM_RAM_GB=60`）。
- **速さは SSD で決まる。** SSD からエキスパートを読まないトークンは 13〜15 ms。読むたびに 3.4 GB/s で約 3〜4 ms 増える。
  `STRATA_GLM_TAIL_SKIP=3 STRATA_GLM_TAIL_SKIP_DISK=1`（ルートの 4〜6 位が SSD にしかないときは計算しない。量子化前の
  モデルに対する KL は 0.229 → 0.240）で生成は約 28 tok/s、なしで 17.7 tok/s。swe-bench のコーディング問題は通しで
  51.6 tok/s。prompt は遅い（アテンションを 1 トークンずつ計算し、prompt のエキスパートも SSD から読む）。
- **軽い dense pack**（v1.3.36、任意）: `attn_q_b` / `attn_output_a` / `attn_output_b` を Q8_0 から Q5_K にすると、
  dense が 8.14 → 6.50 GB、VRAM のエキスパート枠が 177 増え、生成が 10〜15% 速くなる。KL は 0.242 → 0.251。作り方は
  [docs/deepseek41/README.md](deepseek41/README.md#a-lighter-dense-pack-q5_k)。
- **メモリ上限つきのスコープで起動する**（`systemd-run --user --scope -p MemoryMax=88G ...`）。ページキャッシュと pinned の
  層が重なると、systemd-oomd がデスクトップのセッションを止めることがある。
- **まだできないこと:** 投機デコード、層の分割、会話のディスク保存、画像。

## Ornith-1.5-122B

v1.3.46 から、同じエンジンが [Ornith-1.5-122B](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF)（Qwen3.5-MoE 系、
llama.cpp の `qwen35moe`）も動かします。gated delta-net 層 36 + full attention 層 12、1 層 256 エキスパート（8 個を
使用 + ゲート付きの共有エキスパート）、残差は普通の 1 本です。V4.1 と同じく手動でセットアップします。GGUF、pack の
コマンド、設定例、測定は [docs/ornith/README.md](ornith/README.md)（英語）に。要点:

- **メモリ。** unsloth の UD-Q4_K_XL は 79.7 GB: エキスパート 70.4 GB（256 × 48 層、3072 × 1024、IQ4_XS〜Q6_K）と、
  GPU に置く Q8_0 の dense 5.4 GB。32 GB のカードにはエキスパート約 25 GB が載り、残りは RAM 層 60 GB に入るので、
  94 GB の PC ではデコード中に SSD を読みません。
- **速度。** AI PRO R9700 + 8060S レーン: デコード 1 トークン 23〜25 ms（41〜43 tok/s。iGPU の代わりに CPU が
  レーンを計算すると 30 ms）、575 トークンのプロンプトはバッチ経路で 233 tok/s。llama.cpp との KL は 559 位置で 0.024（このモデルでは llama.cpp 自身の CPU と HIP の差が
  0.027〜0.029）。
- **投機デコード。** n-gram ドラフトが検証窓を通って動きます（既定は 1 窓 1 ドラフト、`STRATA_DS4_SPEC=<k>` で増やす）:
  コピー課題で 1 ドラフト 57 tok/s、3 ドラフト 76 tok/s。GGUF の NextN ブロックもドラフターになります（エキスパートを
  `tools/nextn_pack.py` で詰め直して `STRATA_GLM_MTP_GGUF` で読む）: コード 59 tok/s、散文 50、コピー課題 85
  （iGPU なしでは コード 36 → 42、散文 33 → 39）。
- **まだないもの:** 会話スロット、レイヤー分割、画像。CUDA ビルドはコンパイルは通るが未実行。エキスパートカーネルは
  この形状にまだ調整していない。

## EXL3 量子化（GLM-5.3-Flash）

v1.3.35 から、エンジンは [exllamav3](https://github.com/turboderp-org/exllamav3) の EXL3 量子化の GLM-5.3-Flash も、
safetensors からそのまま読む。詳しくは [docs/exl3/README.md](exl3/README.md)（英語）。要点:

- **インストール。** `./abyss.sh --setup` のモデル選択に、Maya の GGUF と並んで turboderp の `EXL3-sc_2.10bpw`（88.3 GB）と
  `EXL3-3.05bpw`（125.3 GB）が出る。それぞれにこの PC での向き不向きを表示する: RTX 30 以降のカードに 8 割以上載るなら
  おすすめ、APU の iGPU か AVX-512 の CPU が RAM 層を計算できるなら品質優先の選択肢、AVX2 だけの CPU には向かない。
  `--model EXL3-sc_2.10bpw` で質問を飛ばせる。セットアップでは NVIDIA の GPU のみ。
- **切り替え。** 量子化ごとに設定ファイルと起動スクリプトができる（`maya-exl3-sc_2.10bpw.json`、
  `run-maya-exl3-sc_2.10bpw.sh`）。GGUF と EXL3 を両方入れておき、起動するスクリプトで選ぶ。エンジンは
  `pack/exl3.txt` があれば EXL3 として読むので、設定項目は要らない。
- **手作業で。** `.venv/bin/python tools/exl3_pack.py --model <dir>`（約 5 秒）で `<dir>/maya-exl3.gguf` と `<dir>/pack/`
  ができる。起動は `--glm-pack <dir>/pack`。
- **目安**（RTX 5090 + Radeon 8060S の lane + RAM 94 GB、採点 1,000 トークン）: sc_2.10bpw は nll 1.591〜1.598、1 位一致
  63.3〜63.7%（Maya-S-v2-IQ2_XXS は 1.634 / 62.3%）。1 トークン 18.8 ms（同 15.4 ms）。v1.3.36 で iGPU のトレリス
  復号を `v_sad_u8` と `v_dot2_f32_f16` にした（v1.3.35 は 22.4 ms）。

## メモリと速度

エンジンは、密な重み（UD-Q2_K_XL で 6.2 GB）、キャッシュ、prompt 用のバッファを置いた残りの VRAM をエキスパートで
埋め、空き RAM から pinned の RAM 層を確保し、足りない分は必要なときに SSD から読む。使ったエキスパートを
`<pack>/expert_usage.txt` に記録し、次の起動ではそれを先に読み込む。

RTX 5090 + Ryzen AI Max+ 395（RAM 94 GB）での実測（UD-Q2_K_XL、コンテキスト 65536、全エキスパートが
VRAM 23 GB + RAM 70 GB に載った状態、iGPU レーンあり）:

| | |
| --- | --- |
| 生成（1,024 トークンの応答、文章 / コード、greedy） | 約 92〜95 tok/s（iGPU の代わりに CPU レーンなら約 55〜60） |
| 生成（swe-bench のコーディング問題 16 問 × 3、サンプリングあり、通し） | 約 100 tok/s（通しで 1 トークン平均 10.1 ms。1,024 トークンの応答では単発 1 トークン 11.1 ms、2 トークンの検証窓 14.6） |
| 生成（swe-bench のコーディング問題 60 問 × 5、サンプリングあり、通し） | 約 107 tok/s（1 問 100.5〜114.5。v1.3.27） |
| 生成（コンテキスト 125K トークン） | 約 65 tok/s |
| prompt（8K / 32K / 125K トークン、サーバー、2 回目） | 約 1,630〜1,760 / 約 3,100〜3,270 / 約 2,800 tok/s |
| 起動（重み + エキスパートの読み込み） | 約 1 分 |
| 65536 トークン分のアテンションキャッシュ | 0.44 GB |

（生成の行: Abyss v1.3.21、Q5_K の attention pack（上の「軽い dense pack」。配布 GGUF のままなら約 82〜84 と約 91〜94）。
prompt と 125K の行: v1.3.16〜1.3.18、配布 GGUF のまま。同じマシン・同じファイルで llama.cpp `bed0a8566` は `-ngl 99 -ncmoe 32` で生成 28〜30 tok/s、
prompt は ubatch 512 で 78 tok/s、ubatch 8192 で 660 tok/s。）各リリースの数値と測り方は
[CHANGELOG.md](../CHANGELOG.md) にある。

起動直後や、メモリがモデルより少ない環境では、エキスパートを SSD から読むので遅くなる。RAM 層のエキスパートは、
一部を CPU が計算し、残りを GPU が PCIe 経由で取り寄せる。その配分は起動時に測って決める。

**iGPU レーン。** AMD の APU（Ryzen AI Max+ 395 の Radeon 8060S）では、内蔵 GPU が CPU と同じメモリを約 240 GB/s
で読める（CPU のスレッドは約 75 GB/s）。ROCm が入っていて `libabyss_igpu.so` がビルドされていれば、エンジンは
起動時に RAM 層を内蔵 GPU に登録し、RAM 層のエキスパートを CPU の代わりに内蔵 GPU で計算する（コピーはしない）。
どちらで計算するかは起動時に両方を測って決め、ログに出す（`iGPU lane: an expert 0.046 ms vs 0.107 on the CPU's
16 threads`）。CPU のほうが速い場合や、ライブラリか HIP のデバイスがない場合は CPU で計算する。
prompt のまとめ処理でもこのレーンを使う。prompt のチャンクが必要とする RAM 層のエキスパートを、CPU のスレッドではなく
内蔵 GPU で計算する（8K の prompt で +40%）。レーンの呼び出しには期限がある（`STRATA_GLM_IGPU_TIMEOUT_MS`、20 ms）。
カーネルドライバが内蔵 GPU のキューを止めているとき（起動後の数分間、読み込みのあとにカーネルがメモリを回収する間に
起きる）は、遅れた呼び出しが終わるまで CPU のスレッドが引き継ぎ、起動時の計測はきれいに測れるまでやり直す。

**投機デコード（`STRATA_DS4_SPEC`、既定でオン）。** この GGUF にはドラフトモデルがないが、応答は prompt や応答の
前のほうに出てきた並び（コード、ファイルパス、ツール呼び出し）を繰り返すことが多い。エンジンは直近 3 トークンが
前に出てきた場所を探し、その次のトークンを候補として次の 1 トークンと一緒に 1 回で確かめ、モデルの答えと一致すれば
採用する。出力は使わない場合と同じで（同じロジットに同じサンプラー）、速さだけが変わる。リクエストごとに採用率を
学習するゲートがあり、割に合わない間は候補を出さない。コーディングの prompt で +10〜13%（候補の 77〜85% が採用）、
文章ではほぼ変わらない（採用は約 50%）。`STRATA_DS4_SPEC=0` でオフ。GPU を分割しているときも使う（v1.3.26）。

## DSpark のドラフトモデル（DeepSeek V4 Flash）

v1.3.43 から、DeepSeek V4 Flash 0731 に DeepSeek の DSpark ブロックドラフトモデルを並べて動かせる。DeepSeek V4 の 3 層が
本体の 40〜42 層の出力を読み、次の 5 トークンをそれぞれの確信度つきでまとめて提案する。確信度 0.6 以上が続く先頭の下書きを、
そのトークンと一緒に 1 回の検証の窓に通し、本体のサンプリングと一致したものだけを残す。回答は本体が出すものとして変わらず、
変わるのは 1 トークンあたりの時間だけ。ファイルは公式のドラフトモデルを bleysg が GGUF にした
[`DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf`](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF)
（6.94 GB、MIT。expert は Q2_K、Markov ヘッドは Q8_0）。

**オンにする。** `./setup-deepseek.sh --dspark` がダウンロードして SHA-256 を照合し、設定に書き足す:

```json
"env": {"STRATA_DS4_DSPARK": "/home/you/llm/apps/Abyss-data/models/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF/DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf"}
```

そのあとサーバーを再起動する（`../Abyss-tools/abyss.sh restart`）。`./setup-deepseek.sh --no-dspark`（またはこの行を消す）で
オフに戻る。起動時のエンジンのログの `dspark:` の行に、読んだファイルと使った VRAM が出る（`3 stages over target layers
40-42, 5 drafts a step; 6.48 GB on CUDA0 ...`）。使えるのは DeepSeek V4 Flash 0731 を GPU 1 枚で動かすときだけで、層の分割、
DeepSeek V4.1、MTP ブロックとは併用できない。そのときは `dspark: not loaded - <理由>` と出て、本体はドラフトモデルなしで動く。

**かかるもの。** ドラフトモデルは重みと expert をすべて VRAM に置く。6.48 GB で、その分 expert のプールが小さくなる。RAM 層は
ドラフトモデルなしのときの大きさのままにするので、いちばん使われない約 6.5 GB の expert は SSD から読むことになる（下の
マシンの生成中で 1 トークンあたり 0.01 回）。RAM に余裕のある PC では、これまでどおり `STRATA_GLM_RAM_GB` で RAM 層の大きさを
手で決められる。この上限がないと、RAM 層がその expert まで pinned で持とうとする。RAM 94 GB の PC では 73.9 GB を pinned に
して空きが 13 GB になり、8K の prompt に 11 秒ではなく 301 秒かかった。下書き 1 回は AI PRO R9700 で約 4.4 ms。

**速さ**（AI PRO R9700、OCuLink PCIe Gen4 x4。RAM 94 GB の Ryzen AI Max+ 395 の PC。UD-Q2_K_XL と Q5_K の attention pack、
サーバー経由）:

| | ドラフトモデルなし | DSpark |
| --- | :---: | :---: |
| swe-bench のコーディング 6 問（サンプリングあり、通し）、内蔵 GPU なし | 48.4 tok/s（v1.3.42） | 61.9 tok/s |
| 同じく内蔵 GPU レーンあり | 51.7 tok/s（v1.3.41） | 67.3 tok/s |
| 1,024 トークンの回答、コード / 散文（貪欲、内蔵 GPU なし） | 46.6〜48.6 / 47.2〜47.5 tok/s | 63.4〜66.4 / 50.5〜51.4 tok/s |
| prompt 8K / 16K | 850 / 978〜1,028 tok/s | 799 / 977〜1,000 tok/s |

swe-bench の問題ごとに +16〜32%。これらの問題では下書きの 68〜82% が受理され、1 回の確認で 2.7〜3.7 トークン進む（コードを
写す問題では 4.7）。散文の伸びがいちばん小さい（1 回 2 トークンほど）。検証の窓は 1 列あたり約 9 ms かかり、これは CPU
レーンが 1 トークン分の RAM 層 expert を計算する時間（8.4 ms。RAM の約 93 GB/s で約 93 個）とほぼ同じ。速さを決めているのは
ドラフトモデルではなく窓なので、速い GPU より速い CPU / RAM や内蔵 GPU レーンのほうが効く。NVIDIA はビルドが通るだけで、
まだ動かしていない。

**1 回ごとの選び方。** 直前の 3 トークンが prompt か回答の前のほうに出ていれば、その続き（最大 5 トークン）を窓の下書きに
して、ドラフトモデルは動かさない。ただし、その繰り返しが当たり続けている間だけ: 毎回、繰り返しの提案を実際に出たトークンと
照らし、位置ごとの当たり率を追う。先頭の位置が 0.75 を下回るとドラフトモデルに切り替え、そうでなければ位置ごとの当たり率の
積が 0.5 以上の長さまで繰り返しを使う。確信度の高い下書きがなければ、そのトークンだけを処理する。回答ごとにエンジンの
ログに出る `ds4 dspark:` の行に、窓の数、下書きと受理の数、1 回で進んだトークン数が出る。

**設定項目**（設定ファイルの `env`）:

| 設定 | 既定 | |
| --- | --- | --- |
| `STRATA_DS4_DSPARK` | オフ | `<ドラフトモデルの GGUF>`: DeepSeek の DSpark ドラフトモデルを読み、それで生成する |
| `STRATA_DS4_DSPARK_THR` | 0.6 | 下書きに必要な確信度。下回る最初の下書きで窓を打ち切る（swe-bench: 0.7 で 61.3、0.8 で 59.8 tok/s、0.6 は 61.9。ds4 は 0.7） |
| `STRATA_DS4_DSPARK_K` | 5 | 1 回の窓の下書きの上限（1〜5） |
| `STRATA_DS4_DSPARK_CHAIN` | オフ | `<p>`: 1 つずつのしきい値の代わりに、確信度の積が p 以上の間の下書きを使う |
| `STRATA_DS4_DSPARK_NGRAM` | オン | 文中の繰り返しを先に使う（上の説明）。`0`: 毎回ドラフトモデル |
| `STRATA_DS4_DSPARK_NGRAM_P` / `_NGRAM_CHAIN` | 0.75 / 0.5 | 繰り返しを使う先頭位置の当たり率 / その長さを決める位置ごとの当たり率の積 |
| `STRATA_DS4_DSPARK_EXPERTS` | 256（全部） | ドラフトモデルの 1 段あたりの VRAM の expert 枠（8〜256。256 で 1 段 2.1 GB）。枠のない expert はその行から外し、下書きのあとによく使うものを入れ替える。128 / 64 で 3〜5 GB 返せるが、本体の VRAM ヒットは 2 ポイントしか上がらず、下書きは 30〜50% 遅くなる（24 GB のカード向けの選択肢。そこでは未計測） |
| `STRATA_DS4_DSPARK_UPLOADS` | 4 | 下書きのあとに 1 段あたり枠へコピーする expert の数（0〜8） |
| `STRATA_DS4_DSPARK_PROF` | オフ | `1`: 終了時の報告に、ドラフトモデルの GPU 時間を段階ごとに出す |
| `STRATA_DS4_DSPARK_LOG` | オフ | `<n>`: n 回の窓ごとに、平均の時間と幅を出す（stderr） |

## 設定項目

エンジンへの環境変数。設定ファイルの `"env"` に書く（Maya の全項目は [docs/maya/README.md](maya/README.md#tuning)。
ここで特に関係するもの）:

| 設定 | 既定値 | |
| --- | --- | --- |
| `STRATA_GLM_WARM` | オン | `0`: 起動時に読み込まない（使われたエキスパートから載っていく。最初の応答が遅い） |
| `STRATA_GLM_RAM_GB` | 空き RAM から | pinned の RAM 層の大きさを GB で固定 |
| `STRATA_GLM_RAM_HEADROOM_GB` | 6 | RAM 層を決めるとき、システムのために空けておく RAM |
| `STRATA_GLM_STREAM_FETCH` | オン | SSD から読んだエキスパートを、読めた塊から順に VRAM へコピーする。PCIe のコピーが SSD の読みのすぐ後ろを走る（v1.3.31。DeepSeek V4.1 で 1 トークン −9%）。`0` で読み終わってからまとめてコピー |
| `STRATA_GLM_RAM_THP` | オン（Linux） | RAM 層を透過的ヒュージページに置く（空きの 2 MB ブロックの範囲で。APU のレーンが再検証するページが 512 分の 1 になり、数秒の停止が消える）。`0` でドライバの pinned ページ |
| `STRATA_GLM_RESERVE_MB` | 700 | エキスパート用プールの横に空けておく VRAM |
| `STRATA_GLM_CPU_LANE` | 物理コアごとに 1 スレッド | RAM 層のエキスパートを計算する CPU スレッド数。`0` で無効 |
| `STRATA_GLM_X3_SPLIT` | オフ | EXL3 の pack で、各ルートの RAM 層エキスパートのうちこの割合を CPU スレッドが、残りを iGPU が同時に計算する（`auto`: 起動時の計測から）。Ryzen AI Max+ 395 では電力と熱を共有するため遅くなる（[EXL3](exl3/README.md#settings)） |
| `STRATA_GLM_EXL3_DECODE` | GPU による | EXL3 の pack で、`new`（tensor core のタイル、RTX 30 以降）か `legacy`（CUDA コア）をどの GPU でも選ぶ |
| `STRATA_GLM_CPU_PIN` | オフ | `<CPU の並び>`（例 `0-15`、`0,2,4-7`）: CPU レーンのワーカーをその CPU に固定し、スレッド数を並びの数までにする。GPU 1 枚でも分割でも、Linux でも Windows でも効く（v1.3.29）。`1` / `0` は従来どおり、分割の各部のプールを固定するかどうか |
| `STRATA_GLM_PCIE_SHARE` | 起動時に測定 | RAM 層のエキスパートのうち、CPU で計算せずに PCIe で GPU に送る割合（`0` = すべて CPU、`1` = すべて GPU） |
| `STRATA_GLM_PREFILL` / `--prefill` | `auto` | prompt のまとめ処理の大きさ: `auto`、トークン数、`0`（1 トークンずつ） |
| `STRATA_GLM_PREFILL_SUB` | auto（NVIDIA）、1024（HIP） | prompt の mixer と dense FFN のサブバッチ（16〜8192）。HIP の既定は v1.3.40 から 1024（以前は 256。AI PRO R9700 で DeepSeek V4 の 8K / 16K の prompt が +8% / +10%、GLM-5.3-Flash が +13% / +15%） |
| `STRATA_GLM_USAGE` | `<pack>/expert_usage.txt` | 起動をまたいで残すエキスパートの使用記録。`0` で無効 |
| `STRATA_GLM_NO_REUSE` | オフ | `1`: 会話を再利用しない |
| `STRATA_GLM_THINK_RESERVE` | 1024 | （サーバー）思考が回答のために必ず残すトークン数 |
| `GPU_MAX_HW_QUEUES` | 2（HIP） | ROCm がデバイスごとに作る HW キューの数。シェルにも設定ファイルにもないとき、サーバーが 2 を渡す（v1.3.40）。AI PRO R9700 の DeepSeek V4 の 8K / 16K の prompt は、1 で 586〜650 / 676〜684 tok/s（R9700 向けの手順でアイドル電力の回避策として広まっている値。この PC ではどちらでもアイドル 12 W）、ROCm の既定の 4 で 653〜737 / 748〜756、2 で 866〜995 / 999〜1,011 |
| `STRATA_GLM_IGPU` | 自動 | iGPU レーン。`libabyss_igpu.so` と HIP のデバイスがあり、CPU より速ければ使う。`0` で使わない、`1` で遅くても使う。`STRATA_GLM_IGPU_LIB=<パス>` で別のライブラリ |
| `STRATA_GLM_IGPU_TIMEOUT_MS` | 20 | これより遅れたレーンの呼び出しは CPU のスレッドが引き継ぐ（0: 待つ） |
| `STRATA_GLM_IGPU_POKE_MS` | 500 | ルーティングのあと、iGPU を空のカーネルで起こしておく時間（50 µs ごと。v1.3.40 で 20 から変更）。DeepSeek V4.1 はルーティングの間に SSD を 20 ms 以上待つことがあり、その間に寝た iGPU は次の呼び出しに 21〜27 ms かかっていた（AI PRO R9700 + 8060S の最も遅い系列: 遅延 123 回、1 token 80.4 ms → 500 で 3 回、65.6 ms） |
| `STRATA_GLM_IGPU_EARLY` | オン | SSD からエキスパートを読むルートで、iGPU レーンを読みの後ではなく前に起動する（v1.3.36。DeepSeek V4.1 で −2%、出力は同じ）。`0`: 読みの後 |
| `STRATA_GLM_IGPU_XQ` | オン | トークンの入力を、GPU が量子化した q8_1 のまま request で受け取り、iGPU で量子化し直さない（v1.3.36、ビット同一）。`0`: iGPU で量子化 |
| `STRATA_GLM_PREFILL_IGPU` | オン | prompt の RAM 層エキスパートを iGPU で計算する。`0`: CPU のスレッド |
| `STRATA_GLM_IGPU_FUSED` | オン | prompt の iGPU エキスパートを fused な int8 tensor core カーネルで計算する（gfx1151、IQ2_XS / IQ3_XXS の 4096 / 2048 エキスパート。v1.3.18）。`0`: 1 行ずつのカーネル。`ABYSS_IGPU_FUSED_CHECK=<n>` で最初の n 回を 1 行カーネルと比べる（stderr） |
| `STRATA_GLM_MV_SPLIT` | auto | dense GEMV の 1 行を 4 warp で分担する: 未設定 = 256 block 未満と、Q6_K は 4,096 block まで（v1.3.18）。`0` 常に 1 warp、`2` / `4` 常にその分担。1 warp のときとビットは一致しない |
| `STRATA_GLM_MV_FUSE` | on（HIP） | 型の違う小さな GEMV の組（DeepSeek V4 の x の射影）を 1 回の起動で、1 行 4 wave で計算する（v1.3.40。AI PRO R9700 では小さな GEMV が大きさによらず 1 回 15〜29 µs かかる）。`0`: 型ごとに起動。型ごとのカーネルとはビットが一致しない |
| `STRATA_GLM_GRID_PAD` | on（HIP） | wave 数が GPU の wave スロット数の倍数ちょうどかその直前になる起動に、空のブロックを足す（v1.3.40）。gfx1201 ではそういう起動の次の起動が 8〜14 µs 遅れ（ROCm/TheRock#9300）、DeepSeek V4 の 4096 行の GEMV と expert の down が 1 token に約 275 回当たっていた（AI PRO R9700: 22.2 → 20.2 ms/token）。ビット一致。`0` で切る |
| `STRATA_GLM_HC_CPU` | on（HIP）、off（NVIDIA） | deepseek4: MoE の層のうち CPU / iGPU レーンが計算した分を、専用の待ちカーネルではなく次の層の attention 側の mHC 読み出し（hc3）の中で足す（1 層あたり起動が 1 回減り、答えは hc3 の 32 ブロックで読む。v1.3.42）。ビット一致。`0` で待ちカーネル、NVIDIA でも `1` で有効（未測定） |
| `STRATA_GLM_WAIT_FETCH` | on（HIP）、off（NVIDIA） | MoE の層のディスク読み込みの待ちと、RAM 層の expert のコピーを 2 回ではなく 1 回の起動で行う（v1.3.42。処理は同じ）。`0` で 2 回の起動 |
| `STRATA_HIP_EXPERTS_LEGACY` | off | `1`: decode の expert を RDNA 版ではなく元のカーネルで計算する（gfx1100、v1.3.40 から gfx1201 も: AI PRO R9700 で gate/up と down が約 2 割速い。gfx1151 は IQ3_XXS の down 以外）。どちらでもビット一致 |
| `STRATA_GLM_PROMOTE_RATIO` / `_MARGIN` | 4 / 16 | RAM 層のエキスパートの使用回数が「VRAM 側の ratio 倍 + margin」を超えたら入れ替える（2 / 4 は以前の、入れ替えの多い規則） |
| `STRATA_GLM_BG_PACE_KB` | 1024 | tier の入れ替えのコピー（prompt の後の補充、入れ替え）を、token の境目でまとめて出さず、MoE 層ごとに最大この KB ずつ、次の層の mixer の間に流す（v1.3.28）。まとめて出すと、次の token の host-mapped の読みがコピーの終わりまで待たされていた（8K prompt の後、約 10 token が 35〜39 ms）。`0`: 従来どおり境目で |
| `STRATA_GLM_ZERO_COPY` | オン | copy engine を使わない token の境目（v1.3.21）: 表の更新・埋め込み行（`token_embd` の pinned コピー、Q8_0 / Q5_K）・トークンを、カーネルが host-mapped で直接読み書きする。`0`: 従来どおり memcpy |
| `STRATA_GLM_GAP_PROF` | オフ | `1`: 答えごとに `glm gap prof` の行を出す。トークンとトークンの間の時間の内訳（host 側のラップと、timing event で測った GPU の空き） |
| `STRATA_GLM_TAIL_SKIP` / `_W` | オフ | ルートの r 位より下（または確率の割合が閾値より下）で VRAM にないエキスパートを計算しない。CPU レーンが遅い環境で速くなるが、答えが変わる（KL 0.01〜0.02 程度） |
| `STRATA_GLM_ROUTE_AFFINITY` | オフ | `<a>`: VRAM にあるエキスパートの選択スコアに +a する（重みはルーターの値のまま）。僅差なら GPU にあるエキスパートが選ばれる（Maya #252 から。v1.3.36）。答えが変わる。GLM-5.3-Flash Maya-S-v2、RTX 5090 で `0.04` が 1 トークン 14.1 ms（既定 15.4、−9%）、VRAM のヒット 56 → 69%、nll 1.631（既定 1.635、既定のロジットに対する KL 0.043）。`0.08` は −13% だが nll が悪化し始める（1.641〜1.651）。hash ルーティングの層は対象外 |
| `STRATA_GLM_ROUTE_AFFINITY_RAM` | オフ | `<a>`: RAM 層のエキスパートにも同じく +a（VRAM にあるものは大きい方）。SSD の層がある構成で、SSD にしかないエキスパートを避ける（v1.3.37 で計測）。DeepSeek V4.1 の tail skip 設定、8 本の評価用シーケンスで、`0.08` が 1 トークン −8〜12%（SSD の読み −22%）、KL 0.2406 → 0.2438、ppl 9.76 → 9.81。`0.16` は −12〜20%、KL 0.2501 / ppl 9.89 |
| `STRATA_GLM_ROUTE_PROF` | オフ | `1`: route kernel が 43,000 ルートごとに段ごとの SM サイクルを出力する（診断用、NVIDIA） |
| `STRATA_GLM_TAIL_SKIP_DISK` | オフ | `1`: 上の省略を、SSD にしかないエキスパートだけに限る（v1.3.29）。`STRATA_GLM_TAIL_SKIP=3` と合わせて DeepSeek V4.1 の速度設定: 1 トークン 56.6 → 34.2 ms、KL 0.229 → 0.240。全エキスパートが VRAM + RAM に載るなら何も変わらない |
| `STRATA_GLM_DEFER` | オフ | expert deferral: 層の MoE のうち CPU レーン側の分を層の終わりで待たず、次の層の attention の後に（元の層の重みで）residual に足す。GPU がレーンを待たなくなる: RTX 5090 + Radeon 8060S で単発 1 トークン 10.9 → 9.5 ms、2 トークンの検証窓 14.4 → 13.1、tok/s +12〜15%。答えの変化は tail skip より大きい（Q8_0 参照に対する KL 0.08、オフなら 0.012。swe-bench L7 16 問 × 3: 107.2 tok/s、resolved 16 問中 13。オフなら 100.9、14）。層分割でも各 part の中で遅らせる（part の最後の層は自分の分を待つ）。AI PRO R9700 で iGPU レーンなし（v1.3.41、DeepSeek V4）: 1 トークンずつ 46.1〜46.5 → 53.0〜53.2 tok/s、CUDA 版の基準に対する KL 0.059（オフなら 0.0057。1 位のトークンの一致はオフの 97% に対して 90%） |
| `STRATA_DS4_IDX`, `STRATA_DS4_ATTN` | 0 | `1`: lightning indexer / prompt の attention を tensor core 版ではなく従来の FP32 カーネルで計算する（A/B 確認用） |
| `STRATA_GLM_MMQ_GLU` | オン | prompt の expert の gate/up の積で SwiGLU まで済ませる（NVIDIA Turing 以降）。`0` で別カーネル（以前のビルドとビット一致） |
| `STRATA_GLM_CPU_MT` | オン | 複数トークンが同じ expert を使うとき（prompt、検証窓）、CPU は重みの行を 1 回だけ展開する。ggml の内積とビット一致（IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS は `STRATA_GLM_CPU_1T` の CPU で格子をレジスタの中で引く。v1.3.41）。`0` で無効 |
| `STRATA_GLM_CPU_1T` | オン | CPU レーンの 1 トークン用の行を軽くする（ggml の内積とビット一致）。VNNI と VBMI のある AVX-512 の CPU（Zen 4 / Ice Lake 以降）では、IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS の格子をレジスタの中で引く（v1.3.41。Ryzen AI Max+ 395 の 16 スレッドで DeepSeek V4 のエキスパート 1 個が 109 → 90 µs、内蔵 GPU なしの生成が +3〜9%。`STRATA_NO_IQ512_VBMI=1` でこれだけ無効）。それ以外の CPU では AVX2 の IQ3_S / IQ3_XXS。`0` で無効 |
| `STRATA_GLM_VRAM_EVICT` | 使用回数 | `lru`: VRAM 層の入れ替えで、最も長く使われていない expert を追い出す（キャッシュの調査用） |
| `STRATA_GLM_DISK_READERS` | オフ | prompt 処理で SSD 上の expert をこの数のスレッドで読む（DRAM なしの NVMe 向け。環境ごとに測ること。DeepSeek V4.1 は `8` で 1K トークンの prompt が 33 → 27 秒） |
| `STRATA_GLM_DISK_LANE` | オフ | `1`: SSD から RAM に読んだエキスパートを、VRAM に送らずその場で CPU / iGPU レーンが計算する（v1.3.29。V4.1 の Q2_K/Q3_K エキスパートで −5%、小さい Q2 では効果なし） |
| `STRATA_DS41_ENGRAM_CACHE_MB` / `_THREADS` | 256 / 15 | DeepSeek V4.1: Engram の行キャッシュと、行を読むスレッド数（v1.3.29） |
| `STRATA_DS41_ENGRAM_ASYNC` | オン | DeepSeek V4.1: トークンの Engram 48 行を、層 0 の計算中に裏のスレッドが読み、層 1 / 14 が GPU 上で到着を待つ（v1.3.36。1 トークン −0.5 ms、出力は同じ）。`0`: トークンの前に読む |
| `STRATA_DS41_ENGRAM_PREFETCH` | 8 | DeepSeek V4.1: prompt の chunk の hash の直後に、その Engram 行を行キャッシュへ読むスレッド数（v1.3.36。1K トークンの prompt で Engram 1.7〜1.9 → 0.6 秒、prompt 全体 −6〜7%）。`0`: オフ |
| `STRATA_DS4_SPEC` | 1（オン） | prompt と応答から作る投機デコード。1 回に試す候補トークン数（1〜3。実測では 1 が最速）。`0` でオフ |
| `STRATA_DS4_SPEC_GATE` | オン | リクエストの採用率が検証窓の分に見合わない間は候補を出さない。`0`: 常に出す |
| `STRATA_DS4_SPEC_MIN` | 3 | 候補を探すときに一致させるトークン数（2〜4） |
| `STRATA_DS4_SPEC_PIPE` | auto（HIP）、1（NVIDIA） | 検証の窓: `1` pipelined（v1.3.43 から列を 2 つのグループに分けて 2 本の HW キューに流し、グループの中はまとめて計算）、`0` batched（dense の重みを窓で 1 回だけ読む）、`auto`: 窓の幅ごとに、最初の窓で両方を 3 回ずつ測り、速いほうをその幅で使い続ける（v1.3.40。幅ごとは v1.3.43 から。HIP の既定。NVIDIA ではまだ確認していない）。RTX 5090 + iGPU では pipelined（swe-bench 90 対 73 tok/s）、AI PRO R9700（HW キュー 2 本、v1.3.43）ではどの幅も pipelined（2 / 4 / 6 / 8 列で 34〜36 / 52〜56 / 71 / 88 ms、batched は 41 / 68.5 / 98 / 133） |
| `STRATA_DS4_SPEC_GROUP` | 窓の半分 | pipelined の窓の 1 グループの列数（v1.3.43）。既定は ceil(T / 2) 列ずつの 2 グループ |
| `STRATA_DS4_WIN_COLS` | オン | 窓のグループで、VRAM にある expert の gate/up と down を列ごとではなく全列まとめて 1 回ずつ起動する（v1.3.43。ビット一致: `STRATA_DS4_WIN_COLS_CHECK=1` で 16 グループに 1 回、列ごとの起動と照合）。`0`: 列ごと |
| `STRATA_GLM_CPU_MULTI` | オン | batched の窓で、1 層ぶんの CPU レーンの expert を全列まとめて 1 回で計算し、重みの行は全列で 1 回だけ読む（v1.3.43。ビット一致）。`0`: 列ごと |

コンテキスト長は `--max-context` で決める。DeepSeek V4 の学習は 1M トークンだが、既定は 65536 で、131072（125K の
prompt）まで測った。アテンションのキャッシュは小さい（65536 で 0.44 GB）が、prompt 用のバッファはコンテキストに
比例して増える。

## llama.cpp との照合

`tools/dsv4_logits_dump.cpp` は、トークン列に対する llama.cpp の logits を書き出す（`deepseek4` に対応した llama.cpp に
対してビルドする）。エンジンの `STRATA_GLM_SCORE` モードで、エンジン自身の logits と比べられる。手順と結果は
[docs/deepseek4/PORT.md](deepseek4/PORT.md)。UD-Q2_K_XL では、3000 トークンで KL 0.009、次トークンの的中率も同じだった。

## うまくいかないとき

- **ビルドが `mma with block scale not supported on .target sm_120` で止まる** - `-DCMAKE_CUDA_ARCHITECTURES=120a` で
  構成し直す。
- **`the engine exited before it was ready`** - エンジンのログ（設定の `log`）を見る。RAM 層の確保でメモリが足りないなら、
  `STRATA_GLM_RAM_GB` を下げるか `STRATA_GLM_RAM_HEADROOM_GB` を上げる。
- **`refers to missing files` や `pack:` のエラー** - pack は、作ったときの 3 つのシャードと同じ場所に置く必要がある。
  シャードを移したら作り直す。
- **思考が一言で止まって回答が始まる** - `max_tokens` が小さすぎる（上の「思考」を参照）。
- **遅く、SSD がずっと動いている** - エキスパートに対して RAM + VRAM が足りない。ほかのプログラムを閉じるか、RAM を増やす。
- **ポートが使われている** - 別のサーバーが動いている。`--port` で変える。

## 制限

- ドラフトモデルはない。この GGUF には MTP ブロックがなく、DeepSeek の DSpark（ドラフトモデル）には未対応。
  投機デコードは文章自体から候補を作る（`STRATA_DS4_SPEC`）。
- 会話のディスク保存（`STRATA_GLM_SLOTS`）は DeepSeek V4 では無効。直前の会話の再利用は動く。
- 画像は扱わない（ここでは DeepSeek V4 Flash はテキストのみ）。
- 複数 GPU: 層の分割は動く（「複数の GPU」）が、遅い 2 枚目を足しても速い 1 枚 + iGPU レーンには
  届かない。そのとき `auto` は速いカード 1 枚で動かす。Windows は DeepSeek V4 では未確認。
- `./abyss.sh`（Maya から引き継いだインストーラ）は GLM 用。DeepSeek V4 は `./setup-deepseek.sh`。DeepSeek V4.1 は手作業
  （[docs/deepseek41/README.md](deepseek41/README.md)）。
- DeepSeek V4.1: 投機デコード（MTP 層を持つ GGUF がない）、層の分割、会話のディスク保存はまだない。prompt のアテンションは
  1 トークンずつ。v1.3.31 までは、コンテキストの 16,385 トークン目で落ちた（8K を超えるとアテンションの行選びも崩れていた）。
  長い会話には v1.3.32 以降を使う。
