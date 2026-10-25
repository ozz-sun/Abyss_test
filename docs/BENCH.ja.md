# 自分の PC で Abyss を測る（`--bench`）

[English](BENCH.md)

Abyss は、GPU メモリに収まらないモデルを動かします。1 トークンごとに、使うエキスパートを GPU メモリ・RAM の層・SSD のどこかから持ってきて、一部は CPU（または iGPU）のレーンで計算します。そのうちどれが速度を決めるかは、PC によって違います。GPU、PCIe スロット、RAM、SSD、CPU の組み合わせ次第です。私たちが測れるのは手元の PC だけなので、このベンチマークはあなたの PC で回して issue に貼ってもらう前提で作っています。どの PC でも同じ条件で数字が出て、1 トークンの時間がどこで使われているかを示し、正常な実行と同じ計算結果になっているかも確かめます。

## 実行方法

先に Abyss を止めてください（ベンチマークは、Abyss が使っている GPU メモリを使います）。そのうえで次のどれかを実行します。

```
./abyss.sh --bench                 # 数分: 速度、時間の内訳、品質チェック
./abyss.sh --bench --deep          # + GPU の稼働率と、PCIe / レーンの分担率の振り分け（ボトルネック探し）
./abyss.sh --bench --full          # + 各質問を 3 回、32k トークンのプロンプト
./abyss.sh --bench --config abyss-ornith-r9700l.json    # 最後に使ったものと別のモデルを測る
```

Windows では `START-ABYSS.bat --bench` です（オプションは同じ）。結果は Abyss のフォルダーの `abyss-bench.txt` に書き出され、`--report` にも含まれます。どこにも送信されません。ファイルにはホスト名・ユーザー名・フォルダーのパスは入りません。中身を確認してから、[速度の報告](https://github.com/zephel01/Abyss/issues/new?template=bench.md)に貼ってください。

ダッシュボードと同じ起動方法で、エンジン単体に次のことをさせます。

| 段階 | 内容 | 目的 |
|---|---|---|
| decode | 決まった質問 3 つに 256 トークンずつ回答。greedy、思考なし（最初に 1 回慣らしの回答をします） | どの PC でも同じ処理なので、tok/s を比べられる |
| prefill | Abyss のドキュメントから 2k と 8k トークンのプロンプト（`--full` では 32k も） | プロンプトを読む速さ |
| カウンター | 各回答のあとにエンジンが出す 1 行（`glm stat`） | 1 トークンの時間がどこで使われているか |
| PCIe | エンジンが動いている間に読んだ GPU のリンク速度 | 止まっている間は Gen1 まで落ちるので、そのとき読んだ値は当てにならない |
| SSD | モデルの GGUF を 2 GB、ディスクから読む（ページキャッシュを通さない） | エンジンが SSD から読むときの速さ |
| 品質 | 決まった 512 トークンを採点し、`bench/ref/<モデル>`（正常な実行の結果）と比べる | この GPU でカーネルが壊れていれば、KL が大きくなる |
| `--deep` | GPU の稼働率（`STRATA_GLM_GAP_PROF=1`）と、PCIe の分担率を 0 / 0.25 / 0.5 / 0.75 / 1 にした 5 回の回答 | ここで何が速度を動かしているか |

## 結果の読み方

1 行目は要約です。PC どうしを 1 行で比べられます（docs/MODELS.md の表は、こうした行から作っています）。

```
Abyss v1.3.55 | Ornith-1.5-122B-UD-Q4_K_XL | AMD Radeon AI PRO R9700 32 GB | PCIe Gen4 x4 (~7.9 GB/s), the GPU can do
Gen5 x16 | RAM 94 GB | SSD 4.9 GB/s | decode 56.4 tok/s | prompt 8k 960 tok/s | quality OK | bottleneck: gpu
```

### 1 トークンの時間の内訳

```
- a token: 17.8 ms (56.2 tokens/s)
- SSD waits: 0.0 ms a token (0%), 0.0 expert reads a token
- iGPU lane: busy 7.1 ms a token (40%), 246 experts a token
- GPU memory: holds 47% of the experts a token uses; 0.4 a token come over PCIe from RAM
- drafter: 76% of its drafts accepted
```

| 行 | 意味 | 値が悪いときに分かること |
|---|---|---|
| a token | 1 トークンの生成時間（ドラフトモデルを使うときは、採用されたトークン 1 つあたり） | - |
| SSD waits | SSD から読むエキスパートを、GPU が待った時間 | 1 トークンの 4 分の 1 以上なら、モデルが GPU + RAM に収まっていない |
| CPU / iGPU lane | レーンが自分の受け持つエキスパートの計算に使った時間（GPU と並行して動く） | 1 トークン全体に近いなら、レーンが一番遅い道になっている |
| GPU memory | 1 トークンで使うエキスパートのうち GPU メモリにすでにある割合と、RAM から PCIe で送ったものの数 | 1 トークンでたくさん送っているなら、PCIe がトークンを運んでいる |
| drafter | ドラフトモデルの推測のうち、本体が採用した割合 | 低いなら、ドラフトモデルは得より損が大きい（`--dspark` を切るか、別のドラフトモデルにする） |
| tail skip | 精度を少し落とす設定が入っている（その KL は docs/MODELS.md に） | - |

### 判定

カウンターから、一番ありそうなボトルネックを判定します。目安の規則で、R9700 と RTX 5090 で確かめています。

| 判定 | 規則 | 速くする方法 |
|---|---|---|
| **ssd** | SSD の待ちが 1 トークンの 25% 以上 | GPU + RAM に収まる小さい量子化、RAM の層を増やす（`STRATA_GLM_RAM_GB`）、速い SSD |
| **cpu** | レーンの稼働が 1 トークンの 80% 以上 | `./abyss.sh --calibrate`（レーンの分担とスレッド数。AMD では振り分けで一番速かった分担率を `STRATA_GLM_PCIE_SHARE` に）、CPU の電源モード、Radeon APU なら iGPU レーン |
| **pcie** | 1 トークンで 10 個以上のエキスパートを PCIe で送っている | もっと良いスロット（Gen4 x16 は x4 / OCuLink の 4 倍運べる）、`--calibrate`（AMD では `STRATA_GLM_PCIE_SHARE`） |
| **gpu** | 上のどれでもない（または `--deep` でドラフトモデルなしのとき、GPU の稼働率が 85% 以上） | `./abyss.sh --gpu-tune` と報告。この GPU 用のカーネルに改善の余地があるかもしれない |

SSD で詰まっている例です（同じ PC で DeepSeek V4.1 Flash 1017 UD-IQ2_XXS、300 GB）。

```
- a token: 50.1 ms (20.0 tokens/s)
- SSD waits: 19.5 ms a token (39%), 8.5 expert reads a token
- iGPU lane: busy 6.4 ms a token (13%), 96 experts a token
- likely bottleneck: the SSD: the experts that fit neither in GPU memory nor in the RAM tier are read from disk every token
```

「gpu」は、ほかに目立つものがないときにも出ます。メモリの各層が間に合っていて、残る改善先は GPU 自身のカーネル（とトークン間のホスト側の処理）、という意味です。私たちが持っていない GPU からの報告が一番役に立つのは、ここです。

### `--deep`：分担率の振り分け

GPU メモリにないエキスパートを、エンジンは 2 つの道に振り分けます。レーン（RAM から読んで CPU / iGPU で計算）と、PCIe（GPU に送って計算）です。`--deep` は、この分担率を 5 段階に変えて、同じ質問に答えさせます。その前後に 1 回ずつ、エンジン自身が選んだ分担率でも答えさせます。振り分けはベンチの最後に回るので、RAM の層が温まっていて、回答も短くなります。そのため、上の生成速度ではなく、同じ条件で測ったこの 2 回と比べます。

```
  share 0.00:   58.4 tokens/s  <- fastest
  share 0.25:   23.3 tokens/s
  ...
  share 1.00:   12.2 tokens/s
  the engine's own choice is already the fastest; share 1.00 is 4.8x slower - PCIe is the narrow path here
```

- 一番速い分担率が、エンジン自身の分担率をはっきり上回る：NVIDIA なら `./abyss.sh --calibrate` で、この PC 向けに設定できます。AMD では `--calibrate` が使えないので、config の `"env"` に `"STRATA_GLM_PCIE_SHARE": "<分担率>"` を書いてください。
- エンジンの既定がすでに最速で、変化が急：片方の道がずっと細いということです。この例では PCIe（OCuLink 経由の Gen4 x4）で、エキスパートを 1 つ PCIe に回すたびに、レーンで計算するより時間がかかります。
- 変化がほとんどない：PCIe もレーンも速度を決めていません。GPU（か SSD）を見てください。

### 品質チェック

```
quality: OK (KL 0.0078 vs a known-good run, same top token 98.2%)
```

エンジンが、決まった文章（英語と日本語）512 トークンを 1 トークンずつ採点します。そして各位置の上位 32 候補の対数確率を、同じモデルが私たちの PC で出した値と比べます。参照データは `bench/ref/<モデル>.json` と `.topk` で、`tools/bench_ref.py` で作っています。同じ PC でも KL は 0〜0.01 ほど出ます。どのエキスパートを GPU とレーンのどちらで計算するかが実行ごとに少し変わり、両者で丸めの誤差が違うためです。別の GPU では、その計算の誤差が少し上乗せされます。判定は、0.02 までが **OK**、0.05 までが **slightly off**、それを超えると **OFF** です。OFF のときは、この PC が正常な実行と違う計算をしています（issue #9 のような、その GPU でのカーネルの不具合）。報告してください。

- 品質チェックには、同じモデルのファイルが必要です。別の量子化や、上乗せをしたパック（`tools/dense_overlay.py`）では「skipped」になります。
- わざと数値を変える設定（`STRATA_GLM_TAIL_SKIP*`、`STRATA_GLM_ABL`、`STRATA_GLM_DEFER`）は、採点のときだけ外し、結果にそう書きます。
- 参照データがあるモデル：Ornith-1.5-122B UD-Q4_K_XL、DeepSeek V4.1 Flash 1017 UD-IQ2_XXS、DeepSeek V4 Flash 0731 UD-Q2_K_XL、GLM-5.3-Flash Maya-S-v2 IQ2_XXS

## 対応環境

| | Linux・AMD | Linux・NVIDIA | Windows・NVIDIA | Windows・AMD |
|---|---|---|---|---|
| 速度、カウンター、判定、品質チェック、`--deep` | ○ | ○ | ○ | ○ |
| PCIe のリンク速度 | ○（sysfs。ルートポートまでで一番細いリンク） | ○（nvidia-smi） | ○（nvidia-smi） | × |
| SSD の速度 | ○ | ○ | ○（キャッシュを通さない読み出し） | ○（キャッシュを通さない読み出し） |

AMD のカードは、カード内部のスイッチを x16 と報告します。ベンチマークはルートポートまでたどるので、OCuLink の x4 やチップセット経由のスロットも、実際のとおりに出ます。

## 報告に添えると助かるもの

- `./abyss.sh --gpu-tune`（1 分ほど、モデル不要）：GPU ごとのカーネルの選び方。`abyss-gpu-tune.txt` も貼ってください。
- `./abyss.sh --report`：PC、セットアップ、エンジンのログの速度の行（`abyss-report.txt`。ベンチマークの結果も含みます）。
- 変更の前後：`--bench` を 2 回回して、両方を貼ってください。何が変わったかが、カウンターに出ます。
