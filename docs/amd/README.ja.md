# AMD Radeon で Abyss を動かす（HIP、試験的）

[English](README.md)

Radeon のドライバと ROCm を入れ、Abyss の HIP 版をビルドして動かすまでの手順です。単体の Radeon と Ryzen AI Max
の内蔵 GPU（iGPU）の両方を扱います。両方がある PC では、単体カードでエンジンを動かし、RAM 層のエキスパートを
iGPU が計算します（iGPU レーン）。

## 対応する GPU

| GPU | アーキテクチャ | 使い方 | 確認状況 |
|---|---|---|---|
| Radeon AI PRO R9700、RX 9070 / 9070 XT | gfx1201 | エンジン | R9700 で測定（2026-10-20、下の表） |
| RX 7900 XT / XTX | gfx1100 | エンジン | コミュニティのビルド（[issue #4](https://github.com/zephel01/Abyss/issues/4)、Windows） |
| Ryzen AI Max の Radeon 8050S / 8060S | gfx1151 | エンジン（単体）、または iGPU レーン | 8060S で測定 |

- エンジンのビルド対象は gfx1100、gfx1201、gfx1151 です。ほかの Radeon（RDNA2 以前、gfx1200 の RX 9060 など）は
  対象外です。
- iGPU レーンは gfx1103 / gfx1150〜1153 の内蔵 GPU で動きます。
- NVIDIA と AMD を 1 つのプロセスで使うこと（CUDA と HIP の混在）には対応していません。NVIDIA の GPU がある PC
  では、CUDA 版が Radeon の iGPU をレーンに使います（[MANUAL の「iGPU レーン」](../MANUAL.ja.md)）。

## 確認した環境

| 項目 | 内容 |
|---|---|
| PC | GMKtec EVO-X2（Ryzen AI Max+ 395、Radeon 8060S、RAM 128 GB のうち OS に 94 GB） |
| 単体 GPU | Radeon AI PRO R9700 32 GB（OCuLink、PCIe Gen4 x4） |
| OS | Ubuntu 24.04.4 LTS、カーネル 7.0.0-38-generic |
| ドライバ | amdgpu-dkms 6.19.14（amdgpu-install 31.40） |
| ROCm | 10.0.0（apt の `amdrocm` パッケージ、`/opt/rocm/core-10.0`） |

ROCm 7 以降で動きます（セットアップが確かめます）。

## 1. ドライバ（amdgpu）

ROCm はカーネルの amdgpu ドライバと、ドライバが作る `/dev/kfd` を通して GPU を使います。

1. **ドライバを入れる。** 上の PC では、AMD の `amdgpu-install`（Radeon Software for Linux）で
   `amdgpu-dkms` と `amdgpu-dkms-firmware` を入れています。DKMS 版は、カーネルを更新したときに作り直されます。
   - 新しいカーネルに入っている amdgpu でも、`/dev/kfd` があり、下の `rocminfo` に GPU が出れば ROCm は動くはずです。
     ただし、この組み合わせは確認していません。
   - Secure Boot が有効な PC では、DKMS のモジュールに署名が要ります（インストール時に MOK の登録を求められます）。
2. **ユーザーを `render` と `video` のグループに入れる。** 入れたあと、ログインし直します。

   ```sh
   sudo usermod -aG render,video $USER
   ```

3. **確かめる。**

   ```sh
   ls -l /dev/kfd /dev/dri/renderD*      # どちらも render グループで読み書きできる
   modinfo amdgpu | grep -E "^(filename|version)"
   lspci -D | grep -i -E "vga|display"   # GPU ごとの PCI アドレス（例: 0000:c7:00.0）
   ```

## 2. ROCm

### ROCm 10（apt の amdrocm パッケージ）

ROCm 10 のパッケージは GPU のアーキテクチャごとに分かれています。**使う GPU のアーキテクチャをすべて入れて
ください。** 入っていないアーキテクチャの GPU では、rocBLAS / hipBLASLt のカーネルが読めません。

1. AMD の ROCm インストールの説明に従って、リポジトリと署名鍵を登録します。上の PC の
   `/etc/apt/sources.list.d/amdrocm-stable.sources` は次のとおりです。

   ```
   Types: deb
   URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/
   Suites: stable
   Components: main
   Architectures: amd64
   Signed-By: /etc/apt/keyrings/amdrocm.gpg
   ```

2. GPU ごとのメタパッケージと、ビルドに使う開発用パッケージ（1 つでよい）を入れます。

   ```sh
   sudo apt update
   # R9700 / RX 9070 と Ryzen AI Max の iGPU がある PC
   sudo apt install amdrocm10.0-gfx1201 amdrocm10.0-gfx1151 amdrocm-core-dev10.0-gfx1151
   # RX 7900 XT / XTX だけの PC
   sudo apt install amdrocm10.0-gfx1100 amdrocm-core-dev10.0-gfx1100
   ```

   `amdrocm10.0-<arch>` からは、そのアーキテクチャの core / blas / dnn / fft / rand / rccl / solver / sparse
   が入ります。あとから GPU を足したときも、そのアーキテクチャの分を追加するだけで済みます。今入っているものを
   消す必要はありません。

3. ROCm 10 は `/opt/rocm/core-10.0` に入り、`/opt/rocm/{bin,lib,llvm}` はそこへのリンクになります。Abyss の
   セットアップは `/opt/rocm` を探し、そこにコンパイラがなければ一番新しい `/opt/rocm/core-<版>` を使います。
   別の場所にあるときは `ROCM_PATH` を設定します。

### ROCm 7

ROCm 7（`amdgpu-install` の `rocm`、または amdrocm 7.x）でも動きます。amdrocm 7.x もアーキテクチャごとに
分かれているので、入れ方は上と同じです（`amdrocm7.14-gfx1201` など）。複数の版を入れると `/opt/rocm` が
どちらを指すかがわかりにくくなるので、使わない版は消すか、`ROCM_PATH` で版を指定してください。

### Windows

Windows では手順が別になります（試験的。RX 7900 XT での報告があり、R9700 は未確認です）。

1. **ドライバ:** AMD Software: Adrenalin Edition を入れます（https://www.amd.com/en/support）。
2. **Visual Studio Build Tools**（2022 または 2026、「C++ によるデスクトップ開発」）を入れます。ROCm のコンパイラが、
   Visual Studio のヘッダとライブラリを使います。

   ```bat
   winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

3. **64 ビットの Python 3.10 以降**を入れます（`winget install -e --id Python.Python.3.12 --scope user`）。
4. **`setup-deepseek.bat --backend hip` を実行します。** GLM-5.3-Flash を使う場合は `START-ABYSS.bat` です。
   - ROCm は、セットアップが聞いてから AMD の ROCm SDK（TheRock の wheel、7.14.1）を Abyss の `.venv` に入れます。
     システムには何も入れません。
   - 選んだ GPU のアーキテクチャのカーネル（`rocm-sdk-device-<arch>`）も一緒に入ります。足りなければ、セットアップが
     入れ直しを勧めます。そのため、Linux の「BLAS のカーネルがない」問題は Windows では起きません。
   - GPU の番号は ROCm の `hipInfo` の順です。hipInfo が見つからないときは Windows の一覧の順になり、HIP の番号と
     違うことがあるという警告が出ます。

Windows での違い:

- **iGPU レーンは Linux だけです。** Windows では、単体の Radeon と Ryzen AI Max の iGPU があっても、RAM 層の
  エキスパートは CPU が計算します。サーバーは `HIP_VISIBLE_DEVICES` に設定の GPU だけを渡します。
- **`"gpu_pci"` は使いません。** 番号は hipInfo のものです。GPU を足したり外したりしたあとは、セットアップを
  もう一度実行して番号を確かめてください。
- **`HSA_OVERRIDE_GFX_VERSION` は Linux の ROCr の設定です。** Windows では関係ありません。
- **RAM の扱い:** Windows は pinned の RAM 層をコミット上限（RAM + ページファイル）で数えます。ページファイルが
  小さいと、RAM が空いていてもエキスパートが SSD に回ります。RX 7900 XT + 128 GB では、次の設定ですべての
  エキスパートが RAM に載りました（[issue #4](https://github.com/zephel01/Abyss/issues/4)）。
  - ページファイルを大きくする（例: 64 GB）
  - `STRATA_GLM_RAM_HEADROOM_GB=4`
  - `STRATA_GLM_RESERVE_MB=512`（そのカードに画面をつないでいない場合）
  - `--prefill 8192`
- **Intel のハイブリッド CPU:** `STRATA_GLM_CPU_PIN=0-15` で、CPU レーンを P コアのスレッドに載せられます。
  プロセス全体を P コアに絞ると、HIP ランタイムのスレッドが足りなくなって遅くなります。

ビルドで起きた問題（MSVC 14.51 と TheRock 7.14.1 の `isgreater` の衝突など）は
[AMD HIP（Maya）](../maya/AMD_HIP.md) の「Windows 11」にあります。

## 3. 環境変数

- **`HSA_OVERRIDE_GFX_VERSION` は設定しないでください。** 以前、ROCm が対応していなかった iGPU のために
  `11.5.1` などを設定する方法が使われていました。gfx1151 はもう ROCm が直接対応しているので不要です。
  これが残っていると、R9700 のような gfx12 の GPU では ROCr が `HSA_STATUS_ERROR_OUT_OF_RESOURCES` で起動しません。

  ```sh
  env | grep HSA_                                     # 何も出なければよい
  grep -rn HSA_OVERRIDE ~/.bashrc ~/.profile /etc/profile.d/ /etc/environment 2>/dev/null
  ```

  見つかった行はコメントアウトして、ログインし直してください。
- `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` も、普段は設定しないでください。Abyss のサーバーが、設定ファイル
  の GPU からエンジンに渡す値を決めます。
- **`GPU_MAX_HW_QUEUES` は Abyss に任せて 2 にしてください。** R9700 向けの手順には、MES ファームウェアの不具合で
  アイドル時の電力が下がらない問題の回避策として `export GPU_MAX_HW_QUEUES=1` を勧めるものがあります。1 にすると、
  エンジンのストリームがすべて 1 本の HW キューに載り、あるストリームを待つと他のストリームの仕事まで待つことに
  なります。上の PC での DeepSeek V4 の 8K / 16K の prompt は、1 で 586〜650 / 676〜684 tok/s、ROCm の既定（4）で
  653〜737 / 748〜756、2 で 866〜995 / 999〜1,011 tok/s でした。decode は同じです。エンジンを読み込んだ状態の
  アイドル時の電力は、1 でも 2 でも 12 W でした。セットアップは設定ファイルに `"GPU_MAX_HW_QUEUES": "2"` を書き、
  サーバーはシェルにも設定ファイルにも値がなければ 2 を渡します。2 でアイドル時の電力が下がらない場合（古い
  ファームウェア）は、設定ファイルの `env` に `"GPU_MAX_HW_QUEUES": "1"` を書いてください。

  ```sh
  grep -rn GPU_MAX_HW_QUEUES ~/.bashrc ~/.profile /etc/profile.d/ /etc/environment 2>/dev/null
  amd-smi metric -g 0 -p    # エンジンを読み込んだまま、アイドル時のカードの電力
  ```

## 4. 動作確認

```sh
rocminfo | grep -E "Marketing Name|Name: +gfx"       # 使う GPU がすべて出る
amd-smi list                                         # GPU の番号と PCI アドレス
hipconfig --version
ls /opt/rocm/lib/rocblas/library | grep -c gfx1201  # 0 なら amdrocm10.0-gfx1201 が入っていない
```

上の PC では、`rocminfo` に R9700（gfx1201）、Radeon 8060S（gfx1151）、NPU が出ます。rocBLAS の gfx1201 のファイル
は 110 個です。

## 5. Abyss のセットアップ

```sh
./setup-deepseek.sh --backend hip   # NVIDIA の GPU がなければ、--backend なしでも HIP を選ぶ
```

- セットアップは KFD のトポロジーから GPU を見つけ、VRAM が一番大きい GPU を選びます。`--gpu N` で番号を指定する
  こともできます。番号は `amd-smi list` の `GPU:` と同じです。
- エンジンは `build-hip/strata` にビルドされます。初回は 10〜20 分かかります。ROCm に `hipcc` があれば、iGPU
  レーンの `build-hip/libabyss_igpu.so` も一緒にビルドされます。
- 設定ファイルには `"backend": "hip"`、`"gpu"`（HIP の番号）、`"gpu_pci"`（PCI アドレス）、`"lib_dirs"`（ROCm の
  `lib`）が入ります。

## 6. 単体の Radeon と iGPU を一緒に使う（Ryzen AI Max + eGPU）

単体の Radeon をつないだ Ryzen AI Max では、次の分担になります。

- エンジン（attention、VRAM のエキスパート）は単体カードで動きます。
- RAM 層のエキスパートは iGPU が計算します。iGPU は RAM を約 240 GB/s で読めるのに対し、CPU のスレッドは約 75 GB/s です。

Linux だけで動きます（Windows は上の「Windows」を参照）。
設定は要りません。サーバーが iGPU を `HIP_VISIBLE_DEVICES` の後ろに足し（例: `0,1`）、エンジンは最初の GPU で
動きます。レーンは integrated の GPU を自分で探して使います。

- `STRATA_GLM_IGPU=0` を設定すると、iGPU を使わず CPU が計算します。
- GLM の自動の層分割は iGPU を候補にしません。iGPU は分割先ではなく、レーンとして使います。
- エンジンのログに `iGPU lane on` と `-> the iGPU computes the lane` が出ていれば、レーンが使われています。

### GPU の番号が変わる場合

HIP の番号は、起動時にドライバが GPU を見つけた順（KFD のトポロジーの順）です。上の PC では R9700 が
`0000:c7:00.0`、iGPU が `c8:00.0` なので、R9700 が 0 番、iGPU が 1 番です。次の場合は順番が入れ替わります。

- 起動したあとで eGPU をつないだ場合や、起動時に eGPU の電源が入っていなかった場合
- 別の OCuLink / USB4 のポートにつないだ場合や、BIOS やカーネルの設定でバス番号が振り直された場合

そのため、セットアップは GPU の PCI アドレスを `"gpu_pci"` として設定ファイルに書きます。起動のたびにサーバーが
アドレスから今の番号を調べ、番号が変わっていれば次のように表示して正しい GPU を使います。

```
[abyss] the AMD GPU(s) at 0000:c7:00.0 are HIP number 1 now (the config says 0): using 1
```

`"gpu_pci"` のない古い設定ファイルは、番号のまま使われます。手で足す場合は、`amd-smi list` か
`lspci -D` に出るアドレスを書いてください。

```json
"gpu": [0],
"gpu_pci": ["0000:c7:00.0"]
```

### OCuLink / USB4 でつなぐとき

- リンクの速さを確かめます。

  ```sh
  sudo lspci -vv -s c7:00.0 | grep -E "LnkCap|LnkSta"
  ```

  R9700 のカードの中にはスイッチがあり、その手前は Gen5 x16 と表示されます。そのため、根元の root port
  （上の PC では `00:03.1`）の値を見てください。アイドル中は 2.5 GT/s まで下がります。
- 上の PC では、pinned メモリで H2D 7.24 GB/s、D2H 7.03 GB/s でした（Gen4 x4）。エキスパートを RAM から運ぶ量が
  多いので、帯域は速度に直接効きます。
- 上の PC のカーネルのパラメータには `pci=realloc pcie_aspm=off pcie_port_pm=off iommu=pt pcie_ports=native` が
  入っています。どれも、以前 NVIDIA の eGPU を安定させるために入れたものです。Radeon でどれが必須かは
  確かめていません。

## 7. 速度の目安

R9700 + Radeon 8060S（iGPU レーン）、RAM 94 GB、swe-bench の 6 問の生成速度（tok/s）です。v1.3.39 で
`bench/model_gpu_cmp.sh` を使って測りました。

| モデル | CPU レーン | iGPU レーン | 8K / 16K の prompt |
|---|---|---|---|
| DeepSeek V4 UD-Q2_K_XL（q5k パック） | 36.4 | 42.4 | 452 / 599 |
| DeepSeek V4 huihui Q2 | 33.9 | 38.1 | 277 / 324 |
| GLM-5.3-Flash Maya-S-v2 | 25.8 | 35.8 | 418 / 541 |
| GLM-5.3-Flash Maya-S24 | 28.2 | 41.2 | 433 / 562 |

v1.3.40（HW キュー 2 本、gfx1201 向けのカーネル）では、DeepSeek V4 UD-Q2_K_XL の 8K / 16K の prompt が
866〜995 / 999〜1,011 tok/s、1 token ずつの生成が iGPU レーンありで 52.2〜52.9 tok/s、なしで 41.8〜42.8 tok/s
です（v1.3.39 はレーンありで 44.7〜45.5）。GLM-5.3-Flash Maya-S-v2 の prompt は 644〜716 / 797〜843 で、生成は
変わりません。

- **精度:** CUDA 版の基準に対する KL は 0.0059 です（v1.3.39 は 0.0052。小さい GEMV のまとめ起動で足し算の順番が
  変わったため）。CUDA 版どうしのぶれ（0.0057〜0.0060）と同じ水準です。
- **生成速度:** RTX 5090 + iGPU の 0.42〜0.58 倍、RTX 3090 + iGPU の 0.73〜0.91 倍です（v1.3.39）。
- **prompt:** 5090 の 0.4〜0.5 倍です（v1.3.40。v1.3.39 は 0.2〜0.4 倍）。

## 8. うまくいかないとき

| 症状 | 原因 | 対処 |
|---|---|---|
| `rocminfo` が `HSA_STATUS_ERROR_OUT_OF_RESOURCES` で終わる | `HSA_OVERRIDE_GFX_VERSION` が残っている | 「3. 環境変数」の手順で消し、ログインし直す |
| `rocminfo` が `/dev/kfd` を開けない（permission denied） | `render` グループに入っていない | `sudo usermod -aG render,video $USER` のあとログインし直す |
| 生成はできるのに、最初の prompt で止まる。サーバーが 503 を返し、180 秒でエンジンが終了させられる | rocBLAS / hipBLASLt にその GPU のカーネルがない（例: gfx1151 の分しか入っていない） | `sudo apt install amdrocm10.0-<arch>`（R9700 なら gfx1201）。確かめ方は「4. 動作確認」 |
| エンジンのログに `No compatible code objects found for: gfx1201` が出る。iGPU レーンで `invalid device function` になる | 古いビルドの `libabyss_igpu.so` に、エンジンの GPU 用のコードが入っていない。HIP は、見えている GPU のどれか 1 つでもコードがないライブラリだと、そのカーネルを全部の GPU で使えなくする | 今のソースでビルドし直す（`./setup-deepseek.sh --backend hip`） |
| 単体カードがあるのに遅く、ログの GPU が iGPU（gfx1151）になっている | HIP の番号が入れ替わり、設定の `"gpu"` が iGPU を指している | 設定に `"gpu_pci"` を書く（「GPU の番号が変わる場合」）。`rocminfo` で順番を確かめる |
| 単体カードと iGPU があるのに、ログに `-> the iGPU computes the lane` が出ない | 設定の `env` で `HIP_VISIBLE_DEVICES` を単体カードだけにしている、または `STRATA_GLM_IGPU=0` | どちらも外す |
| （Windows）`ROCm's hipInfo sees no AMD GPU` と出る | ドライバが古い、または入っていない | Adrenalin を更新して再起動し、警告に出る hipInfo で確かめる |
| （Windows）ビルドが `__device__ function 'isgreater' cannot overload ...` で止まる | MSVC 14.51 と TheRock 7.14.1 の衝突 | [AMD HIP（Maya）](../maya/AMD_HIP.md) の「Windows 11」の手順でヘッダを直す（[issue #2](https://github.com/zephel01/Abyss/issues/2) にスクリプトがある） |
| （Windows）RAM が空いているのに、エキスパートが SSD から読まれる | ページファイルが小さく、コミット上限に当たっている | 「Windows」の「RAM の扱い」の設定 |
| ビルドが `ROCm's HIP compiler ... and hipBLAS are required` で止まる | 開発用パッケージがない、または `/opt/rocm` が別の版を指している | `amdrocm-core-dev10.0-<arch>` を入れる。または `ROCM_PATH=/opt/rocm/core-10.0` を設定する |
| prompt が「7. 速度の目安」よりかなり遅い（R9700 で DeepSeek V4 の 16K の prompt が約 1,000 ではなく約 680 tok/s） | シェルの `GPU_MAX_HW_QUEUES=1`（R9700 向けの手順の設定）がエンジンに渡っていて、設定ファイルで上書きしていない | 設定ファイルの `env` に `"GPU_MAX_HW_QUEUES": "2"`（セットアップは書く）。「3. 環境変数」を参照 |
| DeepSeek V4.1 で、エンジンのログに `iGPU lane: the late call ran 21-27 ms` が多い | エンジンが SSD を待っている間に iGPU が寝ていた（v1.3.40 より前は、ルーティングから 20 ms で keep-alive が止まっていた） | 更新する。または `STRATA_GLM_IGPU_POKE_MS=500` |
