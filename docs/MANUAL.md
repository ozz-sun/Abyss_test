# Abyss manual

How to build Abyss, set up DeepSeek V4 Flash (and V4.1 Flash), run it and tune it. Japanese: [MANUAL.ja.md](MANUAL.ja.md).

Abyss is a fork of [Project Maya](https://github.com/mw00/project-maya) (MIT), itself grown out of
[Strata](https://github.com/Niko1221/Strata) (MIT). The engine, the expert tiers, the server and the dashboard are
Maya's; DeepSeek V4 and V4.1 support is what Abyss adds. Maya's own documentation (GLM-5.3-Flash, AMD, multi-GPU, the
installer) is kept under [docs/maya/](maya/README.md).

- [What you need](#what-you-need)
- [Setup in one command](#setup-in-one-command)
- [Build](#build)
- [Get the model](#get-the-model)
- [Make the pack](#make-the-pack)
- [Configure and start](#configure-and-start)
- [Using it](#using-it)
- [DeepSeek V4.1 Flash](#deepseek-v41-flash)
- [EXL3 quants (GLM-5.3-Flash)](#exl3-quants-glm-53-flash)
- [Memory and speed](#memory-and-speed)
- [Settings](#settings)
- [Checking the output against llama.cpp](#checking-the-output-against-llamacpp)
- [Something went wrong?](#something-went-wrong)
- [Limitations](#limitations)

## What you need

| | Tested | Notes |
| --- | --- | --- |
| OS | Ubuntu (Linux) | Windows is not tested for DeepSeek V4 |
| GPU | RTX 5090 (32 GB), RTX 3090 | build for your GPU's architecture (`120a` for RTX 50, `86` for RTX 30; several at once for a split). On pre-Blackwell GPUs the MXFP4 expert rows run through the q8_1 path instead of FP4 MMA; before Ampere (sm_80) the indexer and the prompt attention use the older FP32 kernels. A V100 (sm_70) builds with a CUDA 12 toolkit but has not been run. AMD Radeon: see "AMD GPUs" (experimental) |
| RAM | 94 GB | the model is 96.8 GB: with VRAM + RAM above ~100 GB every expert stays in memory; with less, the rest are read from the SSD (works, slower) |
| Disk | NVMe SSD, ~100 GB free | the GGUF files; the pack adds 0.2 GB |
| CUDA | 13.3 | with `nvcc`; 12.9 for a V100 (`--nvcc`) |
| Compiler | g++ 13 | a version your CUDA accepts |
| Python | 3.12 | a private `.venv` in the Abyss folder |
| ROCm (optional) | 7 (`hipcc`) | an AMD APU's integrated GPU computes the RAM-tier experts (the iGPU lane, below); without it the CPU does |

`./abyss.sh` (Maya's installer, renamed) still installs Maya's GLM-5.3-Flash models. DeepSeek V4 is set up with
`./setup-deepseek.sh` (next section), or by hand with the steps after it.

## Setup in one command

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --download-model
```

`setup-deepseek.sh` makes the `.venv`, then runs `tools/abyss_setup.py`. Each step is skipped when it is already done,
so it can be run again at any time:

1. checks the PC (an NVIDIA GPU, RAM);
2. Python packages, llama.cpp's source at the pinned commit;
3. compiles the engine for the GPU (`120a` on RTX 50);
4. downloads the model to `../Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/<quant>` - it asks first unless
   `--download-model` is given - and checks every file against Hugging Face's SHA-256;
5. makes the pack next to the shards;
6. writes the run config `abyss-<name>.json` and the start script `run-abyss-<name>.sh` in the Abyss folder;
7. creates the **Abyss-tools** folder beside Abyss: `abyss.sh start | stop | restart | status | update | build | log |
   models`, from the templates in `ops/abyss-tools/`. Running the setup again brings `abyss.sh` / `common.sh` up to date
   with the templates (the old ones kept as `.bak`) and keeps your own `README.md` there.

```sh
../Abyss-tools/abyss.sh start      # then http://127.0.0.1:8082/
../Abyss-tools/abyss.sh stop
```

| Option | |
| --- | --- |
| `--quant UD-IQ3_XXS` | another quant of unsloth/DeepSeek-V4-Flash-0731-GGUF (default `UD-Q2_K_XL`, the tested one) |
| `--name NAME` | the model's name in `abyss-<name>.json` (default `dsv4-<quant>`, e.g. `dsv4-q2_k_xl`) |
| `--context N`, `--port N` | the context (default 65536) and the port (default 8082) |
| `--data-dir DIR`, `--tools-dir DIR` | other places for Abyss-data and Abyss-tools |
| `--gpu N`, `--arch A` | the GPU, the CUDA architecture to compile for |
| `--nvcc PATH` | the CUDA compiler to build with (a CUDA 12 one for a V100) |
| `--gpus 0,1`, `--gpus all` | split the layers across several GPUs (see "Several GPUs"); builds for every GPU's architecture |
| `--download-model`, `--yes` | download without asking; the defaults for every question |
| `--rewrite-config` | write `abyss-<name>.json` again (otherwise an existing one is kept) |
| `--tools-only` | only create / update Abyss-tools |

The sections below are the same steps by hand.

## Build

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss

# the private Python environment
python3 -m venv .venv
.venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests cmake ninja pillow psutil

# llama.cpp's source at the commit the engine is pinned to (setup.py LLAMA_CPP_COMMIT): ggml, the MMQ kernels, gguf-py
C=3cf03257f219afbe7334045ff7c6a06ac68c627d
curl -L -o /tmp/llama.zip https://github.com/ggml-org/llama.cpp/archive/$C.zip
unzip -q /tmp/llama.zip -d third_party && mv third_party/llama.cpp-$C third_party/llama.cpp

# the engine: build/strata
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16
```

With ROCm's `hipcc` installed, the build also makes `build/libabyss_igpu.so` (the iGPU lane, below;
`-DSTRATA_IGPU_ARCH=gfx1151` is the default target, `-DSTRATA_IGPU_LANE=OFF` skips it).

The first build takes a few minutes; after a change only what changed is compiled again. Use `120a`, not `120`, on
RTX 50: the MXFP4 kernels use Blackwell's FP4 MMA instructions, which `120` cannot compile.

## Get the model

unsloth's [DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF), quant
`UD-Q2_K_XL` (3 files, 96.8 GB), with the Hugging Face CLI (`pip install huggingface_hub` gives `hf`):

```sh
M=~/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF   # Abyss-data beside Abyss (any NVMe folder works)
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf --local-dir $M
sha256sum $M/UD-Q2_K_XL/*.gguf
```

| File | SHA-256 |
| --- | --- |
| `-00001-of-00003.gguf` (5 MB) | `de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b` |
| `-00002-of-00003.gguf` (49.4 GB) | `c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327` |
| `-00003-of-00003.gguf` (47.4 GB) | `a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4` |

If a download stops and does not resume, `curl -L -C - -o <file> https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-Q2_K_XL/<file>`
continues it.

Other quants of the same repository should work if their expert types are among the decode kernels' (IQ1_S, IQ1_M,
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_K-Q6_K, MXFP4; IQ1_M has no prompt-path kernel, so such a quant reads
prompts token by token); only UD-Q2_K_XL is tested.

DeepSeek V4 GGUFs whose router and small projections are F16 rather than quantized (antirez's deepseek-v4-gguf, huihui-ai's
abliterated copies) also pack and run: `--compat-bf16` rounds those to BF16 for the engine's GEMVs. The huihui-ai Q2
file was checked against llama.cpp on the same file (KL 0.0075 over 3,000 tokens).

## Make the pack

The pack is the engine's index of the GGUF files: the small tensors (dense.bin), the expert table and the tokenizer
with the chat template. It goes next to the shards:

```sh
cd ~/llm/apps/Abyss
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL/pack-dsv4 --compat-bf16
```

`--compat-bf16` rounds the hyper-connection matrices (stored F32 in the GGUF) to BF16, the form the engine reads. A
few seconds; the experts stay in the GGUF files.

### A lighter dense pack (the attention projections at Q6_K or Q5_K)

Every decoded token reads all of the dense weights (6.6 GB in UD-Q2_K_XL), and 4.6 GB of that is three Q8_0 tensors a
layer: `attn_q_b`, `attn_output_a` and `attn_output_b`. `tools/requant_dense.py` re-quantizes just those (with
`ggml_quantize_chunk` from a llama.cpp build - llama-quantize refuses an i-quant model without an imatrix) and copies
every other byte of the shards as it is; `iq_pack.py` then packs the new files like the original ones:

```sh
cd ~/llm/apps/Abyss
GGUF_PY=~/llm/apps/llama.cpp/gguf-py LIBGGML=~/llm/apps/llama.cpp/build-cuda/bin/libggml-base.so \
  .venv/bin/python -I tools/requant_dense.py $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  $M/UD-Q2_K_XL-attn-q6k q6_k 'blk\.\d+\.attn_q_b\.weight' 'blk\.\d+\.attn_output_[ab]\.weight'
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL-attn-q6k/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL-attn-q6k/pack-dsv4 --compat-bf16
```

It rewrites the 90 GB of shards (10-25 minutes on an NVMe), so run it while the server is stopped; then point the
config's pack and tokenizer paths at the new directory. What the three types cost and give (2026-10-15, v1.3.18;
KL against the engine's own Q8_0 reference logits over the last 1,000 of 4,000 tokens, CPU lane, the same setup for
all three):

| attention projections | the three tensors | KL vs reference | max KL | same top token | nll | VRAM expert slots (RTX 5090) |
|---|---|---|---|---|---|---|
| Q8_0 (the GGUF as published) | 4.28 GiB | 0.0060 | 0.11 | 97.0% | 1.3071 | 3,010 |
| Q6_K | 3.31 GiB | 0.0082 | 0.31 | 95.8% | 1.3116 | 3,139 |
| Q5_K | 2.77 GiB | 0.0119 | 0.57 | 95.9% | 1.3135 | 3,225 |
| Q4_K | 2.27 GiB | 0.0246 | 0.84 | 92.9% | 1.3224 | 3,268 |

- Q6_K: decode +3.5% from the smaller GEMVs (v1.3.18's split kernel) plus 129 more VRAM slots. On the swe-bench L7
  set (16 tasks x 3, RTX 5090 + Radeon 8060S) it ran 93.9 against 91.3 tok/s with the same resolved rate (81.2%) and
  quality score (81.2 vs 81.0). The KL cost is the run-to-run spread's size.
- Q5_K: the KL doubles against Q8_0 (the largest single-token divergence 0.57). On an RTX 5090 it decodes 4% faster
  than Q6_K (1,024-token answers: a single token 12.13 -> 11.65 ms, the verify window 15.52 -> 14.86, 86-89 -> 93
  tok/s; the window and the token path share the saving), and more on a card with less memory bandwidth - a Radeon AI
  PRO R9700 at 640 GB/s reads 1.5 GB less a token. On the same swe-bench set it ran 99.0 tok/s with resolved 87.5%
  (14/16), success rate 83.3% and quality 82.0 - no loss visible for twice Q6_K's KL. **This is the pack the author
  runs, and the one the decode figures under "Memory and speed" are measured with** (v1.3.20).
- Q4_K: where it stops paying. The KL doubles again and the top token changes 3 points more often (Q8_0 -> Q5_K
  cost 1.1), for a single token only 2% faster than Q5_K (11.67 -> 11.40 ms) and 43 more slots. Not run on swe-bench.
- Not worth touching: `attn_q_a`, the kv / compressor / indexer projections and the norms are small, and the shared
  experts, the embedding and the output head are already Q5_K / Q6_K / Q4_K.

## Configure and start

A config file (any name; `strata-dsv4.json` here) in the Abyss folder:

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

| Field | |
| --- | --- |
| `exe` | the engine (`build/strata`) |
| `args` | the engine's arguments: `--glm-pack` the pack (the engine runs DeepSeek V4 through the same entry as GLM), `--max-context` the context in tokens, `--prefill auto` the prompt chunk size |
| `tokenizer` | the pack's `tokenizer` folder (its chat template is DeepSeek's own) |
| `model_name` | the name the API reports |
| `sampling` | defaults for requests that do not set them (DeepSeek's recommendation: temperature 1.0, top_p 0.95) |
| `reasoning_effort` | optional default thinking level (`none`, `low`, `high`, `max`) |
| `thinking_budget` | optional cap on a thinking block in tokens (default 32768; 0 = none) |
| `env` | optional `{"STRATA_GLM_...": "..."}` settings for the engine (see [Settings](#settings)) |
| `log` | the engine's log |

Start the server:

```sh
.venv/bin/python serve/server.py --engine strata --config strata-dsv4.json --port 8082
```

It prints `ready: http://127.0.0.1:8082/v1` after loading. A start takes about a minute: the dense weights go to the
GPU, then every expert is read into the VRAM and RAM tiers (about 20 s at 4 GB/s from an NVMe). Ctrl+C stops it.

Server options: `--host 0.0.0.0` to accept other machines (always with `--api-key <secret>`), `--open` opens the page
in the browser, `--gpu 0` picks the GPU.

## Using it

- **Browser:** `http://127.0.0.1:8082/` - chat, a live monitor of the expert tiers and the GPU, settings.
- **OpenAI-compatible API:** base URL `http://127.0.0.1:8082/v1`, any model name.

  ```sh
  curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
  ```

  The answer is in `content`, the thinking in `reasoning_content`.
- **Anthropic-compatible API:** `http://127.0.0.1:8082/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8082`).
- **Thinking:** DeepSeek V4 thinks unless asked not to. `reasoning_effort` (or the chat's setting):

  | Level | What DeepSeek gets |
  | --- | --- |
  | not set | thinking on |
  | `none` | thinking off (the prompt ends with `</think>`) |
  | `low` | thinking on |
  | `high` | thinking on + DeepSeek's "Reasoning Effort: Absolute maximum" instruction |
  | `max` | thinking on + its "Beyond maximum" instruction |

  The server always leaves the answer at least 1024 tokens (or a quarter of `max_tokens`): with a small `max_tokens`
  (a few hundred) the thinking is closed almost at once. Give thinking requests a `max_tokens` of a few thousand.
- **Tool calls:** OpenAI `tools` / Anthropic `tools` work, including several calls in one turn. The model writes them
  in DeepSeek's DSML form; the server turns them into `tool_calls` and renders past calls and `tool` results back into
  the prompt.
- **Conversation reuse:** a request that continues the previous one (the same messages plus new ones) reads only the
  new tokens.
- **One request at a time;** others wait in line.

### With Strata-tools' strata.sh

If you use the `strata.sh` helper (start / stop / log), put a run script and the config into the Abyss folder under
its naming (`run-<name>.sh`, `strata-<name>.json`) and point it there: `STRATA_DIR=~/llm/apps/Abyss ./strata.sh start <name>`.
The run script is the start command above (`exec .venv/bin/python serve/server.py --engine strata "--config" "<config>" "--port" "8082" "$@"`).

## Several GPUs

With two or more NVIDIA GPUs the layers can be split across them (a layer split). Each GPU keeps only its own layers'
experts in VRAM, so the second card's VRAM holds more of them.

- **Setup:** `./setup-deepseek.sh --gpus 0,1` or `--gpus all`. The config gets `"gpu": [0, 1]`, and the server
  starts the engine with `--layer-split auto`. GPUs of different generations (an RTX 5090 with an RTX 3090) are
  built for both architectures (`86;120a`).
- **What `auto` picks (v1.3.25):** it prices every boundary - and each card by itself - by the decode time it
  predicts (each card's bytes over its memory bandwidth, the misses the CPU / iGPU lane does not hide, a fixed cost
  a boundary; fitted on an RTX 5090 + RTX 3090, within 0.35 ms of the measured 10.9 - 18.1 ms), and runs on one GPU
  when that is faster: a decode token goes through the parts one after another, so a slower second card's layers can
  cost more than its VRAM saves. RTX 5090 + RTX 3090 (both PCIe Gen4 x4): the 5090 alone, 94-95 tok/s (the split it
  picked before, 26 / 17: 84 tok/s; the best split, 33 / 10: 88 tok/s). `STRATA_GLM_SPLIT_KEEP=1` splits anyway
  (more context or VRAM for prompts), `STRATA_GLM_SPLIT_LOG=1` prints every prediction.
- **Choosing the boundary:** `"layer_split": "22"` in the config (the second card's first layer).
- **The iGPU lane:** it works with a split too. One iGPU computes every GPU's RAM tier, one call at a time.
- **What has been checked:** a two-part split on one GPU (`STRATA_GLM_SPLIT_SAME=1 STRATA_GLM_SPLIT=22
  STRATA_GLM_DEVS=0,0`, for testing) matches the unsplit model (KL within the run-to-run spread), and an RTX 5090
  (PCIe Gen4 x4) + RTX 3090 (a USB4 enclosure, ~3.8 GB/s) split 27 / 16 ran at 8K / 32K prompts of 1,506 / 1,985
  tok/s and decode of 54.9 / 56.7 tok/s with the iGPU lane (one RTX 5090 + iGPU: 1,683 / 2,876 and 58.5 / 65.9).
  The second card raised the VRAM hit rate (82 -> 91%) but the hop between the cards and the slow link cost more.
  A fast second card on a real PCIe slot has not been measured.
- **Speculative decoding** (v1.3.26): the verify window runs part after part, each part's columns crossing to the
  next as a token's residual does. RTX 5090 + RTX 3090 split 33 / 10: 87-90 tok/s (without it ~80-88), the drafts
  accepted as often as on one card (57% on code).
- **Prestaging** (v1.3.27): each part copies its next layer's predicted RAM-tier experts during the mixer, as one
  GPU does; on a split the chunk is sized first (8192 at most) and the prestage buffer takes what it leaves. RTX 5090 +
  RTX 3090 split 33 / 10: 32K-token prompts 2,177-2,186 -> 2,274-2,350 tok/s, 8K about even. `STRATA_GLM_PRESTAGE=0`
  turns it off.
- AMD GPUs: see "AMD GPUs" below.
- **V100 (sm_70):** CUDA 13 cannot compile for the V100, so install CUDA 12 (12.9) and build with
  `./setup-deepseek.sh --nvcc /usr/local/cuda-12.9/bin/nvcc` (CUDA 12.9 also builds an RTX 50's `120a`). The sm_70
  build has been checked to compile; it has not run on a V100 yet. The newer tensor-core kernels (the indexer and the
  prompt attention) need sm_80 or newer, so a V100 uses the older ones.

## AMD GPUs (HIP, experimental)

DeepSeek V4 also runs on AMD Radeon cards: the RX 7900 XT / XTX (gfx1100), the RX 9070 / AI PRO R9700 (gfx1201) and
the Ryzen AI Max's Radeon 8060S (gfx1151).

- **Setup:** `./setup-deepseek.sh --backend hip` (chosen by itself when there is no NVIDIA GPU). On Linux it uses ROCm 7
  or newer in `/opt/rocm` (ROCm 10: `/opt/rocm/core-10.0` too). The engine is built in `build-hip/strata`, and the config gets `"backend": "hip"`.
- **Driver and ROCm:** [docs/amd/README.md](amd/README.md) has the install steps (amdgpu, ROCm 10's per-architecture
  packages - an R9700 needs `amdrocm10.0-gfx1201` - and why `HSA_OVERRIDE_GFX_VERSION` must not be set), a check
  list, and troubleshooting.
- **What has been checked:** the Radeon 8060S (gfx1151) alone, and an AI PRO R9700 (gfx1201) with the 8060S as its
  iGPU lane: 36-42 tok/s on DeepSeek V4, KL against the CUDA build within the run-to-run spread (0.005).
- **A discrete Radeon on a Ryzen AI Max:** the iGPU computes the RAM-tier experts as it does beside an NVIDIA card
  (+12-46% decode on the R9700); the config's `"gpu_pci"` keeps the right card when HIP numbers the GPUs otherwise.
- **What is faster:** the indexer's scores and the prompt attention use RDNA's WMMA (gfx11 / gfx12).
  `STRATA_DS4_IDX=1` keeps the older indexer kernels, and `STRATA_DS4_ATTN=1` the older attention.
- **Limits:** mixing it with NVIDIA in one process (CUDA and HIP together) is not supported.

## DeepSeek V4.1 Flash

From v1.3.29 the same engine runs DeepSeek-V4.1-Flash-1017. It is set up by hand (the one-command setup installs V4
only); [docs/deepseek41/README.md](deepseek41/README.md) has the GGUFs that load, the pack command, a config and the
measurements, and [docs/deepseek41/SPEC.md](deepseek41/SPEC.md) what the model changes against V4. In short:

- **Disk and memory.** nazeshinjite's Q2 GGUF is 366 GB: 153 GB of experts (384 a layer, 9.95 MB each), ~200 GB of
  Engram tables and the dense weights. On an RTX 5090 + 94 GB RAM ~57% of the experts are resident (VRAM ~2,300, a
  60 GB RAM tier ~6,500); the rest and every Engram row (48 a token, 264 bytes each) are read from the GGUF. Keep it
  on the fastest NVMe you have, and leave ~25 GB of RAM to the page cache (`STRATA_GLM_RAM_GB=60` on a 94 GB box).
- **Speed is the SSD's.** A token that reads no expert from the SSD takes 13-15 ms; each read adds ~3-4 ms at
  3.4 GB/s. With `STRATA_GLM_TAIL_SKIP=3 STRATA_GLM_TAIL_SKIP_DISK=1` (a route's 4th-6th experts left out when they
  are only on disk; KL 0.229 -> 0.240 against the unquantized model) decode is ~28 tok/s, 17.7 without; swe-bench
  coding tasks ran at 51.6 tok/s end to end. Prompts are slow (the attention runs a token at a time and the prompt's
  experts come from the SSD).
- **A lighter dense pack** (v1.3.36, optional): `attn_q_b` / `attn_output_a` / `attn_output_b` at Q5_K instead of
  Q8_0 (dense 8.14 -> 6.50 GB, 177 more VRAM expert slots) decodes 10-15% faster for KL 0.242 -> 0.251; see
  [docs/deepseek41/README.md](deepseek41/README.md#a-lighter-dense-pack-q5_k).
- **Start it in a memory-capped scope** (`systemd-run --user --scope -p MemoryMax=88G ...`): the page cache and the
  pinned tier together can push systemd-oomd into killing the desktop session.
- **Not yet:** speculative decoding, a layer split, conversation slots on disk, images.

## EXL3 quants (GLM-5.3-Flash)

From v1.3.35 the engine also reads [exllamav3](https://github.com/turboderp-org/exllamav3)'s EXL3 quants of
GLM-5.3-Flash, in place from their safetensors; [docs/exl3/README.md](exl3/README.md) has the details. In short:

- **Installing.** `./abyss.sh --setup` lists turboderp's `EXL3-sc_2.10bpw` (88.3 GB) and `EXL3-3.05bpw` (125.3 GB)
  beside Maya's GGUFs and says what each suits on this PC: recommended when RTX 30+ cards hold 80% of it, a quality
  option with an APU iGPU or an AVX-512 CPU for the RAM tier, not for a CPU with AVX2 only. `--model EXL3-sc_2.10bpw`
  skips the question. NVIDIA GPUs only in the setup.
- **Switching.** Each quant gets its own config and start script (`maya-exl3-sc_2.10bpw.json`,
  `run-maya-exl3-sc_2.10bpw.sh`); a GGUF install and an EXL3 install sit side by side. The engine knows an EXL3 pack by
  its `pack/exl3.txt` - no setting.
- **By hand.** `.venv/bin/python tools/exl3_pack.py --model <dir>` (~5 s) writes `<dir>/maya-exl3.gguf` and
  `<dir>/pack/`; start with `--glm-pack <dir>/pack`.
- **What to expect** (RTX 5090 + Radeon 8060S lane + 94 GB RAM, 1,000 scored tokens): sc_2.10bpw nll 1.591-1.598 and
  top-1 63.3-63.7% against Maya-S-v2-IQ2_XXS's 1.634 / 62.3%, at 18.8 ms a token against 15.4 (v1.3.36: the iGPU's
  trellis decode by `v_sad_u8` and `v_dot2_f32_f16`; 22.4 in v1.3.35).

## Memory and speed

The engine fills the GPU's free VRAM with experts (after the dense weights, 6.2 GB for UD-Q2_K_XL, the caches and the
prompt buffers), sizes a pinned RAM tier from the free RAM, and reads the rest from the SSD when needed. It records
which experts you use (`<pack>/expert_usage.txt`) and loads those first next time.

Measured on one RTX 5090 + Ryzen AI Max+ 395 with 94 GB RAM (UD-Q2_K_XL, context 65536, every expert in VRAM 23 GB +
RAM 70 GB; the iGPU lane on):

| | |
| --- | --- |
| Decode, 1,024-token answers (prose / code, greedy) | ~92-95 tok/s (CPU lane instead of the iGPU: ~55-60) |
| Decode, 16 swe-bench coding tasks x 3 (sampled, end to end) | ~100 tok/s (10.1 ms a token averaged over the run; 1,024-token answers: a single token 11.1 ms, the 2-token verify window 14.6) |
| Decode, 60 swe-bench coding tasks x 5 (sampled, end to end) | ~107 tok/s (100.5-114.5 a task; v1.3.27) |
| Decode at a 125K-token context | ~65 tok/s |
| Prompt, 8K / 32K / 125K tokens (server, second run) | ~1,630-1,760 / ~3,100-3,270 / ~2,800 tok/s |
| Start-up (weights + expert warm-up) | ~1 min |
| Attention caches at 65536 tokens | 0.44 GB |

(Decode rows: Abyss v1.3.21 with the Q5_K attention pack ("A lighter dense pack", above; the GGUF as published:
~82-84 and ~91-94). Prompt and 125K rows: v1.3.16-1.3.18, the GGUF as published. On the same machine and file
llama.cpp `bed0a8566` with `-ngl 99 -ncmoe 32` decodes at 28-30 tok/s
and reads prompts at 78 tok/s with ubatch 512, 660 with ubatch 8192.) The numbers of every release, and how each was
measured, are in [CHANGELOG.md](../CHANGELOG.md).

The first prompts after a start, or with less memory than the model, read experts from the SSD and are slower. The
CPU computes part of the RAM-tier experts while the GPU copies others over PCIe; the split is measured at start.

**The iGPU lane.** On an AMD APU (Ryzen AI Max+ 395: Radeon 8060S) the integrated GPU reads the same RAM as the CPU,
at ~240 GB/s against the CPU threads' ~75. With ROCm installed (the build then makes `libabyss_igpu.so`), the engine
registers its RAM tier with the iGPU at start and lets it compute the RAM-tier experts instead of the CPU threads -
the experts are not copied. The start-up log says which lane it chose and why (`iGPU lane: an expert 0.046 ms vs 0.107
on the CPU's 16 threads`); it keeps the CPU when that is faster, and when the library or a HIP device is missing.
The prompt path uses the lane too: the RAM-tier experts a prompt chunk needs are computed on the iGPU instead of the
CPU threads (an 8K prompt +40%). A lane call has a deadline (`STRATA_GLM_IGPU_TIMEOUT_MS`, 20 ms): when the kernel
driver holds the iGPU's queue (it happens for a few minutes after a start, while the kernel reclaims memory after the
warm-up), the CPU threads take over until the late call finishes, and the start-up timing is repeated until it is
clean.

**Speculative decoding (`STRATA_DS4_SPEC`, on by default).** The GGUF has no draft model, but the answer often repeats
what is in the prompt or earlier in the answer (code, file paths, tool calls). The engine looks up the last 3 tokens'
earlier occurrence, checks the token that followed it together with the next one in one pass, and keeps it when the
model agrees. The output is the same as without it (the same sampler on the same logits); only the time per token
changes. A gate learns each request's acceptance and skips the draft while it would not pay. Coding prompts: +10-13%
(77-85% of the drafts accepted); prose: about even (~50% accepted). `STRATA_DS4_SPEC=0` turns it off; a layer split
across GPUs runs it too (v1.3.26).

## Settings

Environment variables for the engine, in the config's `"env"` block (all of Maya's are listed in
[docs/maya/README.md](maya/README.md#tuning); these matter most here):

| Setting | Default | |
| --- | --- | --- |
| `STRATA_GLM_WARM` | on | `0`: no warm-up at start (the tiers fill as experts are used; the first answers are slower) |
| `STRATA_GLM_RAM_GB` | from free RAM | a fixed size of the pinned RAM tier in GB |
| `STRATA_GLM_RAM_HEADROOM_GB` | 6 | RAM left free for the system when the RAM tier is sized |
| `STRATA_GLM_STREAM_FETCH` | on | an expert read from the SSD is copied to VRAM chunk by chunk as the reads land, the PCIe copy running behind the SSD (v1.3.31; DeepSeek V4.1 -9% a token); `0`: the whole expert after the read |
| `STRATA_GLM_RAM_THP` | on (Linux) | the RAM tier in transparent huge pages, as far as the free 2 MB blocks go (an APU lane re-validates 512x fewer pages: no multi-second lane stalls); `0`: the driver's pinned pages |
| `STRATA_GLM_RESERVE_MB` | 700 | VRAM left free beside the expert pool |
| `STRATA_GLM_CPU_LANE` | one thread per physical core | CPU threads computing RAM-tier experts; `0` = off |
| `STRATA_GLM_X3_SPLIT` | off | an EXL3 pack: the CPU threads compute this fraction of each route's RAM-tier experts while the iGPU computes the rest (`auto`: by their timings); slower on a Ryzen AI Max+ 395 (shared power and heat), see [EXL3](exl3/README.md#settings) |
| `STRATA_GLM_EXL3_DECODE` | by GPU | an EXL3 pack: `new` (tensor-core tiles, RTX 30+) or `legacy` (CUDA cores) on any GPU |
| `STRATA_GLM_CPU_PIN` | off | `<CPU list>` (e.g. `0-15`, `0,2,4-7`): pin the CPU lane's workers to those CPUs and cap their count at the list's size, one GPU or a split, Linux and Windows (v1.3.29); `1` / `0`: a split's per-part pools pinned or not, as before |
| `STRATA_GLM_PCIE_SHARE` | measured at start | the share of RAM-tier experts copied over PCIe instead of computed on the CPU (`0` = all on the CPU, `1` = none) |
| `STRATA_GLM_PREFILL` / `--prefill` | `auto` | the prompt chunk: `auto`, a number of tokens, or `0` (token by token) |
| `STRATA_GLM_USAGE` | `<pack>/expert_usage.txt` | your expert usage between starts; `0` = off |
| `STRATA_GLM_NO_REUSE` | off | `1`: no conversation reuse |
| `STRATA_GLM_THINK_RESERVE` | 1024 | (server) tokens a thinking block always leaves for the answer |
| `STRATA_GLM_IGPU` | auto | the iGPU lane: on when `libabyss_igpu.so` and a HIP device are there and it is faster than the CPU; `0` off, `1` on even when slower. `STRATA_GLM_IGPU_LIB=<path>`: another library |
| `STRATA_GLM_IGPU_TIMEOUT_MS` | 20 | a lane call later than this goes to the CPU threads (0: wait) |
| `STRATA_GLM_IGPU_EARLY` | on | a route that reads experts from the SSD starts its iGPU lane before the reads instead of after them (v1.3.36; DeepSeek V4.1 -2%, same logits); `0`: after |
| `STRATA_GLM_IGPU_XQ` | on | the lane takes the token's input already quantized (q8_1) by the GPU from the request instead of quantizing it again (v1.3.36, bit-identical); `0`: quantized on the iGPU |
| `STRATA_GLM_PREFILL_IGPU` | on | the prompt's RAM-tier experts on the iGPU; `0`: the CPU threads |
| `STRATA_GLM_IGPU_FUSED` | on | the prompt's iGPU experts through the fused int8 tensor-core pass (gfx1151, IQ2_XS / IQ3_XXS experts at 4096 / 2048; v1.3.18); `0`: the one-row kernels. `ABYSS_IGPU_FUSED_CHECK=<n>` compares the first n calls against them (stderr) |
| `STRATA_GLM_MV_SPLIT` | auto | dense GEMVs with a row over 4 warps: unset = under 256 blocks, and Q6_K up to 4,096 blocks (v1.3.18); `0` one warp a row always, `2` / `4` that split always. Not bit-identical to one warp a row |
| `STRATA_GLM_PROMOTE_RATIO` / `_MARGIN` | 4 / 16 | a RAM-tier expert replaces a resident one when its use count passes ratio x the resident's + margin (2 / 4: the old, churnier rule) |
| `STRATA_GLM_BG_PACE_KB` | 1024 | the tier moves' copies (the refill after a prompt, the swaps) go in pieces of up to this many KB a MoE layer, while the next layer's mixer runs, instead of all at the token boundary - there they held up the next token's host-mapped reads until they had landed (after an 8K prompt ~10 tokens at 35-39 ms). `0`: at the boundary, as before |
| `STRATA_GLM_ZERO_COPY` | on | the token boundary without the copy engines (v1.3.21): the table edits, the embedding row (a pinned copy of `token_embd`, Q8_0 / Q5_K) and the token are read and written host-mapped by the kernels. `0`: the copies, as before |
| `STRATA_GLM_GAP_PROF` | off | `1`: a `glm gap prof` line after each answer - where the time between two tokens goes (host laps, the GPU's idle gap by timing events) |
| `STRATA_GLM_TAIL_SKIP` / `_W` | off | a route's experts past rank r (or below a probability share) that are not in VRAM are left out - faster on a slow CPU lane, changes the answers (KL ~0.01-0.02) |
| `STRATA_GLM_ROUTE_AFFINITY` | off | `<a>`: a VRAM-resident expert's routing score gets +a (its weight stays the router's), so a near-tie goes to the expert already on the GPU (after Maya #252; v1.3.36). Changes the answers. GLM-5.3-Flash Maya-S-v2 on an RTX 5090: `0.04` -> 14.1 ms a token against 15.4 (-9%), VRAM hits 56 -> 69%, nll 1.631 against 1.635 (KL 0.043 to the default's logits); `0.08` -13% but nll starts to rise (1.641-1.651). Not on hash-routed layers |
| `STRATA_GLM_ROUTE_AFFINITY_RAM` | off | `<a>`: the same for experts in the RAM tier, a VRAM one getting the larger of the two (with an SSD tier: avoids experts that only the SSD has; v1.3.37 measured). DeepSeek V4.1 with the tail skip, 8 held-out sequences: `0.08` -8-12% a token (SSD reads -22%) for KL 0.2406 -> 0.2438 and ppl 9.76 -> 9.81; `0.16` -12-20% for KL 0.2501 / ppl 9.89 |
| `STRATA_GLM_ROUTE_PROF` | off | `1`: the route kernel prints its SM cycles per phase every 43,000 routes (diagnostics, NVIDIA) |
| `STRATA_GLM_TAIL_SKIP_DISK` | off | `1`: the tail skip leaves out only experts that are on the SSD alone (v1.3.29). With `STRATA_GLM_TAIL_SKIP=3` the DeepSeek V4.1 speed setting: 56.6 -> 34.2 ms a token, KL 0.229 -> 0.240; changes nothing when every expert is in VRAM + RAM |
| `STRATA_GLM_DEFER` | off | expert deferral: the CPU lane's part of a layer's MoE is not waited for at the layer's end but written into the residual after the next layer's attention (with its own layer's weights) - the GPU never idles for the lane: single tokens 10.9 -> 9.5 ms, the 2-token verify window 14.4 -> 13.1, +12-15% tok/s on an RTX 5090 + Radeon 8060S; changes the answers more than the tail skip (KL 0.08 against the Q8_0 reference, 0.012 without; swe-bench L7 16 tasks x 3: 107.2 tok/s and resolved 13 of 16 against 100.9 and 14 with it off). With a layer split each part defers within its own layers (its last layer waits for its own part) |
| `STRATA_DS4_IDX`, `STRATA_DS4_ATTN` | 0 | `1`: the older FP32 kernels for the lightning indexer / the prompt attention instead of the tensor-core ones (A/B checks) |
| `STRATA_GLM_MMQ_GLU` | on | the prompt path's expert gate/up product writes the SwiGLU itself (NVIDIA Turing and newer); `0`: the separate SwiGLU kernel, bit-identical to older builds |
| `STRATA_GLM_CPU_MT` | on | CPU experts shared by several tokens (prompts, verify windows) decode each weight row once for all of them, bit-identical to ggml's dot; `0` off |
| `STRATA_GLM_CPU_1T` | on | cheaper one-token IQ3_S / IQ3_XXS rows on the CPU lane, bit-identical to ggml's dot; `0` off |
| `STRATA_GLM_VRAM_EVICT` | counts | `lru`: the VRAM tier's refill evicts the least recently used expert (for cache studies) |
| `STRATA_GLM_DISK_READERS` | off | prompts read SSD-held experts with this many threads (try it on a DRAM-less NVMe; DeepSeek V4.1: `8`, a 1K-token prompt 33 -> 27 s) |
| `STRATA_GLM_DISK_LANE` | off | `1`: an expert read from the SSD into RAM is computed there by the CPU / iGPU lane instead of being copied to VRAM (v1.3.29; -5% on V4.1 with Q2_K/Q3_K experts, nothing with the smaller Q2 ones) |
| `STRATA_DS41_ENGRAM_CACHE_MB` / `_THREADS` | 256 / 15 | DeepSeek V4.1: the Engram raw-row cache and the row readers (v1.3.29) |
| `STRATA_DS41_ENGRAM_ASYNC` | on | DeepSeek V4.1: a token's 48 Engram rows are read by a background thread while layer 0 runs, layers 1 / 14 wait for them on the GPU (v1.3.36; -0.5 ms a token, same logits); `0`: read before the token |
| `STRATA_DS41_ENGRAM_PREFETCH` | 8 | DeepSeek V4.1: threads reading a prompt chunk's Engram rows into the row cache right after its hash (v1.3.36; a 1K-token prompt's Engram 1.7-1.9 -> 0.6 s, the prompt -6-7%); `0` off |
| `STRATA_DS4_SPEC` | 1 (on) | speculative decoding from the prompt and the answer: draft tokens a step (1-3; 1 is the fastest measured); `0` off |
| `STRATA_DS4_SPEC_GATE` | on | drop a draft while the request's acceptance does not pay for the verify window; `0`: always draft |
| `STRATA_DS4_SPEC_MIN` | 3 | the tokens a draft's lookup matches (2-4) |

`--max-context` sets the context. DeepSeek V4 is trained for 1M tokens; 65536 is the default and 131072 (a 125K
prompt) has been measured. The attention caches are small (0.44 GB at 65536), but the prompt path's buffers grow with
the context.

## Checking the output against llama.cpp

`tools/dsv4_logits_dump.cpp` writes llama.cpp's logits for a token sequence (build it against a llama.cpp build that
supports `deepseek4`); the engine's `STRATA_GLM_SCORE` mode compares its own logits with them. See
[docs/deepseek4/PORT.md](deepseek4/PORT.md) for the commands and the results: on UD-Q2_K_XL the two engines agree to
KL 0.009 over 3000 tokens, with the same next-token accuracy.

## Something went wrong?

- **The build stops at `mma with block scale not supported on .target sm_120`** - configure with
  `-DCMAKE_CUDA_ARCHITECTURES=120a`.
- **`the engine exited before it was ready`** - read the engine log (`log` in the config). Out of memory while pinning
  the RAM tier: lower `STRATA_GLM_RAM_GB` or raise `STRATA_GLM_RAM_HEADROOM_GB`.
- **`refers to missing files` / `pack:` errors** - the pack must sit next to the three shards it was made from; make it
  again after moving them.
- **The thinking stops after a word and the answer starts** - `max_tokens` is too small (see Thinking above).
- **Slow, the SSD busy** - not enough RAM + VRAM for the experts; close programs or add RAM.
- **Port in use** - another server runs on it; use `--port`.

## Limitations

- No draft model: this GGUF has no MTP block, and DSpark (DeepSeek's drafter) is not supported - speculative decoding
  drafts from the text itself (`STRATA_DS4_SPEC`).
- Conversation slots on disk (`STRATA_GLM_SLOTS`) are off for DeepSeek V4; the live reuse of the last conversation works.
- No images (DeepSeek V4 Flash is text-only here).
- Several GPUs: the layer split runs (see "Several GPUs"), and a slower second card
  does not beat one fast card with the iGPU lane - `auto` then runs on the fast card alone. Windows is untested for DeepSeek V4.
- `./abyss.sh` (Maya's installer) is for GLM; DeepSeek V4 is set up with `./setup-deepseek.sh`; DeepSeek V4.1 by hand
  ([docs/deepseek41/README.md](deepseek41/README.md)).
- DeepSeek V4.1: no speculative decoding (no GGUF carries its MTP layers), no layer split, no conversation slots on
  disk, the prompt's attention a token at a time. Up to v1.3.31 it crashed on the 16,385th token of a context (and
  chose its attention rows wrongly past 8K); use v1.3.32 or later for long chats.
