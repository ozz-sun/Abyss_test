# DeepSeek V4 Flash in Abyss (the `deepseek4` port)

Abyss is a copy of Maya (project-maya, the GLM-5.3-Flash fork of Strata). DeepSeek V4 Flash
(llama.cpp arch `deepseek4`) runs on the `Glm5Model` fast path as a second architecture: the mHC residual, the
router + expert tiers (VRAM / pinned RAM / disk, CPU lane), the shared expert, the server and the prompt path are
Maya's; the attention mixer, the routing function, the head and the packaging are new. The math is in
[SPEC.md](SPEC.md) (transcribed from llama.cpp `src/models/deepseek4.cpp`).

DeepSeek V4.1 Flash (`deepseek41`, v1.3.29) is a third architecture on the same path; what it changes and where is in
[../deepseek41/SPEC.md](../deepseek41/SPEC.md), how to run it in [../deepseek41/README.md](../deepseek41/README.md).

## What is new

| Piece | Where |
| --- | --- |
| Geometry from `deepseek4.*` metadata (`Glm5Geometry::arch == 1`, `from_gguf_dsv4`) | `src/core/glm_model.cu`, `include/strata/core/glm_model.hpp` |
| Token path attention: q_a/kv/compressor projections, norms, partial RoPE (YaRN for compressed layers), the 128-token window ring, CSA (ratio 4, overlapping windows) / HCA (ratio 128) compressors, the lightning indexer top-512, attention with sinks, de-rope, grouped output LoRA | `src/kernels/cuda/dsv4_fast.cu`, `src/core/dsv4_fast_path.cu` |
| The gated hyper-connection head (`output_hc_*`) | `dsv4_fast.cu` `head_prep` |
| Router: sqrt(softplus) scoring, hash-routed layers 0-2 (`ffn_gate_tid2eid`) | `glm_fast.cu` `moe_route`, `glm_batch.cu` `route` |
| MXFP4 expert rows (2 layers of unsloth UD-Q2_K_XL): decode dot product; prompt-path MMQ with llama.cpp's FP4 activations (needs `CMAKE_CUDA_ARCHITECTURES=120a`) | `glm_fast.cu` `dw<39>`, `src/prefill/moe_mmq.cu` |
| Prompt path (batched): window keys in a linear buffer, batched compressors, the indexer's scores and the attention on tensor cores (`mma.sync`, Q and P as FP16 hi + lo pairs, ~FP32 accuracy; sm_80+), with the FP32 kernels as the fallback (`STRATA_DS4_IDX=1`, `STRATA_DS4_ATTN=1`) | `dsv4_fast.cu` (`*_batch`, `*_mma_kernel`), `glm_prefill.cu` `prefill_ds4_mixer` |
| Token path on tensor cores: the lightning indexer's scores (sm_80+); the exact top-k select (radix passes, block scan); the score + select launches skipped while every row is visible anyway (positions below 2,047) | `dsv4_fast.cu`, `src/kernels/cuda/dsa_topk.cuh` |
| Speculative decoding from the text itself (n-gram lookup, a verify window of k + 1 staggered streams, an acceptance gate; on by default) | `src/core/dsv4_verify.cu`, `STRATA_DS4_SPEC*` |
| The iGPU lane: an AMD APU's integrated GPU computes the RAM-tier experts in place (decode and prompt) through a dlopen'd HIP library; a call deadline hands late work to the CPU threads | `src/igpu/igpu_lane.cpp`, `libabyss_igpu.so` |
| AMD (HIP): the build links (MXFP4 in the HIP MMQ instances), the indexer and the prompt attention on RDNA WMMA (gfx11 / gfx12) | `dsv4_fast.cu` (`*_wmma_kernel`), `setup-deepseek.sh --backend hip` |
| Layer split across NVIDIA GPUs: hash-routed layers get their token ids in every part, one iGPU lane shared by the parts | `glm_prefill.cu`, `igpu_lane.cpp`, `setup-deepseek.sh --gpus` |
| V100 (sm_70) builds with CUDA 12: the floor is compute capability 7.0, BF16 projections as FP32 GEMMs | `CMakeLists.txt`, `setup-deepseek.sh --nvcc` |
| Conversation snapshot (window + compressor rings; the compressed caches are append-only by position) | `dsv4_fast_path.cu` `dsv4_snap_copy`. File slots: not implemented (off) |
| Packer: `deepseek4` arch, I32 hash tables, hc fn matrices rounded to BF16 (`--compat-bf16`) | `tools/iq_pack.py`, `tools/gguf_reader.py` |
| Tokenizer: the `joyai-llm` pre-tokenizer (three regexes applied in sequence) | `tools/strata_tokenizer.py` |
| Server: DeepSeek's chat template (thinking / reasoning_effort, `bos_token`, `from_json`), DSML tool calls, `<｜end▁of▁sentence｜>` stop | `serve/frontend.py`, `serve/server.py` |

The Hadamard rotations llama.cpp applies to quantized caches and to the indexer keys are orthonormal on both sides of
every dot product and are omitted (caches are FP16, llama.cpp's default).

## Validation (unsloth `DeepSeek-V4-Flash-0731` UD-Q2_K_XL, RTX 5090 32 GB, 94 GB RAM)

Reference: llama.cpp `bed0a8566` on the same GGUF (`tools/dsv4_logits_dump.cpp`, batch mode on the GPU).

| Test | Result |
| --- | --- |
| 10-token chat prompt, token path | KL mean 0.008 (max 0.017), top-1 equal 9/9 |
| 3000-token document, token path | KL mean 0.0093, top-1 equal 94.5%, ppl 6.36; next-token top-1 60.35% (llama.cpp: 60.35%). No drift by position: [0,128) 0.015, [128,512) 0.012, [512,1024) 0.011, [1024,2048) 0.0095, [2048,3000) 0.0065 |
| 2499 tokens through the prompt path, then 500 by the token path | KL mean 0.0059, top-1 equal 95.2% |
| Tokenizer vs llama-tokenize | equal on a 5578-token document and the chat prompt |

The window ring wrap (p >= 128), HCA compression (every 128) and the indexer's top-512 selection (p >= 2048) are all
inside the 3000-token test.

## Speed (same machine; all 11008 experts fit VRAM 23 GB + pinned RAM 70 GB, warm-up 21 s)

RTX 5090 on PCIe Gen4 x4 + Ryzen AI Max+ 395 (the Radeon 8060S as the iGPU lane). Decode rows: Abyss v1.3.21-1.3.27
with the Q5_K attention pack; prompt and 125K rows: v1.3.16-1.3.18 with the GGUF as published. llama.cpp
`bed0a8566` on the same file: its best settings (`-ngl 99 -ncmoe 32 -lm none`), and with every expert on the CPU
(`-ot exps=CPU`, the v1.0 comparison).

| | Abyss | llama.cpp (`-ncmoe 32`) | llama.cpp (`-ot exps=CPU`) |
| --- | :---: | :---: | :---: |
| Decode, 1,024-token answers (prose / code, greedy) | ~92-95 tok/s (v1.3.16: 82-84, v1.2: 73, v1.1: 56-57, v1.0: 48) | 28-30 tok/s | 5-7 tok/s |
| Decode, swe-bench coding tasks (sampled, end to end) | ~107 tok/s over 60 tasks x 5, ~100 over 16 x 3 (v1.3.16: ~94 over 13 prompts; v1.2: 66, 78 with spec; v1.1: 51) | - | 5-7 tok/s |
| Decode at a 125K-token context | ~65 tok/s | 28-30 to 32K | - |
| Prompt, 8K tokens (server, second run) | ~1,630-1,760 tok/s (v1.1: ~1,000-1,090 at 5.6K; v1.0: ~690) | 78 (ubatch 512) - 660 (ubatch 8192) | 3-6 tok/s |
| Prompt, 32K / 125K tokens | ~3,100-3,270 / ~2,800 tok/s (v1.3.1: 1,720 / 1,010) | - | - |

How it got there (each step's measurement and its KL check are in [CHANGELOG.md](../../CHANGELOG.md)):

- **v1.1.0:** the token-path attention split over key chunks (flash-decoding), a lighter route kernel and CPU-answer
  read, the prompt path's CPU rows by the AVX-512 i-quant kernels, an adaptive prestage. 3000 tokens KL 0.0091 (CPU
  lane on), 2499 + 500 KL 0.0054.
- **v1.2.0:** the RAM-tier experts on the APU's integrated GPU (the iGPU lane, gfx1151 through ROCm, `src/igpu/`)
  instead of its CPU threads - 0.046 against 0.107 ms an expert, the same numbers as the RTX 5090's kernels (relative
  L2 0.00000; 0.0137 against a double reference, the CPU lane's 0.0265). Speculative decoding with drafts from the
  text itself (`STRATA_DS4_SPEC`, `src/core/dsv4_verify.cu`): a window of 1 + k tokens as k + 1 staggered streams;
  the window and the token path give the same logits (KL 0.000000 with every expert on the GPU,
  `STRATA_GLM_SCORE_VERIFY=<T>`). On by default from v1.3.0, behind an acceptance gate.
- **v1.3.1-1.3.5:** a deadline on the lane's calls (the kernel driver can hold the iGPU's queue for seconds); the
  lightning indexer's scores on tensor cores for decode and for prompts, a faster exact top-k select, and the prompt
  attention on tensor cores - a 125K prompt 1,010 -> 2,540 tok/s, decode at 125K 58 -> 66 tok/s. Every kernel
  within ~2e-6 of the FP32 one; KL against llama.cpp unchanged (0.009).
- **v1.3.6:** the prompt's RAM-tier experts on the iGPU as well, only the host rows' activations gathered down, and
  a prestage-adaptation fix: 8K prompt ~1,310 -> ~1,830 tok/s.
- **v1.3.7-1.3.11:** the layer split for deepseek4 (RTX 5090 + RTX 3090 measured), sm_70 builds, the HIP build with
  WMMA kernels (Radeon 8060S: KL 0.005 against the CUDA build).
- **v1.3.12-1.3.16:** Maya v1.1.7's sampler and CPU-row speedups, far fewer VRAM/RAM tier swaps (the 8 MB copies
  stalled the route over PCIe), the shared expert after the route so the lane's request goes out earlier, and a
  survey of the rest (23 candidates measured): decode 67-77 -> 82-84 tok/s, prompts +3-7%.
- **v1.3.18-1.3.23:** the Q6_K GEMV split over 4 warps, the attention projections re-quantized to Q5_K (KL 0.0119,
  the same swe-bench quality, the default pack), the token boundary without the copy engines (zero-copy), and the
  opt-in expert deferral (`STRATA_GLM_DEFER=1`, +12-15% at KL 0.08): decode 82-84 -> 92-95 tok/s, swe-bench ~100.
- **v1.3.24-1.3.27:** the layer split gets deferral, a fitted placement cost that runs on one card when that is faster,
  speculative decoding and prestaging.
- **v1.3.28:** the tier moves' copies paced through the MoE layers - the tokens right after a prompt 16-20 -> 11-13 ms.

## Build and pack

```sh
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16
.venv/bin/python tools/iq_pack.py --gguf <shard 1>.gguf --out ../Abyss-data/models/<repo>/<quant>/pack-dsv4 --compat-bf16
```

## Not done yet

- The MTP / DSpark draft (the GGUF has no MTP block; DSpark needs its own model) - speculative decoding drafts from
  the prompt and the answer instead (n-gram lookup, `STRATA_DS4_SPEC`).
- File conversation slots (`STRATA_GLM_SLOTS`) - off for deepseek4; the live snapshot works.
- Images.
- A layer split runs with speculative decoding and prestaging (v1.3.26 / v1.3.27), but a slower second card does not
  beat one RTX 5090 + the iGPU lane, and its prompts are ~30% behind one card. A fast second card on a real PCIe slot,
  a V100 and a discrete Radeon have not been run.
- Short prompts: at 8K the staging of RAM-tier experts is still ~65% of the time (PCIe Gen4 x4). At 125K the GPU time
  is spread evenly over the indexer, the attention and the expert GEMMs.
- Decode: what is left on this machine is the lane call (~114 us a layer, the iGPU's own time) and the dense GEMVs'
  bandwidth (~5 GB a token with the Q5_K attention pack). Ideas and dead ends: the changelog's v1.3.16-1.3.23 entries.
