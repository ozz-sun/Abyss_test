# EXL3 quants on Abyss (GLM-5.3-Flash, experimental)

From v1.3.35 Abyss runs [exllamav3](https://github.com/turboderp-org/exllamav3)'s **EXL3** quants of
GLM-5.3-Flash on the same fast path as its GGUFs: the experts in VRAM, pinned RAM (computed by an APU's iGPU or the
CPU) and the SSD. The engine reads the trellis-coded weights in place from the model's safetensors. The code is
[Project Maya](https://github.com/mw00/project-maya)'s EXL3 support (by @needmorevram, after exllamav3's format and
kernels), ported to Abyss's engine, with the iGPU lane added.

DeepSeek V4 / V4.1 EXL3 quants do not load yet (they need their own packer); this page is about GLM-5.3-Flash.

## Why EXL3

EXL3 is a QTIP-style trellis code: each weight matrix is rotated (random signs and a 128-point Hadamard on both
sides), cut into 16 x 16 tiles, and each tile is a tail-biting trellis of K-bit steps found by a Viterbi search.
At the same size it predicts text better than a GGUF k- or i-quant; the price is more arithmetic per weight to
decode it.

On this machine (RTX 5090 32 GB + Radeon 8060S iGPU lane + 94 GB RAM), `glm_ids4k`'s last 1,000 tokens after 3,000:

| | size | nll (lower is better) | top-1 | ms a token | a RAM-tier expert on the iGPU |
| --- | --- | --- | --- | --- | --- |
| Maya-S-v2-IQ2_XXS (GGUF) | 96.5 GB | 1.634 | 62.3% | 14.8 | 0.041 ms |
| **EXL3 sc_2.10bpw** (turboderp) | 88.3 GB | **1.591-1.598** | **63.3-63.7%** | 22.4-22.7 | 0.088 ms |

Better text at 8 GB less, at about two thirds of the decode speed: here ~140-190 experts a token are computed on the
RAM tier's lane, and an EXL3 expert costs the iGPU about twice a GGUF one (its trellis decode is compute-bound there;
the GPU's own experts decode on its tensor cores and are cheap). With a card that holds most of the experts the
difference mostly goes away.

## Which PCs it suits

The installer (`./abyss.sh --setup`) shows this next to each EXL3 quant in the model question:

| this PC | the installer says |
| --- | --- |
| NVIDIA RTX 30 or newer whose VRAM holds 80% of the quant | **recommended for quality** (the default: the largest such quant) |
| an AMD APU's iGPU (Ryzen AI Max) or an AVX-512 CPU for the RAM tier | experimental, quality first: better text, slower decode |
| a CPU with AVX2 only and no iGPU | experimental, not for this PC (an EXL3 expert on the CPU there is ~20x a copy over PCIe) |
| the GPU and 3/4 of the RAM do not hold it | experimental - experts would come from the SSD |

Each quant gets its own config (`maya-<quant>.json`) and start script (`run-maya-<quant>.sh`), so a GGUF and an EXL3
install sit side by side: start the one you want. `--model EXL3-sc_2.10bpw` skips the question.

The setup offers EXL3 with NVIDIA GPUs (the CUDA build); the HIP build compiles the kernels (they pass their tests on
a Radeon 8060S) but runs no EXL3 model through the installer.

## The quants it offers

| `--model` | from | size | experts | notes |
| --- | --- | --- | --- | --- |
| `EXL3-sc_2.10bpw` | [turboderp, sc_2.10bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/sc_2.10bpw) @ 8d2eca0e | 88.3 GB | 2-bit (3-bit in layer 3), 6-bit attention / shared | self-calibrated; no MTP draft block |
| `EXL3-3.05bpw` | [turboderp, 3.05bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw) @ 332ab457 | 125.3 GB | 3-bit, 5-6-bit attention / shared | with the MTP draft block |

Every file is checked against its published size and sha256; the vision files are the GGUF quants' (the same tower).

## By hand

```sh
hf download turboderp/GLM-5.3-Flash-exl3 --revision sc_2.10bpw --local-dir <dir>
.venv/bin/python tools/exl3_pack.py --model <dir>          # ~5 s: <dir>/maya-exl3.gguf and <dir>/pack/
```

The pack keeps the index, ~1 GB of small float weights, a ~2 GB GGUF of the unquantized tensors and the tokenizer;
the EXL3 matrices stay in the safetensors. Then start the engine on `--glm-pack <dir>/pack` as for a GGUF pack (the
engine sees `pack/exl3.txt` and reads the model as EXL3 - no switch). `--check <a GGUF of the same model, shard 1>`
compares the decoded weights with that GGUF's (cosine ~0.95+ between two quants; a layout mistake shows as ~0).

## How Abyss runs it

- **VRAM experts and the dense weights**: Maya's EXL3 kernels (`src/kernels/cuda/exl3.cu`): on RTX 30 and newer a
  tile decodes straight into an `mma.m16n8k16`, before that on the CUDA cores (`exl3_legacy.cu`).
- **The RAM tier's lane**: the iGPU (`abyss_igpu_experts_x3` in `libabyss_igpu.so`, the same kernels built by hipcc
  for gfx1151) when it is faster than the CPU at start, else the CPU threads (Maya's split-K AVX-512 / AVX2 kernels,
  `src/kernels/cpu/exl3_cpu.cpp`: 0.118 ms an expert on 16 Zen 5 threads).
- **Prompts**: a lightly routed expert through the tensor-core kernels, the rest decoded to FP16 and multiplied by
  cuBLAS with its rows rotated instead of its weights. The RAM-tier experts are staged to the GPU (no CPU share yet).
- **The SSD tier, tail skip, streamed fetch, deferral**: unchanged; an EXL3 expert is a blob of [gate | up | down]
  pieces like a GGUF one.

## Settings

| variable | default | effect |
| --- | --- | --- |
| `STRATA_GLM_X3_SPLIT` | off | the CPU threads compute this fraction of each route's RAM-tier experts while the iGPU computes the rest (`auto`: by their start-up timings). Slower on a Ryzen AI Max+ 395, whose CPU and iGPU share power and heat: 22.7 ms a token on the iGPU alone vs 23.5 (CPU 46% on 8 threads), 25.1 (25% on 4), 31.6 (16 threads) |
| `STRATA_GLM_EXL3_DECODE` | by GPU | `new` (tensor-core tiles) or `legacy` (the CUDA-core kernels) on any GPU (A/B) |
| `STRATA_GLM_EXL3_ROT` | on | prompts: an expert's rows rotated, its weight only decoded; `0`: the weight rotated |
| `STRATA_GLM_EXL3_MMA_ROWS` | 64 | prompts: the most rows an expert may have to go through the tensor-core kernels instead of cuBLAS (`0`: none) |

## Not yet

DeepSeek V4 / V4.1 EXL3 quants (turboderp's V4 0731 2.52 bpw would be KL 0.143 against the Q2_K_XL's 0.203 at about
the same size), EXL3's half-integer bitrates (exllamav3 1.6's K 1.5 / 2.5 / 3.5), a CPU share of a prompt's
RAM-tier experts, and an iGPU trellis kernel tuned for RDNA.
