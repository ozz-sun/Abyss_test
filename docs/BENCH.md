# Measuring Abyss on your PC (`--bench`)

[日本語](BENCH.ja.md)

Abyss runs models that do not fit in GPU memory: each token's experts come from GPU memory, from a RAM tier, or from
the SSD, and some are computed by a CPU (or iGPU) lane. Which of those limits the speed depends on the PC - the GPU,
its PCIe slot, the RAM, the SSD, the CPU. We can only measure our own PCs, so the benchmark is built to be run on
yours and pasted into an issue: it gives the same numbers on every machine, says where a token's time goes, and checks
that the PC computes what a known-good run did.

## Running it

Stop Abyss first (the benchmark needs the GPU memory it holds), then:

```
./abyss.sh --bench                 # a few minutes: speed, where the time goes, the quality check
./abyss.sh --bench --deep          # + the GPU timeline and a PCIe / lane share sweep (finds the bottleneck)
./abyss.sh --bench --full          # + three runs of each question and a 32k prompt
./abyss.sh --bench --config abyss-ornith-r9700l.json    # another installed model than the last one used
```

Windows: `START-ABYSS.bat --bench` (the same options). The results go to `abyss-bench.txt` in the Abyss folder
(`--report` includes it). Nothing is sent anywhere, and the file has no host name, user name or folder path - read it,
then paste it into a [speed report](https://github.com/zephel01/Abyss/issues/new?template=bench.md).

What it runs, through the engine alone (the way the dashboard starts it):

| step | what | why |
|---|---|---|
| decode | three fixed questions, 256 tokens each, greedy, thinking off (after one warm-up answer) | the same work on every PC, so tok/s compare |
| prefill | prompts of 2k and 8k tokens of the Abyss docs (32k with `--full`) | prompt reading speed |
| counters | the engine's per-request line (`glm stat`) of each answer | where a token's time goes |
| PCIe | the GPU's link, read while the engine works | links drop to Gen1 when idle, so an idle reading lies |
| SSD | 2 GB of the model's own GGUF, read from the disk (past the page cache) | the speed the engine's SSD tier gets |
| quality | 512 known tokens scored against `bench/ref/<model>` (a known-good run) | a broken kernel on this GPU shows as a large KL |
| `--deep` | the GPU timeline (`STRATA_GLM_GAP_PROF=1`) and five answers at PCIe shares 0 / 0.25 / 0.5 / 0.75 / 1 | what moves the speed here |

## Reading the result

The first line is the summary - one line to compare PCs (docs/MODELS.md's tables come from lines like it):

```
Abyss v1.3.55 | Ornith-1.5-122B-UD-Q4_K_XL | AMD Radeon AI PRO R9700 32 GB | PCIe Gen4 x4 (~7.9 GB/s), the GPU can do
Gen5 x16 | RAM 94 GB | SSD 4.9 GB/s | decode 56.4 tok/s | prompt 8k 960 tok/s | quality OK | bottleneck: gpu
```

### Where a token's time goes

```
- a token: 17.8 ms (56.2 tokens/s)
- SSD waits: 0.0 ms a token (0%), 0.0 expert reads a token
- iGPU lane: busy 7.1 ms a token (40%), 246 experts a token
- GPU memory: holds 47% of the experts a token uses; 0.4 a token come over PCIe from RAM
- drafter: 76% of its drafts accepted
```

| line | what it is | what a bad value means |
|---|---|---|
| a token | decode time per token (with the drafter: per accepted token) | - |
| SSD waits | time the GPU waited for experts read from the SSD | a quarter of a token or more: the model does not fit GPU + RAM |
| CPU / iGPU lane | time the lane was busy with the experts it computes (in parallel with the GPU) | close to the whole token: the lane is the critical path |
| GPU memory | the share of a token's experts already in GPU memory, and those copied from RAM over PCIe | many copies a token: PCIe carries the token |
| drafter | the share of the drafter's guesses the model kept | low: the drafter costs more than it saves (`--dspark` off, or another drafter) |
| tail skip | lossy setting on (its KL is in docs/MODELS.md) | - |

### The verdict

The benchmark names the likely bottleneck from those counters (rules of thumb, checked on an R9700 and an RTX 5090):

| verdict | rule | what moves it |
|---|---|---|
| **ssd** | SSD waits >= 25% of a token | a smaller quant that fits GPU + RAM, more RAM for the RAM tier (`STRATA_GLM_RAM_GB`), a faster SSD |
| **cpu** | the lane busy >= 80% of a token | `./abyss.sh --calibrate` (the lane's share and threads), the CPU's power mode, the iGPU lane on a Radeon APU |
| **pcie** | >= 10 experts a token copied over PCIe | a better slot (Gen4 x16 carries 4x an x4 / OCuLink link), `--calibrate` |
| **gpu** | none of the above (or, with `--deep` and no drafter, the GPU busy >= 85%) | `./abyss.sh --gpu-tune`, and a report: the kernels for this GPU may have room |

An SSD-bound run (DeepSeek V4.1 Flash 1017 UD-IQ2_XXS, 300 GB, on the same PC) reads like this:

```
- a token: 50.1 ms (20.0 tokens/s)
- SSD waits: 19.5 ms a token (39%), 8.5 expert reads a token
- iGPU lane: busy 6.4 ms a token (13%), 96 experts a token
- likely bottleneck: the SSD: the experts that fit neither in GPU memory nor in the RAM tier are read from disk every token
```

"gpu" is also what is left when nothing else stands out - the memory tiers keep up, so the GPU's own kernels (and the
host between tokens) are what is left to speed up. That is where a report from a GPU we do not have helps most.

### `--deep`: the share sweep

The engine splits the experts that are not in GPU memory between the lane (computed on the CPU / iGPU, from RAM) and
PCIe (copied to the GPU). `--deep` answers the same question at five shares:

```
  share 0.00:   58.4 tokens/s  <- fastest
  share 0.25:   23.3 tokens/s
  ...
  share 1.00:   12.2 tokens/s
  the engine's own choice is already the fastest; share 1.00 is 4.8x slower - PCIe is the narrow path here
```

- The fastest share well above the engine's own speed: `./abyss.sh --calibrate` sets it for this PC.
- The engine already the fastest, and the curve steep: one path is much narrower - here PCIe (Gen4 x4 over OCuLink):
  every expert moved to it costs more than the lane's work.
- A flat curve: neither PCIe nor the lane limits the speed; look at the GPU (or the SSD).

### The quality check

```
quality: OK (KL 0.0078 vs a known-good run, same top token 98.2%)
```

The engine scores 512 tokens of known text (English and Japanese) one token at a time and compares each position's
top-32 log-probs with those the same model gave on our PC (`bench/ref/<model>.json` / `.topk`, made by
`tools/bench_ref.py`). The same PC gives 0 to ~0.01 (which experts the GPU and which the lane computes
varies run to run, and the two round differently); another GPU's arithmetic adds a little. **OK** is up to 0.02,
**slightly off** up to 0.05, **OFF** above - then this PC computes something the known-good run did not (a kernel bug
on that GPU, issue #9 style): please report it.

- The check needs the same model files: another quant, or a pack with an overlay (`tools/dense_overlay.py`), says
  "skipped" instead.
- Settings that change the numbers on purpose (`STRATA_GLM_TAIL_SKIP*`, `STRATA_GLM_ABL`, `STRATA_GLM_DEFER`) are
  left out of the scoring run, and the result says so.
- References ship for: Ornith-1.5-122B UD-Q4_K_XL, DeepSeek V4.1 Flash 1017 UD-IQ2_XXS, DeepSeek V4 Flash 0731
  UD-Q2_K_XL, GLM-5.3-Flash Maya-S-v2 IQ2_XXS.

## Platforms

| | Linux, AMD | Linux, NVIDIA | Windows, NVIDIA | Windows, AMD |
|---|---|---|---|---|
| speed, counters, verdict, quality, `--deep` | yes | yes | yes | yes |
| PCIe link | yes (sysfs, the narrowest link up to the root port) | yes (nvidia-smi) | yes (nvidia-smi) | no |
| SSD speed | yes | yes | yes (unbuffered read) | yes (unbuffered read) |

An AMD card shows its own internal switch as x16; the benchmark walks up to the root port, so an OCuLink x4 or a
chipset slot shows as what it is.

## What else helps a report

- `./abyss.sh --gpu-tune` (a minute, no model): the per-GPU kernel choices; paste `abyss-gpu-tune.txt` too.
- `./abyss.sh --report`: the PC, the setup and the engine log's speed lines (`abyss-report.txt`, includes the
  benchmark).
- Before / after a change: run `--bench` twice and paste both - the counters show what the change moved.
