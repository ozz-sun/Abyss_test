# Abyss on AMD Radeon (HIP, experimental)

[日本語](README.ja.md)

How to install the Radeon driver and ROCm, then build and run Abyss's HIP engine. The page covers discrete Radeon
cards and the Ryzen AI Max's integrated GPU (iGPU). On a PC that has both, the engine runs on the discrete card and
the iGPU computes the RAM-tier experts (the iGPU lane).

## Supported GPUs

| GPU | Architecture | Role | Checked |
|---|---|---|---|
| Radeon AI PRO R9700, RX 9070 / 9070 XT | gfx1201 | engine | measured on an R9700 (2026-10-20, table below) |
| RX 7900 XT / XTX | gfx1100 | engine | community builds ([issue #4](https://github.com/zephel01/Abyss/issues/4), Windows) |
| Ryzen AI Max's Radeon 8050S / 8060S | gfx1151 | engine (alone), or the iGPU lane | measured on an 8060S |

- The engine is built for gfx1100, gfx1201 and gfx1151. Other Radeons are not targets, such as RDNA2 and older or
  gfx1200 (RX 9060).
- The iGPU lane runs on the integrated gfx1103 and gfx1150-1153.
- NVIDIA and AMD in one process (CUDA and HIP together) is not supported. On a PC with an NVIDIA GPU, the CUDA engine
  uses the Radeon iGPU as its lane (the "iGPU lane" in the [manual](../MANUAL.md)).

## The machine this was checked on

| Item | What |
|---|---|
| PC | GMKtec EVO-X2 (Ryzen AI Max+ 395, Radeon 8060S, 128 GB RAM, 94 GB of it seen by the OS) |
| Discrete GPU | Radeon AI PRO R9700 32 GB (OCuLink, PCIe Gen4 x4) |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-38-generic |
| Driver | amdgpu-dkms 6.19.14 (amdgpu-install 31.40) |
| ROCm | 10.0.0 (apt's `amdrocm` packages, `/opt/rocm/core-10.0`) |

ROCm 7 or newer works; setup checks the version.

## 1. The driver (amdgpu)

ROCm reaches the GPU through the kernel's amdgpu driver and the `/dev/kfd` device it creates.

1. **Install the driver.** The machine above has `amdgpu-dkms` and `amdgpu-dkms-firmware` from AMD's
   `amdgpu-install` (Radeon Software for Linux). The DKMS module is rebuilt when the kernel is updated.
   - A recent kernel's own amdgpu should work as well, as long as `/dev/kfd` exists and `rocminfo` (below) lists the
     GPU. That combination has not been checked here.
   - With Secure Boot on, the DKMS module must be signed; the install asks you to enroll a MOK key.
2. **Add your user to the `render` and `video` groups,** then log in again.

   ```sh
   sudo usermod -aG render,video $USER
   ```

3. **Check.**

   ```sh
   ls -l /dev/kfd /dev/dri/renderD*      # both readable and writable by the render group
   modinfo amdgpu | grep -E "^(filename|version)"
   lspci -D | grep -i -E "vga|display"   # each GPU's PCI address (e.g. 0000:c7:00.0)
   ```

## 2. ROCm

### ROCm 10 (apt's amdrocm packages)

ROCm 10's packages are split by GPU architecture. **Install every architecture you will use.** On a GPU whose
architecture is missing, rocBLAS and hipBLASLt find no kernels.

1. Add the repository and its signing key as AMD's ROCm install guide describes. On the machine above,
   `/etc/apt/sources.list.d/amdrocm-stable.sources` reads:

   ```
   Types: deb
   URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/
   Suites: stable
   Components: main
   Architectures: amd64
   Signed-By: /etc/apt/keyrings/amdrocm.gpg
   ```

2. Install one metapackage for each GPU, plus one development package for the build.

   ```sh
   sudo apt update
   # an R9700 / RX 9070 and a Ryzen AI Max iGPU
   sudo apt install amdrocm10.0-gfx1201 amdrocm10.0-gfx1151 amdrocm-core-dev10.0-gfx1151
   # an RX 7900 XT / XTX alone
   sudo apt install amdrocm10.0-gfx1100 amdrocm-core-dev10.0-gfx1100
   ```

   `amdrocm10.0-<arch>` brings that architecture's core, blas, dnn, fft, rand, rccl, solver and sparse packages.
   When you add a GPU later, install its architecture next to what is there; nothing has to be removed.

3. ROCm 10 lives in `/opt/rocm/core-10.0`, and `/opt/rocm/{bin,lib,llvm}` link there. Abyss's setup looks in
   `/opt/rocm` first, then in the newest `/opt/rocm/core-<version>` when `/opt/rocm` has no compiler. Set
   `ROCM_PATH` when ROCm is somewhere else.

### ROCm 7

ROCm 7 also works, either `amdgpu-install`'s `rocm` or amdrocm 7.x. amdrocm 7.x is split by architecture too, so
install it the same way (`amdrocm7.14-gfx1201` and so on). With several versions installed it is hard to tell which
one `/opt/rocm` points to: remove the ones you do not use, or name one with `ROCM_PATH`.

### Windows

On Windows, setup installs AMD's ROCm SDK (TheRock's wheels) into Abyss's `.venv`; the driver is AMD Software:
Adrenalin Edition. See [AMD HIP (Maya)](../maya/AMD_HIP.md).

## 3. Environment variables

- **Do not set `HSA_OVERRIDE_GFX_VERSION`.** Old guides set it, to `11.5.1` for example, for iGPUs that ROCm did not
  support yet. ROCm supports gfx1151 directly now, so it is not needed. Left set, it stops ROCr on a gfx12 GPU such as
  the R9700 with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`.

  ```sh
  env | grep HSA_                                     # should print nothing
  grep -rn HSA_OVERRIDE ~/.bashrc ~/.profile /etc/profile.d/ /etc/environment 2>/dev/null
  ```

  Comment out any line this finds, then log in again.
- Leave `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` unset as well. Abyss's server works out the engine's value
  from the GPUs in the config.

## 4. Check the install

```sh
rocminfo | grep -E "Marketing Name|Name: +gfx"       # every GPU you will use is listed
amd-smi list                                         # each GPU's number and PCI address (BDF)
hipconfig --version
ls /opt/rocm/lib/rocblas/library | grep -c gfx1201  # 0: amdrocm10.0-gfx1201 is not installed
```

On the machine above, `rocminfo` lists the R9700 (gfx1201), the Radeon 8060S (gfx1151) and the NPU, and rocBLAS has
110 files for gfx1201.

## 5. Abyss's setup

```sh
./setup-deepseek.sh --backend hip   # with no NVIDIA GPU, HIP is chosen without --backend too
```

- Setup finds the GPUs in the KFD topology and picks the one with the most VRAM. Use `--gpu N` to pick another; the
  numbers are those of `amd-smi list`'s `GPU:` lines.
- The engine is built in `build-hip/strata`, which takes 10-20 minutes the first time. With ROCm's `hipcc` present,
  the iGPU lane's `build-hip/libabyss_igpu.so` is built with it.
- The config gets `"backend": "hip"`, `"gpu"` (the HIP numbers), `"gpu_pci"` (the PCI addresses) and `"lib_dirs"`
  (ROCm's `lib`).

## 6. A discrete Radeon and the iGPU together (Ryzen AI Max + an eGPU)

On a Ryzen AI Max with a discrete Radeon attached, the work is split as follows:

- The engine (attention, the VRAM experts) runs on the discrete card.
- The iGPU computes the RAM-tier experts. It reads RAM at ~240 GB/s, against ~75 GB/s for the CPU threads.

Nothing needs to be configured. The server appends the iGPU to `HIP_VISIBLE_DEVICES` (for example `0,1`), the engine
runs on the first GPU, and the lane finds the integrated GPU by itself.

- `STRATA_GLM_IGPU=0` leaves the iGPU unused, and the CPU computes the lane.
- GLM's automatic layer split never picks the iGPU: it is the lane, not a split target.
- The engine log shows `iGPU lane on` and `-> the iGPU computes the lane` when the lane is in use.

### When the GPU numbers change

HIP numbers the GPUs in the order the driver found them at boot (the KFD topology's order). On the machine above the
R9700 is `0000:c7:00.0` and the iGPU `c8:00.0`, so the R9700 is 0 and the iGPU 1. The order flips:

- when the eGPU is attached after boot, or was not powered at boot
- when it is moved to another OCuLink / USB4 port, or a BIOS or kernel setting renumbers the buses

Setup therefore also writes the GPUs' PCI addresses into the config as `"gpu_pci"`. At each start the server looks
up the current numbers from the addresses. When a number has changed, it says so and uses the right card:

```
[abyss] the AMD GPU(s) at 0000:c7:00.0 are HIP number 1 now (the config says 0): using 1
```

Older configs without `"gpu_pci"` keep using the numbers. To add it by hand, take the address from `amd-smi list` or
`lspci -D`:

```json
"gpu": [0],
"gpu_pci": ["0000:c7:00.0"]
```

### Over OCuLink / USB4

- Check the link speed:

  ```sh
  sudo lspci -vv -s c7:00.0 | grep -E "LnkCap|LnkSta"
  ```

  The R9700 has a switch inside the card, and the switch reports Gen5 x16. Read the root port the link comes from
  instead (`00:03.1` on the machine above). The link drops to 2.5 GT/s while idle.
- The machine above measured 7.24 GB/s host-to-device and 7.03 GB/s device-to-host with pinned memory (Gen4 x4).
  Experts are copied from RAM a lot, so this bandwidth shows directly in the speed.
- The machine above boots with `pci=realloc pcie_aspm=off pcie_port_pm=off iommu=pt pcie_ports=native`. These were
  added earlier to keep an NVIDIA eGPU stable; which of them a Radeon needs has not been checked.

## 7. What to expect

R9700 + Radeon 8060S (the iGPU lane), 94 GB RAM, decode speed over 6 swe-bench tasks (tok/s), measured with
`bench/model_gpu_cmp.sh`:

| Model | CPU lane | iGPU lane | 8K / 16K prompt |
|---|---|---|---|
| DeepSeek V4 UD-Q2_K_XL (q5k pack) | 36.4 | 42.4 | 452 / 599 |
| DeepSeek V4 huihui Q2 | 33.9 | 38.1 | 277 / 324 |
| GLM-5.3-Flash Maya-S-v2 | 25.8 | 35.8 | 418 / 541 |
| GLM-5.3-Flash Maya-S24 | 28.2 | 41.2 | 433 / 562 |

- **Accuracy:** KL against the CUDA build is 0.0052, the same level as the spread between two CUDA runs
  (0.0057-0.0060).
- **Decode:** 0.42-0.58x an RTX 5090 + iGPU, and 0.73-0.91x an RTX 3090 + iGPU.
- **Prompts:** 0.2-0.4x a 5090. HIP's prompt path (MMQ, hipBLASLt) is still being improved.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `rocminfo` ends with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` | `HSA_OVERRIDE_GFX_VERSION` is still set | remove it as in "3. Environment variables", then log in again |
| `rocminfo` cannot open `/dev/kfd` (permission denied) | not in the `render` group | `sudo usermod -aG render,video $USER`, then log in again |
| Decode works, but the first prompt hangs; the server answers 503 and the engine is stopped after 180 s | rocBLAS / hipBLASLt have no kernels for that GPU (for example only gfx1151's are installed) | `sudo apt install amdrocm10.0-<arch>` (gfx1201 for an R9700); check as in "4. Check the install" |
| The engine log shows `No compatible code objects found for: gfx1201`, or the iGPU lane fails with `invalid device function` | an older build's `libabyss_igpu.so` has no code for the engine's GPU. HIP refuses a library's kernels on every GPU when any visible GPU has no code in it | rebuild from the current source (`./setup-deepseek.sh --backend hip`) |
| Slow with a discrete card attached, and the log's GPU is the iGPU (gfx1151) | the HIP numbers flipped and the config's `"gpu"` now names the iGPU | add `"gpu_pci"` to the config ("When the GPU numbers change"); check the order with `rocminfo` |
| A discrete card and the iGPU are there, but the log never shows `-> the iGPU computes the lane` | the config's `env` narrows `HIP_VISIBLE_DEVICES` to the discrete card, or sets `STRATA_GLM_IGPU=0` | remove both |
| The build stops at `ROCm's HIP compiler ... and hipBLAS are required` | no development package, or `/opt/rocm` points at another version | install `amdrocm-core-dev10.0-<arch>`, or set `ROCM_PATH=/opt/rocm/core-10.0` |
