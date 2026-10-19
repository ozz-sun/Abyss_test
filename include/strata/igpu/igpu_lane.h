// include/strata/igpu/igpu_lane.h - the iGPU LANE: the CPU lane's RAM-tier experts computed by an AMD APU's integrated
// GPU (Ryzen AI Max+ 395: Radeon 8060S, gfx1151) instead of the CPU cores.  The iGPU reads the same DRAM at ~240 GB/s
// (the CPU lane's threads ~75), the RAM tier included: the engine registers its pinned slots with HIP, so nothing is
// copied.  Built with hipcc into libabyss_igpu.so (src/igpu/igpu_lane.cpp + the HIP build of glm_fast.cu's expert
// kernels) and loaded at run time by the CUDA engine (src/core/glm_fast_path.cu, STRATA_GLM_IGPU): a C interface, so
// neither runtime's headers meet the other's.
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ABYSS_IGPU_API 2   // abyss_igpu_version() of this interface

typedef int (*abyss_igpu_version_fn)(void);
/// The first HIP device; n_embd / n_ff: the largest of the model (buffers).  0 ok, else err holds why.
typedef int (*abyss_igpu_open_fn)(int n_embd, int n_ff, char* err, int err_n);
/// Make host memory [host, host + bytes) readable by the iGPU (pinned already or not: hipHostRegister, mapped).
typedef int (*abyss_igpu_register_fn)(const void* host, size_t bytes, char* err, int err_n);
/// out[n_embd] = sum_i w[i] * expert_i(x): ne (1..8) experts whose blobs (gate | up | down at down_off) lie in
/// registered memory.  Synchronous.  0 ok; 1 a blob is not in registered memory (nothing done: compute it elsewhere);
/// < 0 an error (the lane should stop).
typedef int (*abyss_igpu_experts_fn)(int gu_type, int d_type, int n_embd, int n_ff, size_t down_off, float limit, int ne,
                                     const void* const* blobs, const float* w, const float* x, float* out);
/// Forget a range registered by abyss_igpu_register (before the host frees it).
typedef void (*abyss_igpu_unregister_fn)(const void* host);
// an EXL3 layer's experts (optional symbol abyss_igpu_experts_x3): off9 = suh[3], svh[3], trellis[3] byte offsets
typedef int (*abyss_igpu_experts_x3_fn)(const int* K3, int cb, const size_t* off9, int n_embd, int n_ff, float limit,
                                        int ne, const void* const* blobs, const float* w, const float* x, float* out);
// ... in two halves (abyss_igpu_x3_start / abyss_igpu_x3_finish): the CPU computes another share between them
typedef int (*abyss_igpu_x3_start_fn)(const int* K3, int cb, const size_t* off9, int n_embd, int n_ff, float limit, int ne,
                                      const void* const* blobs, const float* w, const float* x);
typedef int (*abyss_igpu_x3_finish_fn)(float* out, int n_embd);
/// The prompt path's experts on the iGPU (glm_fast's light row kernels): expert i (blob blobs[i], registered) over the
/// rows [r0[i], r0[i] + nr[i]) of [rlo, rhi); row r is token row_tok[r] of x (n_tok rows of n_embd floats).  Gate/up,
/// the clamped swiglu, down: out row r (ld floats apart, ld >= max(n_embd, 2 n_ff)) = expert(x[row_tok[r]]), unweighted.
/// row_tok, x and out are host memory registered with abyss_igpu_register.  Synchronous.  0 ok; 1 not here (a type
/// without the kernels, memory not registered, the lane busy: compute it elsewhere); < 0 an error.
typedef int (*abyss_igpu_rows_fn)(int gu_type, int d_type, int n_embd, int n_ff, size_t down_off, float limit, int ne,
                                  const void* const* blobs, const int* r0, const int* nr, const int* row_tok,
                                  const float* x, int n_tok, int rlo, int rhi, float* out, int ld);
/// A no-op kernel when the lane is idle: called every ~50 us while a decode runs, it keeps the iGPU's clocks up -
/// after 0.3 ms idle a one-expert call took 103 us instead of 57 (Radeon 8060S, power_dpm "auto").
typedef void (*abyss_igpu_poke_fn)(void);
typedef void (*abyss_igpu_close_fn)(void);

#ifdef __cplusplus
}
#endif
