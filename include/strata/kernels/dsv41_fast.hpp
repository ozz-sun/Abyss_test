// include/strata/kernels/dsv41_fast.hpp - the deepseek41 (DeepSeek V4.1 Flash) decode kernels that differ from
// deepseek4's (dsv4_fast.hpp): the compressed rows of a KV-source layer (pairs pooled by a per-channel softmax gate,
// or one row a token) with their index keys, q's RoPE without the per-head norm, the candidate blocks, Engram's gate
// and the head's collapse.  The math is in Abyss-tools/worklog/2026-10-18_deepseek41-spec.md (sections 2, 3, 5, 6),
// transcribed from antirez's ds4 (ds4.c ds41_*, ds4_deepseek41_cuda.cuh) and llama.cpp's deepseek41.cpp.
#pragma once

#include "strata/kernels/dsv4_fast.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::dsv41 {

using dsv4::Rope;

/// One token at position p on a KV-source layer (spec 5.3).  v (and g, ratio 2) = the compressor's projections
/// (D = 512 each).  Ratio 2: an even p parks v and g in pend (2 x D) and writes nothing; an odd p pools the pair per
/// channel (softmax of the two gate values) into row p / 2, RoPE position p - 1.  Ratio 1: row p = v, RoPE position p.
/// A completed row c: c = rms(c) * norm; the index key ik = rms(idx_k . c) * idx_k_norm (BF16 idx_k [128][D], from
/// the UNROTATED c), RoPE on ik's last n_rot dims -> idx_cache row (FP16, 128); RoPE on c's last n_rot dims ->
/// comp_cache row (FP16, D).
struct PublishArgs {
    const float* v = nullptr;
    const float* g = nullptr;
    float* pend = nullptr;
    const float* norm = nullptr;
    const uint16_t* idx_k = nullptr;
    const float* idx_k_norm = nullptr;
    uint16_t* comp_cache = nullptr;
    uint16_t* idx_cache = nullptr;
    int ratio = 2, p = 0, D = 512, idx_dim = 128, n_rot = 64;
    Rope rope;
    float eps = 1e-20f;
};
void publish(const PublishArgs& a, cudaStream_t s);

/// q (n_head x head_dim): RoPE on the last n_rot dims at p (V4.1 has no per-head q norm); iq (may be null): RoPE on
/// its last n_rot dims at p with comp_rope.
void q_rope(float* q, int n_head, int head_dim, int n_rot, const Rope& rope, float* iq, int idx_heads, int idx_dim,
            const Rope& comp_rope, int p, cudaStream_t s);

/// The candidate blocks (spec 2) at the candidate-source layer: bs[c] = max of score over rows [8c, 8c+8) < n_vis,
/// the newest block +inf, then the top min(top_blocks, B) blocks -> keep[c] = 1 (0 otherwise), B = ceil(n_vis / bs).
/// bscore (B floats) and bsel (top_blocks ints) are scratch.
void candidates(const float* score, int n_vis, int block, int top_blocks, float* bscore, int* bsel, unsigned char* keep,
                cudaStream_t s);
/// score[j] = -inf where keep[j / block] == 0, j < n_vis.
void candidate_mask(float* score, int n_vis, int block, const unsigned char* keep, cudaStream_t s);

/// Engram's add (spec 3.5) into the 4 streams of R (hc x n_embd) in place: kv = 5 x n_embd (4 keys, the value),
/// qn / kn = [4][n_embd] norm weights; per stream h: s = rms(R_h) * qn_h . rms(key_h) * kn_h / sqrt(n_embd),
/// g = sigmoid(sign(s) sqrt(max(|s|, 1e-6))), R_h += g * value.
void engram_add(float* R, const float* kv, const float* qn, const float* kn, int n_embd, float eps, cudaStream_t s);
/// Waits on the device until the host-pinned *flag == want, then copies n floats from the pinned src to dst.
void engram_wait(const unsigned int* flag, unsigned int want, const float* src, float* dst, int n, cudaStream_t s);

/// The head's prologue (spec 6.2): h = sum_s pre_s R_s (the last sublayer's pre), x = rms(h) * norm_w -> x (f32) and
/// xq (q8_1); h (may be null): h itself, before the norm (the DSpark drafter's confidence reads it).
void head_prep(const float* R, const float* pre, const float* norm_w, float eps, int n_embd, float* x, void* xq,
               cudaStream_t s, float* h = nullptr);

/// rows rows of n floats (n % 32 == 0, any width - glmf::rms_q8 stops at 4096): x[r] = rms(in[r]) * w -> x (f32) and
/// xq (q8_1, n / 32 blocks a row).
void rms_rows(const float* in, int rows, const float* w, float eps, int n, float* x, void* xq, cudaStream_t s);

}  // namespace strata::kernels::dsv41
