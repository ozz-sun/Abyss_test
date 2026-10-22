// include/strata/kernels/qwen35_fast.hpp - the qwen35moe (Qwen3.5-MoE: Ornith-1.5-122B) decode kernels that have no
// equivalent in glm_fast / dsv4_fast: the gated delta-net's per-head gates (gdn.hpp's kernels are the recurrence
// itself), the attention layers' q / k norms + NEOX RoPE + KV cache write, the GQA attention of one token over the
// FP16 cache (the positions in chunks, then merged), the q projection's sigmoid output gate and the shared expert's
// sigmoid gate.  The math is llama.cpp's src/models/qwen35moe.cpp (build_layer_attn, build_layer_attn_linear,
// build_layer_ffn) and src/models/delta-net-base.cpp (build_delta_net_autoregressive).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::q35 {

/// x[i] = silu(x[i]), i < n.
void silu(float* x, int n, cudaStream_t s);
/// x[i] *= a, i < n.
void scale(float* x, int n, float a, cudaStream_t s);
/// The delta-net's per-head gates: gate[h] = softplus(alpha[h] + dt_bias[h]) * ssm_a[h] (ssm_a = -exp(A_log): the
/// decay exp(gate) lies in (0, 1]); beta[h] = sigmoid(beta[h]) in place.
void gdn_gates(const float* alpha, const float* dt_bias, const float* ssm_a, float* beta, float* gate, int h_v,
               cudaStream_t s);
/// The attention layer's prologue at position p: q_full ([n_head][2 * head_dim], a head's q then its gate) -> q
/// ([n_head][head_dim]: rms-normed by q_norm, NEOX RoPE on the first n_rot dims); k ([n_head_kv][head_dim]: k_norm,
/// RoPE) and v into row p of the FP16 caches kc / vc ([ctx][n_head_kv][head_dim]).  head_dim 256.
void attn_prep(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm, float eps,
               int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, int p, uint16_t* kc, uint16_t* vc,
               float* q, cudaStream_t s);
/// The scratch (floats) attn_decode needs for n_head heads over a cache of max_ctx positions.
size_t attn_part_floats(int n_head, int head_dim, int max_ctx);
/// One token's attention: out[h] = softmax(scale * q[h] . K[g][t]) V[g][t] over t < n_pos, g = h / (n_head /
/// n_head_kv), from the FP16 caches; the positions in chunks (a block each: a partial max / sum / accumulator into
/// part), then merged.  head_dim 256.
void attn_decode(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int n_pos, float scale, float* part, float* out, cudaStream_t s);
/// o[h][d] *= sigmoid(q_full[h][head_dim + d]) (the q projection's gate half), in place.
void attn_gate(float* o, const float* q_full, int n_head, int head_dim, cudaStream_t s);
/// x[i] *= sigmoid(g[0]), i < n (the shared expert's gate).
void scale_sigmoid(float* x, int n, const float* g, cudaStream_t s);

}  // namespace strata::kernels::q35
