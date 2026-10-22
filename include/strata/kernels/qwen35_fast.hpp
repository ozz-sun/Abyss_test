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

// ---- the prompt path: the same layers over T rows (a chunk's sub-batch of tokens, row t = position p0 + t)
/// R[t] (row stride ldr floats; its first n) += add[t] (add: null adds nothing), then x[t] = rms(R[t]) * w and x16 its
/// FP16 copy (the plain residual's read, as rms_q8 is the token path's).
void add_rms_rows(float* R, int ldr, const float* add, const float* w, float eps, int T, int n, float* x, void* x16,
                  cudaStream_t s);
/// The causal conv over T rows of x ([T][C]) with the carried state ([C][d_conv - 1], the oldest first):
/// out[t][c] = silu(sum_i kW[c][i] * in(t - (d_conv - 1) + i, c)), then the state holds the last d_conv - 1 rows.
void conv_rows(const float* x, int T, int C, const float* kW, int d_conv, float* conv_state, float* out,
               cudaStream_t s);
/// In place on the conv rows h ([T][C]: a row's q heads, then its k heads, then v): every q and k head L2-normed over
/// S, q scaled by 1 / sqrt(S).  S <= 128.
void gdn_prep_rows(float* h, int T, int C, int h_k, int S, float eps, cudaStream_t s);
/// gdn_gates over T rows: alpha[t][h] -> the gate softplus(alpha + dt_bias[h]) * ssm_a[h], beta[t][h] -> sigmoid.
void gdn_gates_rows(float* alpha, float* beta, const float* dt_bias, const float* ssm_a, int T, int h_v, cudaStream_t s);
/// gdn_prep_rows and gdn_gates_rows in one launch (the gates' blocks beside the heads').
void gdn_prep_gates(float* h, int T, int C, int h_k, int S, float eps, float* alpha, float* beta, const float* dt_bias,
                    const float* ssm_a, int h_v, cudaStream_t s);
/// The token path's conv step (gdn_conv_step: one input row x, the state slid) with the SiLU in the same launch.
void conv_silu_step(float* conv_state, const float* x, const float* kW, int C, int d_conv, float* out, cudaStream_t s);
/// The recurrence over T rows in order (gdn_step's math, the state (S, h_v, S) updated in place): q / k / v from the
/// conv rows (stride C), gate / beta [T][h_v]; o [T][h_v * S].  S == 128.
void gdn_scan(float* state, const float* h, int T, int C, const float* gate, const float* beta, int S, int h_k, int h_v,
              float* o, cudaStream_t s);
/// attn_prep over T rows at positions p0 + t: q_full [T][n_head][2 * head_dim], k / v [T][n_head_kv][head_dim] ->
/// q [T][n_head][head_dim] and the caches' rows p0 .. p0 + T - 1.
void attn_prep_rows(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm,
                    float eps, int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, int64_t p0, int T,
                    uint16_t* kc, uint16_t* vc, float* q, cudaStream_t s);
/// The causal attention of T queries (row t at position p0 + t, over the cache's positions <= p0 + t):
/// out [T][n_head][head_dim].  A block takes 4 queries and one KV head's group of n_head / n_head_kv heads (8, 16 or
/// 32 of them), a warp 1, 2 or 4 heads, a lane 8 dims; the keys stream past once a block.  head_dim 256.
void attn_prompt(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int64_t p0, int T, float scale, float* out, cudaStream_t s);
/// attn_gate over T rows.
void attn_gate_rows(float* o, const float* q_full, int T, int n_head, int head_dim, cudaStream_t s);
/// ffn[t] *= sigmoid(x[t] . w) over T rows of n (the shared expert's gate from its input rows).
void shexp_gate_rows(const float* x, const float* w, float* ffn, int T, int n, cudaStream_t s);

}  // namespace strata::kernels::q35
