// src/kernels/cuda/dsv41_fast.cu - see include/strata/kernels/dsv41_fast.hpp.
//
// Transcribed from antirez's ds4 (MIT: ds4.c ds41_attention_publish / ds41_attention_candidates / ds41_graph_logits,
// ds4_deepseek41_cuda.cuh dsv41_pool_kernel / dsv41_candidates_kernel / dsv41_engram_kernel) and llama.cpp's
// deepseek41.cpp (MIT), without ds4's BF16 / FP8 / FP4 rounding emulation (the caches are FP16, as for deepseek4).
#include "strata/kernels/dsv41_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "dsa_topk.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace strata::kernels::dsv41 {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    __syncthreads();
    return t;
}
__device__ __forceinline__ void q8_1_store_warp(float v, block_q8_1* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t) v << 16); }

// dsv4_fast.cu's rope_cs / rope_pair (ggml's rope_yarn, NORM mode: adjacent pairs of the last n_rot dims)
__device__ __forceinline__ void rope_cs(const Rope& r, int pos, int iw, float& c, float& s) {
    const float theta_extrap = (float) pos * powf(r.theta_scale, iw / 2.0f);
    const float theta_interp = r.freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = r.attn_factor;
    if (r.ext_factor != 0.0f) {
        const float y = (float) (iw / 2 - r.corr_lo) / fmaxf(0.001f, r.corr_hi - r.corr_lo);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float ramp_mix = ramp * r.ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / r.freq_scale);
    }
    c = cosf(theta) * mscale;
    s = sinf(theta) * mscale;
}
__device__ __forceinline__ void rope_pair(float* v, int dim, int n_rot, const Rope& r, int pos, int t) {
    const int iw = 2 * t;
    const int i0 = dim - n_rot + iw;
    float c, s;
    rope_cs(r, pos, iw, c, s);
    const float x0 = v[i0], x1 = v[i0 + 1];
    v[i0] = x0 * c - x1 * s;
    v[i0 + 1] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------- the compressed row of a KV source
constexpr int PUB_THREADS = 512;
__global__ void __launch_bounds__(PUB_THREADS) publish_kernel(const PublishArgs a) {
    __shared__ float sh[32];
    __shared__ float c[512];
    __shared__ float ik[128];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int D = a.D;
    int row, rpos;
    if (a.ratio == 2) {
        if ((a.p & 1) == 0) {   // the pair's first token: park it
            for (int d = tid; d < D; d += PUB_THREADS) {
                a.pend[d] = a.v[d];
                a.pend[D + d] = a.g[d];
            }
            return;
        }
        for (int d = tid; d < D; d += PUB_THREADS) {
            const float g0 = a.pend[D + d], g1 = a.g[d];
            const float m = fmaxf(g0, g1);
            const float e0 = expf(g0 - m), e1 = expf(g1 - m);
            c[d] = (a.pend[d] * e0 + a.v[d] * e1) / (e0 + e1);
        }
        row = a.p / 2;
        rpos = a.p - 1;
    } else {
        for (int d = tid; d < D; d += PUB_THREADS) c[d] = a.v[d];
        row = a.p;
        rpos = a.p;
    }
    __syncthreads();
    float ss = 0.0f;
    for (int d = tid; d < D; d += PUB_THREADS) ss += c[d] * c[d];
    const float inv = rsqrtf(block_sum(ss, sh) / (float) D + a.eps);
    for (int d = tid; d < D; d += PUB_THREADS) c[d] = c[d] * inv * a.norm[d];
    __syncthreads();
    // the index key from the unrotated latent: one warp an output row of idx_k
    for (int o = warp; o < a.idx_dim; o += PUB_THREADS / 32) {
        const uint16_t* w = a.idx_k + (size_t) o * D;
        float acc = 0.0f;
        for (int d = lane; d < D; d += 32) acc += bf(w[d]) * c[d];
        acc = warp_sum(acc);
        if (lane == 0) ik[o] = acc;
    }
    __syncthreads();
    float ss2 = 0.0f;
    for (int i = tid; i < a.idx_dim; i += PUB_THREADS) ss2 += ik[i] * ik[i];
    const float inv2 = rsqrtf(block_sum(ss2, sh) / (float) a.idx_dim + a.eps);
    for (int i = tid; i < a.idx_dim; i += PUB_THREADS) ik[i] = ik[i] * inv2 * a.idx_k_norm[i];
    __syncthreads();
    if (tid < a.n_rot / 2) rope_pair(ik, a.idx_dim, a.n_rot, a.rope, rpos, tid);
    else if (tid >= 64 && tid < 64 + a.n_rot / 2) rope_pair(c, D, a.n_rot, a.rope, rpos, tid - 64);
    __syncthreads();
    uint16_t* kr = a.idx_cache + (size_t) row * a.idx_dim;
    for (int i = tid; i < a.idx_dim; i += PUB_THREADS) kr[i] = __half_as_ushort(__float2half(ik[i]));
    uint16_t* cr = a.comp_cache + (size_t) row * D;
    for (int d = tid; d < D; d += PUB_THREADS) cr[d] = __half_as_ushort(__float2half(c[d]));
}

// ---------------------------------------------------------------- q / iq RoPE (no norm)
__global__ void __launch_bounds__(128) q_rope_kernel(float* q, int n_head, int head_dim, int n_rot, Rope rope, float* iq,
                                                     int idx_dim, Rope comp_rope, int p) {
    __shared__ float v[512];
    const int tid = threadIdx.x;
    const bool isq = (int) blockIdx.x < n_head;
    const int dim = isq ? head_dim : idx_dim;
    float* x = isq ? q + (size_t) blockIdx.x * head_dim : iq + (size_t) (blockIdx.x - n_head) * idx_dim;
    // only the rotated tail moves: the pairs [dim - n_rot, dim)
    const int base = dim - n_rot;
    for (int i = tid; i < n_rot; i += blockDim.x) v[base + i] = x[base + i];
    __syncthreads();
    if (tid < n_rot / 2) rope_pair(v, dim, n_rot, isq ? rope : comp_rope, p, tid);
    __syncthreads();
    for (int i = tid; i < n_rot; i += blockDim.x) x[base + i] = v[base + i];
}

// ---------------------------------------------------------------- candidate blocks
__global__ void block_max_kernel(const float* __restrict__ score, int n_vis, int block, float* __restrict__ bscore,
                                 int B) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= B) return;
    float m = -INFINITY;
    const int j1 = min(n_vis, (c + 1) * block);
    for (int j = c * block; j < j1; ++j) m = fmaxf(m, score[j]);
    bscore[c] = c == (n_vis - 1) / block ? INFINITY : m;
}
__global__ void __launch_bounds__(1024) block_select_kernel(const float* __restrict__ bscore, int B, int top,
                                                            int* __restrict__ bsel) {
    strata::kernels::dsa::select_top_pools(bscore, B, top, 1, bsel);
}
__global__ void keep_kernel(const int* __restrict__ bsel, int top, int B, unsigned char* __restrict__ keep) {
    // keep was zeroed by the caller's memset; the selected blocks set it
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < top) {
        const int c = bsel[i];
        if (c >= 0 && c < B) keep[c] = 1;
    }
}
__global__ void mask_kernel(float* __restrict__ score, int n_vis, int block, const unsigned char* __restrict__ keep) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_vis && keep[j / block] == 0) score[j] = -INFINITY;
}

// ---------------------------------------------------------------- Engram's gate and add
__global__ void __launch_bounds__(512) engram_kernel(float* __restrict__ R, const float* __restrict__ kv,
                                                     const float* __restrict__ qn, const float* __restrict__ kn,
                                                     int n_embd, float eps) {
    __shared__ float sh[32];
    const int h = blockIdx.x, tid = threadIdx.x;
    float* x = R + (size_t) h * n_embd;
    const float* key = kv + (size_t) h * n_embd;
    const float* val = kv + (size_t) 4 * n_embd;
    const float* wq = qn + (size_t) h * n_embd;
    const float* wk = kn + (size_t) h * n_embd;
    float sx = 0.0f, sk = 0.0f, dot = 0.0f;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        const float xv = x[i], kv_ = key[i];
        sx += xv * xv;
        sk += kv_ * kv_;
        dot += xv * wq[i] * kv_ * wk[i];
    }
    sx = block_sum(sx, sh);
    sk = block_sum(sk, sh);
    dot = block_sum(dot, sh);
    const float sc = dot * rsqrtf(sx / (float) n_embd + eps) * rsqrtf(sk / (float) n_embd + eps) / sqrtf((float) n_embd);
    const float gl = copysignf(sqrtf(fmaxf(fabsf(sc), 1e-6f)), sc);
    const float g = 1.0f / (1.0f + expf(-gl));
    for (int i = tid; i < n_embd; i += blockDim.x) x[i] += g * val[i];
}

// ---------------------------------------------------------------- the head's collapse
__global__ void __launch_bounds__(1024) head_kernel(const float* __restrict__ R, const float* __restrict__ pre,
                                                    const float* __restrict__ norm_w, float eps, int n_embd,
                                                    float* __restrict__ x, block_q8_1* __restrict__ xq) {
    __shared__ float sh[32];
    __shared__ float sm[5120];
    const int tid = threadIdx.x;
    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    float ss = 0.0f;
    for (int e = tid; e < n_embd; e += blockDim.x) {
        const float v = p0 * R[e] + p1 * R[e + n_embd] + p2 * R[e + 2 * (size_t) n_embd] + p3 * R[e + 3 * (size_t) n_embd];
        sm[e] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / (float) n_embd + eps);
    for (int e = tid; e < n_embd; e += blockDim.x) {   // blockDim 1024: whole warps over 32 consecutive e
        const float v = sm[e] * inv * norm_w[e];
        x[e] = v;
        q8_1_store_warp(v, xq + e / 32, tid & 31);
    }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "dsv41_fast %s: %s\n", what, cudaGetErrorString(e));
}

}  // namespace

void publish(const PublishArgs& a, cudaStream_t s) {
    publish_kernel<<<1, PUB_THREADS, 0, s>>>(a);
    check("publish");
}

void q_rope(float* q, int n_head, int head_dim, int n_rot, const Rope& rope, float* iq, int idx_heads, int idx_dim,
            const Rope& comp_rope, int p, cudaStream_t s) {
    const int blocks = n_head + (iq != nullptr ? idx_heads : 0);
    q_rope_kernel<<<blocks, 128, 0, s>>>(q, n_head, head_dim, n_rot, rope, iq, idx_dim, comp_rope, p);
    check("q_rope");
}

void candidates(const float* score, int n_vis, int block, int top_blocks, float* bscore, int* bsel, unsigned char* keep,
                cudaStream_t s) {
    if (n_vis <= 0) return;
    const int B = (n_vis + block - 1) / block;
    const int top = std::min({top_blocks, B, strata::kernels::dsa::kMaxTop});   // (cand_blocks 2048 = kMaxTop)
    cudaMemsetAsync(keep, 0, (size_t) B, s);
    block_max_kernel<<<(B + 255) / 256, 256, 0, s>>>(score, n_vis, block, bscore, B);
    block_select_kernel<<<1, 1024, 0, s>>>(bscore, B, top, bsel);
    keep_kernel<<<(top + 255) / 256, 256, 0, s>>>(bsel, top, B, keep);
    check("candidates");
}

void candidate_mask(float* score, int n_vis, int block, const unsigned char* keep, cudaStream_t s) {
    if (n_vis <= 0) return;
    mask_kernel<<<(n_vis + 255) / 256, 256, 0, s>>>(score, n_vis, block, keep);
    check("candidate_mask");
}

// the token's Engram rows of one table, once the host's reader has landed them in pinned memory (*flag == want):
// copied into the device rows by zero-copy reads (no memcpy on the stream - it would queue behind the expert DMAs)
__global__ void engram_wait_kernel(const volatile unsigned int* flag, unsigned int want, const float* src, float* dst,
                                   int n) {
    if (threadIdx.x == 0) {
        const long long t0 = clock64();
        while (*flag != want)
            if (clock64() - t0 > (1ll << 36)) break;   // (a failed read: the host reports it at the next token)
        __threadfence_system();
    }
    __syncthreads();
    const volatile float* vs = (const volatile float*) src;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = vs[i];
}

void engram_wait(const unsigned int* flag, unsigned int want, const float* src, float* dst, int n, cudaStream_t s) {
    engram_wait_kernel<<<8, 256, 0, s>>>(flag, want, src, dst, n);
    check("engram_wait");
}

void engram_add(float* R, const float* kv, const float* qn, const float* kn, int n_embd, float eps, cudaStream_t s) {
    engram_kernel<<<4, 512, 0, s>>>(R, kv, qn, kn, n_embd, eps);
    check("engram");
}

void head_prep(const float* R, const float* pre, const float* norm_w, float eps, int n_embd, float* x, void* xq,
               cudaStream_t s) {
    head_kernel<<<1, 1024, 0, s>>>(R, pre, norm_w, eps, n_embd, x, (block_q8_1*) xq);
    check("head");
}

}  // namespace strata::kernels::dsv41
