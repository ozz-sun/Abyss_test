// src/kernels/cuda/qwen35_fast.cu - see include/strata/kernels/qwen35_fast.hpp.
#include "strata/kernels/qwen35_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace strata::kernels::q35 {
namespace {

constexpr int kD = 256;       // head_dim
constexpr int kChunk = 256;   // attention positions a block
constexpr int kWarps = 4;     // ... its warps

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }   // (ggml's)
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// a 256-thread block's sum (sh: 8 floats)
__device__ __forceinline__ float block_sum256(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = threadIdx.x < 8 ? sh[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }

__global__ void silu_kernel(float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * sigmoid_f(x[i]);
}
__global__ void scale_kernel(float* __restrict__ x, int n, float a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}
__global__ void gdn_gates_kernel(const float* __restrict__ alpha, const float* __restrict__ dt, const float* __restrict__ A,
                                 float* __restrict__ beta, float* __restrict__ gate, int h_v) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= h_v) return;
    gate[h] = softplus_f(alpha[h] + dt[h]) * A[h];
    beta[h] = sigmoid_f(beta[h]);
}
__global__ void scale_sigmoid_kernel(float* __restrict__ x, int n, const float* __restrict__ g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigmoid_f(g[0]);
}
__global__ void attn_gate_kernel(float* __restrict__ o, const float* __restrict__ qf, int n_head) {
    const int h = blockIdx.x, d = threadIdx.x;
    if (h < n_head) o[h * kD + d] *= sigmoid_f(qf[(size_t) h * 2 * kD + kD + d]);
}

// block b < n_head: q head b (its q half of q_full); b >= n_head: KV head b - n_head.  256 threads, one a dim
__global__ void __launch_bounds__(kD) attn_prep_kernel(const float* __restrict__ qf, const float* __restrict__ k,
                                                       const float* __restrict__ v, const float* __restrict__ q_norm,
                                                       const float* __restrict__ k_norm, float eps, int n_head,
                                                       int n_head_kv, int n_rot, float rope_base, int p,
                                                       uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                                       float* __restrict__ q) {
    __shared__ float sv[kD];
    __shared__ float sred[8];
    const int b = blockIdx.x, d = threadIdx.x;
    const bool is_q = b < n_head;
    const int g = is_q ? 0 : b - n_head;
    const float x = is_q ? qf[(size_t) b * 2 * kD + d] : k[(size_t) g * kD + d];
    const float ss = block_sum256(x * x, sred);
    const float inv = rsqrtf(ss / (float) kD + eps);
    const float xn = x * inv * (is_q ? q_norm[d] : k_norm[d]);
    sv[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < n_rot) {   // NEOX pairs (i, i + n_rot / 2), theta_i = p * base^(-2 i / n_rot)
        const int half = n_rot / 2;
        const int i = d < half ? d : d - half;
        const float theta = (float) p * powf(rope_base, -2.0f * (float) i / (float) n_rot);
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = sv[i], x1 = sv[i + half];
        y = d < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    if (is_q) {
        q[(size_t) b * kD + d] = y;
    } else {
        const size_t row = ((size_t) p * n_head_kv + g) * kD;
        kc[row + d] = f2h(y);
        vc[row + d] = f2h(v[(size_t) g * kD + d]);
    }
}

// grid (n_head, chunks), 128 threads: warp w takes positions t0 + w, t0 + w + 4, ..; a lane holds 8 dims of q (scaled)
// and of the V accumulator; the online softmax per warp, the 4 warps merged into the chunk's partial (max, sum, acc)
__global__ void __launch_bounds__(kWarps * 32) attn_part_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                                                                const uint16_t* __restrict__ vc, int n_head, int n_head_kv,
                                                                int n_pos, float scale, float* __restrict__ part,
                                                                int n_chunks) {
    __shared__ float sm[kWarps], sl[kWarps];
    __shared__ float sacc[kWarps][kD];
    const int h = blockIdx.x, c = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = h / (n_head / n_head_kv);
    const int t0 = c * kChunk, t1 = min(n_pos, t0 + kChunk);
    float qv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) qv[i] = q[(size_t) h * kD + lane * 8 + i] * scale;
    float m = -INFINITY, l = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    for (int t = t0 + warp; t < t1; t += kWarps) {
        const size_t row = ((size_t) t * n_head_kv + g) * kD + lane * 8;
        const uint4 kw = *(const uint4*) (kc + row);
        const uint4 vw = *(const uint4*) (vc + row);
        const uint16_t* kh = (const uint16_t*) &kw;
        const uint16_t* vh = (const uint16_t*) &vw;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) dot += qv[i] * h2f(kh[i]);
        const float sc = warp_sum(dot);
        const float mn = fmaxf(m, sc);
        const float a = expf(m - mn), pr = expf(sc - mn);
        l = l * a + pr;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = acc[i] * a + pr * h2f(vh[i]);
        m = mn;
    }
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) sacc[warp][lane * 8 + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float M = sm[0];
#pragma unroll
        for (int w = 1; w < kWarps; ++w) M = fmaxf(M, sm[w]);
        float L = 0.0f, f[kWarps];
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            f[w] = sm[w] == -INFINITY ? 0.0f : expf(sm[w] - M);
            L += sl[w] * f[w];
        }
        float* out = part + ((size_t) h * n_chunks + c) * (kD + 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float o = 0.0f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) o += sacc[w][lane * 8 + i] * f[w];
            out[lane * 8 + i] = o;
        }
        if (lane == 0) {
            out[kD] = M;
            out[kD + 1] = L;
        }
    }
}

// grid n_head, 256 threads (a dim each): the chunks' partials merged
__global__ void __launch_bounds__(kD) attn_merge_kernel(const float* __restrict__ part, int n_chunks, float* __restrict__ out) {
    const int h = blockIdx.x, d = threadIdx.x;
    const float* P = part + (size_t) h * n_chunks * (kD + 2);
    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, P[(size_t) c * (kD + 2) + kD]);
    float L = 0.0f, o = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float* pc = P + (size_t) c * (kD + 2);
        const float f = pc[kD] == -INFINITY ? 0.0f : expf(pc[kD] - M);
        L += pc[kD + 1] * f;
        o += pc[d] * f;
    }
    out[(size_t) h * kD + d] = o / L;
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "qwen35 %s: %s\n", what, cudaGetErrorString(e));
}
}  // namespace

void silu(float* x, int n, cudaStream_t s) {
    if (n <= 0) return;
    silu_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n);
    check("silu");
}
void scale(float* x, int n, float a, cudaStream_t s) {
    if (n <= 0) return;
    scale_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n, a);
    check("scale");
}
void gdn_gates(const float* alpha, const float* dt_bias, const float* ssm_a, float* beta, float* gate, int h_v,
               cudaStream_t s) {
    if (h_v <= 0) return;
    gdn_gates_kernel<<<(h_v + 127) / 128, 128, 0, s>>>(alpha, dt_bias, ssm_a, beta, gate, h_v);
    check("gdn_gates");
}
void attn_prep(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm, float eps,
               int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, int p, uint16_t* kc, uint16_t* vc,
               float* q, cudaStream_t s) {
    if (head_dim != kD) {
        std::fprintf(stderr, "qwen35 attn_prep: head_dim %d (the kernels take %d)\n", head_dim, kD);
        return;
    }
    attn_prep_kernel<<<n_head + n_head_kv, kD, 0, s>>>(q_full, k, v, q_norm, k_norm, eps, n_head, n_head_kv, n_rot,
                                                       rope_base, p, kc, vc, q);
    check("attn_prep");
}
size_t attn_part_floats(int n_head, int head_dim, int max_ctx) {
    const size_t chunks = ((size_t) max_ctx + kChunk - 1) / kChunk + 1;
    return (size_t) n_head * chunks * ((size_t) head_dim + 2);
}
void attn_decode(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int n_pos, float scale, float* part, float* out, cudaStream_t s) {
    if (head_dim != kD || n_pos <= 0 || n_head % n_head_kv != 0) {
        std::fprintf(stderr, "qwen35 attn_decode: head_dim %d, n_pos %d, heads %d / %d\n", head_dim, n_pos, n_head, n_head_kv);
        return;
    }
    const int n_chunks = (n_pos + kChunk - 1) / kChunk;
    attn_part_kernel<<<dim3((unsigned) n_head, (unsigned) n_chunks), kWarps * 32, 0, s>>>(q, kc, vc, n_head, n_head_kv,
                                                                                            n_pos, scale, part, n_chunks);
    attn_merge_kernel<<<n_head, kD, 0, s>>>(part, n_chunks, out);
    check("attn_decode");
}
void attn_gate(float* o, const float* q_full, int n_head, int head_dim, cudaStream_t s) {
    if (head_dim != kD) return;
    attn_gate_kernel<<<n_head, kD, 0, s>>>(o, q_full, n_head);
    check("attn_gate");
}
void scale_sigmoid(float* x, int n, const float* g, cudaStream_t s) {
    if (n <= 0) return;
    scale_sigmoid_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n, g);
    check("scale_sigmoid");
}

}  // namespace strata::kernels::q35
