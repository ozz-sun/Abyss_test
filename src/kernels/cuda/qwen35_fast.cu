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
__global__ void scale_sigmoid_rows_kernel(float* __restrict__ x, int n, const float* __restrict__ g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i < n) x[(size_t) t * n + i] *= sigmoid_f(g[t]);
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

// grid (n_head, rows), 256 threads (a dim each): the chunks' partials merged (part and out row-major: [row][head])
__global__ void __launch_bounds__(kD) attn_merge_kernel(const float* __restrict__ part, int n_chunks, float* __restrict__ out) {
    const int h = blockIdx.y * gridDim.x + blockIdx.x, d = threadIdx.x;
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

// ---------------------------------------------------------------- the prompt path (rows = a sub-batch's tokens)
// R[t][i] (row stride ldr) += add[t][i], x[t][i] = rms(R[t]) * w[i], x16 its FP16; a block a row
__global__ void __launch_bounds__(256) add_rms_rows_kernel(float* __restrict__ R, int ldr, const float* __restrict__ add,
                                                           const float* __restrict__ w, float eps, int n,
                                                           float* __restrict__ x, uint16_t* __restrict__ x16) {
    __shared__ float sred[8];
    const int t = blockIdx.x;
    float* r = R + (size_t) t * ldr;
    const float* a = add != nullptr ? add + (size_t) t * n : nullptr;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) {
        float v = r[i];
        if (a != nullptr) {
            v += a[i];
            r[i] = v;
        }
        ss += v * v;
    }
    ss = block_sum256(ss, sred);
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) {
        const float y = r[i] * inv * w[i];
        x[(size_t) t * n + i] = y;
        x16[(size_t) t * n + i] = f2h(y);
    }
}
// out[t][c] = silu(sum_i kW[c][i] * in(t - (dc - 1) + i, c)), the rows before the chunk from the state (cs[c][0] the
// oldest - gdn_conv_kernel's layout); grid (C / 256, T)
__global__ void __launch_bounds__(256) conv_rows_kernel(const float* __restrict__ x, int C, const float* __restrict__ kW,
                                                        int dc, const float* __restrict__ cs, float* __restrict__ out) {
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc; ++i) {
        const int tau = t - (dc - 1) + i;
        const float v = tau >= 0 ? x[(size_t) tau * C + c] : cs[(size_t) c * (dc - 1) + (tau + dc - 1)];
        acc += v * w[i];
    }
    out[(size_t) t * C + c] = acc * sigmoid_f(acc);
}
// the state after T rows: the last dc - 1 inputs (the state's own rows when T < dc - 1); dc <= 9
__global__ void __launch_bounds__(256) conv_state_rows_kernel(const float* __restrict__ x, int T, int C, int dc,
                                                              float* __restrict__ cs) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    float nv[8];
    for (int i = 0; i < dc - 1; ++i) {
        const int tau = T - (dc - 1) + i;
        nv[i] = tau >= 0 ? x[(size_t) tau * C + c] : st[tau + dc - 1];
    }
    for (int i = 0; i < dc - 1; ++i) st[i] = nv[i];
}
// row t's q head hh (hh < h_k) or k head hh - h_k, in place: x / sqrt(sum x^2 + eps) (the sum in double, as
// gdn_l2_kernel's), q also scaled; grid (T, 2 h_k), S threads
__global__ void __launch_bounds__(128) gdn_prep_rows_kernel(float* __restrict__ h, int C, int h_k, int S, float eps,
                                                            float qscale) {
    __shared__ double sh[4];
    const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    float* p = h + (size_t) t * C + (size_t) hh * S;   // the q heads, then the k heads: head hh of [q | k]
    const float v = i < S ? p[i] : 0.0f;
    double acc = (double) v * (double) v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((i & 31) == 0) sh[i >> 5] = acc;
    __syncthreads();
    const double sum = sh[0] + sh[1] + sh[2] + sh[3];
    const float inv = (float) (1.0 / sqrt(sum + (double) eps)) * (hh < h_k ? qscale : 1.0f);
    if (i < S) p[i] = v * inv;
}
__global__ void gdn_gates_rows_kernel(float* __restrict__ alpha, float* __restrict__ beta, const float* __restrict__ dt,
                                      const float* __restrict__ A, int n, int h_v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % h_v;
    alpha[i] = softplus_f(alpha[i] + dt[h]) * A[h];   // (the gate, in place)
    beta[i] = sigmoid_f(beta[i]);
}
// gdn_prep_rows_kernel's blocks (y < 2 h_k) and, past them, the gates of row t (gdn_gates_rows over its h_v heads):
// one launch for both; grid (T, 2 h_k + ceil(h_v / 128))
__global__ void __launch_bounds__(128) gdn_prep_gates_kernel(float* __restrict__ h, int C, int h_k, int S, float eps,
                                                             float qscale, float* __restrict__ alpha,
                                                             float* __restrict__ beta, const float* __restrict__ dt,
                                                             const float* __restrict__ A, int h_v) {
    __shared__ double sh[4];
    const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    if (hh >= 2 * h_k) {
        const int hv = (hh - 2 * h_k) * 128 + i;
        if (hv < h_v) {
            const size_t at = (size_t) t * h_v + hv;
            alpha[at] = softplus_f(alpha[at] + dt[hv]) * A[hv];
            beta[at] = sigmoid_f(beta[at]);
        }
        return;
    }
    float* p = h + (size_t) t * C + (size_t) hh * S;
    const float v = i < S ? p[i] : 0.0f;
    double acc = (double) v * (double) v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((i & 31) == 0) sh[i >> 5] = acc;
    __syncthreads();
    const double sum = sh[0] + sh[1] + sh[2] + sh[3];
    const float inv = (float) (1.0 / sqrt(sum + (double) eps)) * (hh < h_k ? qscale : 1.0f);
    if (i < S) p[i] = v * inv;
}
// the decode conv (gdn_conv_kernel's state slide) with the SiLU in the same launch
__global__ void conv_silu_step_kernel(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ kW,
                                      float* __restrict__ out, int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];   // kernel[0] reads the OLDEST state row
    acc += x[c] * w[dc - 1];                                // the new input lands in the LAST tap
    out[c] = acc * sigmoid_f(acc);
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];     // slide: drop the oldest, append the newest
    st[dc - 2] = x[c];
}
// the recurrence over T rows: a block a v head, a thread a state column j held in registers, the rows in order.  The
// state (S, h_v, S) and the math are gdn_step's (gdn.cu): s *= dec, d = (v - s.k) beta, s += k d, o = s.q
template <int S>
__global__ void __launch_bounds__(S) gdn_scan_kernel(float* __restrict__ state, const float* __restrict__ h, int T, int C,
                                                     const float* __restrict__ gate, const float* __restrict__ beta,
                                                     int h_k, int h_v, float* __restrict__ o) {
    __shared__ float ks[S], qs[S];
    const int hv = blockIdx.x, j = threadIdx.x, src = hv % h_k;   // (MODULO head pairing, as gdn_step)
    const size_t stride = (size_t) h_v * S;
    float* col = state + (size_t) hv * S + j;
    float st[S];
#pragma unroll
    for (int i = 0; i < S; ++i) st[i] = col[(size_t) i * stride];
    for (int t = 0; t < T; ++t) {
        const float* row = h + (size_t) t * C;
        __syncthreads();
        qs[j] = row[src * S + j];
        ks[j] = row[(h_k + src) * S + j];
        __syncthreads();
        const float dec = expf(gate[(size_t) t * h_v + hv]), b = beta[(size_t) t * h_v + hv];
        const float v = row[(2 * h_k + hv) * S + j];
        float sk = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            st[i] *= dec;
            sk += st[i] * ks[i];
        }
        const float d = (v - sk) * b;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            st[i] += ks[i] * d;
            dot += st[i] * qs[i];
        }
        o[((size_t) t * h_v + hv) * S + j] = dot;
    }
#pragma unroll
    for (int i = 0; i < S; ++i) col[(size_t) i * stride] = st[i];
}
// attn_prep_kernel over rows: grid (n_head + n_head_kv, T), row t at position p0 + t
__global__ void __launch_bounds__(kD) attn_prep_rows_kernel(const float* __restrict__ qf, const float* __restrict__ k,
                                                            const float* __restrict__ v, const float* __restrict__ q_norm,
                                                            const float* __restrict__ k_norm, float eps, int n_head,
                                                            int n_head_kv, int n_rot, float rope_base, int64_t p0,
                                                            uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                                            float* __restrict__ q) {
    __shared__ float sv[kD];
    __shared__ float sred[8];
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const bool is_q = b < n_head;
    const int g = is_q ? 0 : b - n_head;
    const int64_t p = p0 + t;
    const float* qrow = qf + (size_t) t * 2 * kD * n_head;
    const float* krow = k + (size_t) t * kD * n_head_kv;
    const float* vrow = v + (size_t) t * kD * n_head_kv;
    const float x = is_q ? qrow[(size_t) b * 2 * kD + d] : krow[(size_t) g * kD + d];
    const float ss = block_sum256(x * x, sred);
    const float inv = rsqrtf(ss / (float) kD + eps);
    const float xn = x * inv * (is_q ? q_norm[d] : k_norm[d]);
    sv[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < n_rot) {
        const int half = n_rot / 2;
        const int i = d < half ? d : d - half;
        const float theta = (float) p * powf(rope_base, -2.0f * (float) i / (float) n_rot);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = sv[i], x1 = sv[i + half];
        y = d < half ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
    if (is_q) {
        q[((size_t) t * n_head + b) * kD + d] = y;
    } else {
        const size_t row = ((size_t) p * n_head_kv + g) * kD;
        kc[row + d] = f2h(y);
        vc[row + d] = f2h(vrow[(size_t) g * kD + d]);
    }
}
__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 x = __half22float2(h[i]);
        f[2 * i] = x.x;
        f[2 * i + 1] = x.y;
    }
}
// the causal attention of 4 queries a block over one KV head's group: 8 warps x HW heads, a lane 8 dims; the keys
// stream past once, each (query, head) pair an online softmax in registers.  grid (ceil(T / 4), n_head_kv)
template <int HW>
__global__ void __launch_bounds__(256) attn_prompt_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                                                          const uint16_t* __restrict__ vc, int n_head, int n_head_kv,
                                                          int64_t p0, int T, float scale, float* __restrict__ out) {
    constexpr int QB = 4;
    const int g = blockIdx.y, t0 = blockIdx.x * QB, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = n_head / n_head_kv, h0 = g * G + warp * HW;
    const int nq = min(QB, T - t0);
    float qr[QB][HW][8], acc[QB][HW][8], m[QB][HW], l[QB][HW];
#pragma unroll
    for (int qi = 0; qi < QB; ++qi)
#pragma unroll
        for (int hi = 0; hi < HW; ++hi) {
            const float* qp = q + ((size_t) (t0 + (qi < nq ? qi : 0)) * n_head + h0 + hi) * kD + lane * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                qr[qi][hi][e] = qp[e] * scale;
                acc[qi][hi][e] = 0.0f;
            }
            m[qi][hi] = -INFINITY;
            l[qi][hi] = 0.0f;
        }
    const int64_t pend = p0 + t0 + nq;   // the keys: positions < pend, each query's own and before
    for (int64_t kp = 0; kp < pend; ++kp) {
        const size_t row = ((size_t) kp * n_head_kv + g) * kD + lane * 8;
        float kf[8], vf[8];
        unpack8(*reinterpret_cast<const uint4*>(kc + row), kf);
        unpack8(*reinterpret_cast<const uint4*>(vc + row), vf);
#pragma unroll
        for (int hi = 0; hi < HW; ++hi)
#pragma unroll
            for (int qi = 0; qi < QB; ++qi) {
                float sdot = 0.0f;
#pragma unroll
                for (int e = 0; e < 8; ++e) sdot += qr[qi][hi][e] * kf[e];
                sdot = warp_sum(sdot);
                if (qi < nq && kp <= p0 + t0 + qi) {   // (warp-uniform)
                    const float mn = fmaxf(m[qi][hi], sdot);
                    const float corr = expf(m[qi][hi] - mn), pe = expf(sdot - mn);
                    l[qi][hi] = l[qi][hi] * corr + pe;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[qi][hi][e] = acc[qi][hi][e] * corr + pe * vf[e];
                    m[qi][hi] = mn;
                }
            }
    }
#pragma unroll
    for (int qi = 0; qi < QB; ++qi) {
        if (qi >= nq) break;
#pragma unroll
        for (int hi = 0; hi < HW; ++hi) {
            float* op = out + ((size_t) (t0 + qi) * n_head + h0 + hi) * kD + lane * 8;
            const float inv = 1.0f / l[qi][hi];
#pragma unroll
            for (int e = 0; e < 8; ++e) op[e] = acc[qi][hi][e] * inv;
        }
    }
}
// a verify window's attention (T <= 8 rows): attn_prompt_kernel's block (4 queries x one KV head's group) over one
// chunk of kWinChunk key positions, its partials (max, sum, the unnormalized accumulator) into part
// [row][head][chunk] for attn_merge_kernel - the window's few rows alone gave attn_prompt 1-4 blocks, each walking
// every cached position.  grid (ceil(T / 4), n_head_kv, n_chunks)
constexpr int kWinChunk = 128;
template <int HW>
__global__ void __launch_bounds__(256) attn_window_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                                                          const uint16_t* __restrict__ vc, int n_head, int n_head_kv,
                                                          int64_t p0, int T, float scale, float* __restrict__ part,
                                                          int n_chunks) {
    constexpr int QB = 4;
    const int g = blockIdx.y, t0 = blockIdx.x * QB, c = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = n_head / n_head_kv, h0 = g * G + warp * HW;
    const int nq = min(QB, T - t0);
    float qr[QB][HW][8], acc[QB][HW][8], m[QB][HW], l[QB][HW];
#pragma unroll
    for (int qi = 0; qi < QB; ++qi)
#pragma unroll
        for (int hi = 0; hi < HW; ++hi) {
            const float* qp = q + ((size_t) (t0 + (qi < nq ? qi : 0)) * n_head + h0 + hi) * kD + lane * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                qr[qi][hi][e] = qp[e] * scale;
                acc[qi][hi][e] = 0.0f;
            }
            m[qi][hi] = -INFINITY;
            l[qi][hi] = 0.0f;
        }
    const int64_t k0 = (int64_t) c * kWinChunk, k1 = min(k0 + kWinChunk, p0 + t0 + nq);
    for (int64_t kp = k0; kp < k1; ++kp) {
        const size_t row = ((size_t) kp * n_head_kv + g) * kD + lane * 8;
        float kf[8], vf[8];
        unpack8(*reinterpret_cast<const uint4*>(kc + row), kf);
        unpack8(*reinterpret_cast<const uint4*>(vc + row), vf);
#pragma unroll
        for (int hi = 0; hi < HW; ++hi)
#pragma unroll
            for (int qi = 0; qi < QB; ++qi) {
                float sdot = 0.0f;
#pragma unroll
                for (int e = 0; e < 8; ++e) sdot += qr[qi][hi][e] * kf[e];
                sdot = warp_sum(sdot);
                if (qi < nq && kp <= p0 + t0 + qi) {   // (warp-uniform)
                    const float mn = fmaxf(m[qi][hi], sdot);
                    const float corr = expf(m[qi][hi] - mn), pe = expf(sdot - mn);
                    l[qi][hi] = l[qi][hi] * corr + pe;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[qi][hi][e] = acc[qi][hi][e] * corr + pe * vf[e];
                    m[qi][hi] = mn;
                }
            }
    }
#pragma unroll
    for (int qi = 0; qi < QB; ++qi) {
        if (qi >= nq) break;
#pragma unroll
        for (int hi = 0; hi < HW; ++hi) {
            float* op = part + (((size_t) (t0 + qi) * n_head + h0 + hi) * n_chunks + c) * (kD + 2);
#pragma unroll
            for (int e = 0; e < 8; ++e) op[lane * 8 + e] = acc[qi][hi][e];
            if (lane == 0) {
                op[kD] = m[qi][hi];
                op[kD + 1] = l[qi][hi];
            }
        }
    }
}
__global__ void attn_gate_rows_kernel(float* __restrict__ o, const float* __restrict__ qf, int n_head) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    o[((size_t) t * n_head + h) * kD + d] *= sigmoid_f(qf[((size_t) t * n_head + h) * 2 * kD + kD + d]);
}
// out[0] = exp(l[tok] - max) / sum exp(l - max): one block of 256
__global__ void __launch_bounds__(256) top_prob_kernel(const float* __restrict__ l, int n, const int* __restrict__ tok,
                                                       float* __restrict__ out) {
    __shared__ float sred[8];
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 256) m = fmaxf(m, l[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) sred[threadIdx.x >> 5] = m;
    __syncthreads();
    m = sred[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) m = fmaxf(m, sred[w]);
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) sum += expf(l[i] - m);
    sum = block_sum256(sum, sred);
    if (threadIdx.x == 0) out[0] = expf(l[tok[0]] - m) / sum;
}
// ffn[t] *= sigmoid(x[t] . w): a block a row
__global__ void __launch_bounds__(256) shexp_gate_rows_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                              float* __restrict__ ffn, int n) {
    __shared__ float sred[8];
    const int t = blockIdx.x;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) dot += x[(size_t) t * n + i] * w[i];
    dot = block_sum256(dot, sred);
    const float gs = sigmoid_f(dot);
    for (int i = threadIdx.x; i < n; i += 256) ffn[(size_t) t * n + i] *= gs;
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
void scale_sigmoid_rows(float* x, int T, int n, const float* g, cudaStream_t s) {
    if (n <= 0 || T <= 0) return;
    scale_sigmoid_rows_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(x, n, g);
    check("scale_sigmoid_rows");
}

// ---- the prompt path
void add_rms_rows(float* R, int ldr, const float* add, const float* w, float eps, int T, int n, float* x, void* x16,
                  cudaStream_t s) {
    if (T <= 0) return;
    add_rms_rows_kernel<<<T, 256, 0, s>>>(R, ldr, add, w, eps, n, x, (uint16_t*) x16);
    check("add_rms_rows");
}
void conv_rows(const float* x, int T, int C, const float* kW, int d_conv, float* conv_state, float* out,
               cudaStream_t s) {
    if (T <= 0 || C <= 0) return;
    if (d_conv < 2 || d_conv > 9) {
        std::fprintf(stderr, "qwen35 conv_rows: d_conv %d\n", d_conv);
        return;
    }
    conv_rows_kernel<<<dim3((unsigned) ((C + 255) / 256), (unsigned) T), 256, 0, s>>>(x, C, kW, d_conv, conv_state, out);
    conv_state_rows_kernel<<<(C + 255) / 256, 256, 0, s>>>(x, T, C, d_conv, conv_state);
    check("conv_rows");
}
void gdn_prep_rows(float* h, int T, int C, int h_k, int S, float eps, cudaStream_t s) {
    if (T <= 0 || h_k <= 0) return;
    if (S <= 0 || S > 128) {
        std::fprintf(stderr, "qwen35 gdn_prep_rows: S %d\n", S);
        return;
    }
    gdn_prep_rows_kernel<<<dim3((unsigned) T, (unsigned) (2 * h_k)), 128, 0, s>>>(h, C, h_k, S, eps,
                                                                                   1.0f / std::sqrt((float) S));
    check("gdn_prep_rows");
}
void gdn_prep_gates(float* h, int T, int C, int h_k, int S, float eps, float* alpha, float* beta, const float* dt_bias,
                    const float* ssm_a, int h_v, cudaStream_t s) {
    if (T <= 0 || h_k <= 0) return;
    if (S <= 0 || S > 128) {
        std::fprintf(stderr, "qwen35 gdn_prep_gates: S %d\n", S);
        return;
    }
    const unsigned gb = (unsigned) ((h_v + 127) / 128);
    gdn_prep_gates_kernel<<<dim3((unsigned) T, (unsigned) (2 * h_k) + gb), 128, 0, s>>>(
        h, C, h_k, S, eps, 1.0f / std::sqrt((float) S), alpha, beta, dt_bias, ssm_a, h_v);
    check("gdn_prep_gates");
}
void conv_state_rows(const float* x, int T, int C, int d_conv, float* conv_state, cudaStream_t s) {
    if (T <= 0 || C <= 0 || d_conv < 2 || d_conv > 9) return;
    conv_state_rows_kernel<<<(C + 255) / 256, 256, 0, s>>>(x, T, C, d_conv, conv_state);
    check("conv_state_rows");
}
void conv_silu_step(float* conv_state, const float* x, const float* kW, int C, int d_conv, float* out, cudaStream_t s) {
    if (C <= 0 || d_conv < 2) return;
    conv_silu_step_kernel<<<(C + 255) / 256, 256, 0, s>>>(conv_state, x, kW, out, C, d_conv);
    check("conv_silu_step");
}
void gdn_gates_rows(float* alpha, float* beta, const float* dt_bias, const float* ssm_a, int T, int h_v, cudaStream_t s) {
    const int n = T * h_v;
    if (n <= 0) return;
    gdn_gates_rows_kernel<<<(n + 255) / 256, 256, 0, s>>>(alpha, beta, dt_bias, ssm_a, n, h_v);
    check("gdn_gates_rows");
}
void gdn_scan(float* state, const float* h, int T, int C, const float* gate, const float* beta, int S, int h_k, int h_v,
              float* o, cudaStream_t s) {
    if (T <= 0 || h_v <= 0) return;
    if (S != 128 || h_k <= 0 || h_v % h_k != 0) {
        std::fprintf(stderr, "qwen35 gdn_scan: S %d, heads %d / %d\n", S, h_k, h_v);
        return;
    }
    gdn_scan_kernel<128><<<h_v, 128, 0, s>>>(state, h, T, C, gate, beta, h_k, h_v, o);
    check("gdn_scan");
}
void attn_prep_rows(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm,
                    float eps, int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, int64_t p0, int T,
                    uint16_t* kc, uint16_t* vc, float* q, cudaStream_t s) {
    if (T <= 0) return;
    if (head_dim != kD) {
        std::fprintf(stderr, "qwen35 attn_prep_rows: head_dim %d (the kernels take %d)\n", head_dim, kD);
        return;
    }
    attn_prep_rows_kernel<<<dim3((unsigned) (n_head + n_head_kv), (unsigned) T), kD, 0, s>>>(
        q_full, k, v, q_norm, k_norm, eps, n_head, n_head_kv, n_rot, rope_base, p0, kc, vc, q);
    check("attn_prep_rows");
}
void attn_prompt(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int64_t p0, int T, float scale, float* out, cudaStream_t s) {
    if (T <= 0) return;
    const int G = n_head_kv > 0 ? n_head / n_head_kv : 0;
    if (head_dim != kD || G * n_head_kv != n_head || (G != 8 && G != 16 && G != 32)) {
        std::fprintf(stderr, "qwen35 attn_prompt: head_dim %d, heads %d / %d (a group of 8, 16 or 32)\n", head_dim,
                     n_head, n_head_kv);
        return;
    }
    const dim3 grid((unsigned) ((T + 3) / 4), (unsigned) n_head_kv);
    if (G == 8) attn_prompt_kernel<1><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, out);
    else if (G == 16) attn_prompt_kernel<2><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, out);
    else attn_prompt_kernel<4><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, out);
    check("attn_prompt");
}
size_t attn_window_floats(int n_head, int head_dim, int max_ctx, int max_rows) {
    const size_t chunks = ((size_t) max_ctx + kWinChunk - 1) / kWinChunk + 1;
    return (size_t) max_rows * n_head * chunks * ((size_t) head_dim + 2);
}
void attn_window(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int64_t p0, int T, float scale, float* part, float* out, cudaStream_t s) {
    if (T <= 0) return;
    const int G = n_head_kv > 0 ? n_head / n_head_kv : 0;
    if (head_dim != kD || G * n_head_kv != n_head || (G != 8 && G != 16 && G != 32)) {
        std::fprintf(stderr, "qwen35 attn_window: head_dim %d, heads %d / %d (a group of 8, 16 or 32)\n", head_dim,
                     n_head, n_head_kv);
        return;
    }
    const int n_chunks = (int) ((p0 + T + kWinChunk - 1) / kWinChunk);
    const dim3 grid((unsigned) ((T + 3) / 4), (unsigned) n_head_kv, (unsigned) n_chunks);
    if (G == 8) attn_window_kernel<1><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks);
    else if (G == 16) attn_window_kernel<2><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks);
    else attn_window_kernel<4><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks);
    attn_merge_kernel<<<dim3((unsigned) n_head, (unsigned) T), kD, 0, s>>>(part, n_chunks, out);
    check("attn_window");
}
void attn_gate_rows(float* o, const float* q_full, int T, int n_head, int head_dim, cudaStream_t s) {
    if (T <= 0 || head_dim != kD) return;
    attn_gate_rows_kernel<<<dim3((unsigned) n_head, (unsigned) T), kD, 0, s>>>(o, q_full, n_head);
    check("attn_gate_rows");
}
void attn_kv_rows(const float* k, const float* v, const float* k_norm, float eps, int n_head_kv, int head_dim, int n_rot,
                  float rope_base, int64_t p0, int T, uint16_t* kc, uint16_t* vc, cudaStream_t s) {
    if (T <= 0 || head_dim != kD) return;
    // n_head 0: every block a KV head (the q blocks come first in attn_prep_rows_kernel's numbering)
    attn_prep_rows_kernel<<<dim3((unsigned) n_head_kv, (unsigned) T), kD, 0, s>>>(k, k, v, k_norm, k_norm, eps, 0,
                                                                                n_head_kv, n_rot, rope_base, p0, kc, vc,
                                                                                nullptr);
    check("attn_kv_rows");
}
void top_prob(const float* logits, int n, const int* tok, float* out, cudaStream_t s) {
    if (n <= 0) return;
    top_prob_kernel<<<1, 256, 0, s>>>(logits, n, tok, out);
    check("top_prob");
}
void shexp_gate_rows(const float* x, const float* w, float* ffn, int T, int n, cudaStream_t s) {
    if (T <= 0) return;
    shexp_gate_rows_kernel<<<T, 256, 0, s>>>(x, w, ffn, n);
    check("shexp_gate_rows");
}

}  // namespace strata::kernels::q35
