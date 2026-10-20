// src/igpu/igpu_fused.cu - see igpu_fused.h.  From Maya's src/prefill/moe_fused_rdna4.cu (mw00/project-maya, MIT;
// adapted from Niko1221/Strata fb58e0d and PR #1570, MIT): the IQ blocks decoded with GGML's codebooks into signed
// int8 and their scales (LDS, double-buffered), int32 dots on the matrix cores (v_wmma_i32_16x16x16_iu8, wave32), the
// MAGIC bias turning each dot into an exact FP32 before the scales.  Activations: signed int8 codes and FP32 scales per
// 32 values (80 bytes per 64 values, natural order).  Changes for Abyss's iGPU lane: the experts' blobs come from a
// pointer table (registered host memory, read in place) instead of one slot array with a stride; the tiles are made
// on the host; no Q2_0 (not a ggml format here); numeric type ids (no ggml.h in this build).
#include "igpu_fused.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace abyss::igpu_fused {
namespace {

constexpr int kTileRows = 64;
constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1}, 8 unused
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
constexpr int WLD = 80;                 // 64 decoded int8 weights plus padding to avoid LDS bank conflicts
constexpr int GU_ROWS_K = 4096, D_ROWS_K = 2048;   // K of gate/up (n_embd) and of down (n_ff)

// the formats (ggml type ids)
constexpr int T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23;
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 &&
              sizeof(block_iq4_nl) == 18,
              "the block layouts this file decodes");

// block bytes, scale per 16 values, codebook bytes in shared memory
__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110
         : t == T_IQ4_XS ? 136 : 18;
}
__host__ __device__ constexpr int block_values(int t) { return t == T_IQ4_NL ? 32 : 256; }
// These covered formats have no additive minimum and need at most five raw words.
__host__ __device__ constexpr bool has_min(int) { return false; }
__host__ __device__ constexpr int raw_words(int) { return 5; }
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4
         : t == T_IQ3_S ? 512 * 4 : 0;
}
__host__ __device__ constexpr bool gu_ok(int t) {
    return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS || t == T_IQ3_S || t == T_IQ4_XS;
}
__host__ __device__ constexpr bool dn_ok(int t) { return t == T_IQ4_NL || t == T_IQ2_S || t == T_IQ3_XXS; }

// ---- activations in natural order: one warp per 64 values
__global__ void quant_act_nat_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o, 32));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o, 32));
    }
    uint8_t* out = xa + w * AB;
    const int c0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0, c1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    out[lane] = (uint8_t) (int8_t) c0;
    out[32 + lane] = (uint8_t) (int8_t) c1;
    int s0 = c0, s1 = c1;   // the codes' sums per 32 values (the formats with a minimum read them)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o, 32);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o, 32);
    }
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, (float) s0, (float) s1);
}

__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
// ---- the load stage: a 32-value sub-block's bytes into registers, then int8 and scales
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// gfx11: the signs from a shared-memory table.  An entry (one per sign byte b; for the 7-bit sign formats per 7 bits,
// the parity bit folded in) is {m0, c0, m1, c1}: m = 0xFF in the bytes whose sign bit is set (bits 0-3 for .x, 4-7 for
// .y), c = m & 0x01010101.  q = (g ^ m) + c is -g where the sign is set - the two's complement, byte-wise - and g
// elsewhere; no byte carries into the next, because no codebook entry has a zero byte (the grids' bytes are 1..62).
__device__ __forceinline__ void signed8t(uint32_t gx, uint32_t gy, const uint4 sg, uint32_t& qx, uint32_t& qy) {
    qx = (gx ^ sg.x) + sg.y;
    qy = (gy ^ sg.z) + sg.w;
}
__host__ __device__ constexpr int sign_entries(int t) { return t == T_IQ2_S || t == T_IQ3_S ? 256 : 128; }
__device__ __forceinline__ uint4 sign_entry(int i, bool parity) {
    uint32_t b = (uint32_t) i;
    if (parity) b ^= (__popc(b) & 1) << 7;
    const uint32_t m0 = __vcmpne4(((b & 15) * 0x01010101u) & 0x08040201u, 0),
                   m1 = __vcmpne4(((b >> 4) * 0x01010101u) & 0x08040201u, 0);
    return make_uint4(m0, m0 & 0x01010101u, m1, m1 & 0x01010101u);
}
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in .x, the high ones' in .y
// (direct-selector form on gfx: p = low 3 bits of each nibble picks within the table half, bit 3 picks the half)
__device__ __forceinline__ uint32_t table16_one(uint32_t x, const uint32_t (&t)[4]) {
    const uint32_t p = x & 0x07070707u;
    const uint32_t a = __builtin_amdgcn_perm(t[1], t[0], p), b = __builtin_amdgcn_perm(t[3], t[2], p);
    return __builtin_amdgcn_perm(b, a, ((x >> 1) & 0x04040404u) | 0x03020100u);
}
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    lo = table16_one(q4, t);
    hi = table16_one(q4 >> 4, t);
}

// Raw bytes of sub-block `ib` of the block at `bp` (IQ: the 256-value super-block, ib 0..7; IQ4_NL: the 32-value block).
template <int T, int NW> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[NW]) {
    static_assert(NW >= raw_words(T), "raw words");
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 8 + 16 * ib + 4 * k);
        const uint32_t ls = ((bp[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((ld16(bp + 2) >> (2 * ib)) & 3) << 4);
        w[4] = ld16(bp) | (ls << 16);
    } else {   // IQ4_NL
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    }
}

// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T, int NW>
__device__ __forceinline__ void convert(const uint32_t (&w)[NW], const uint8_t* grid, const uint32_t (&kv)[4],
                                        uint32_t (&q)[8], float& s0, float& s1, const uint4* sgn) {
    if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
            signed8t(e.x, e.y, sgn[(w[1] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
            signed8t(e.x, e.y, sgn[(c >> 9) & 127], q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
            signed8t(e.x, e.y, sgn[(w[1] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8t(g[i0], g[i1], sgn[(w[2] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8t(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                     sgn[(w[2] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else {   // IQ4_XS, IQ4_NL
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = T == T_IQ4_XS ? half_at(w[4]) * (float) ((int) (w[4] >> 16) - 32) : half_at(w[4]);
    }
}

template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else return nullptr;
}

// ---- the fused experts on gfx11 (RDNA3 / RDNA3.5) matrix cores, v_wmma_i32_16x16x16_iu8 (wave32).  A 32-value
// sub-block decoded to int8 and its scales; per 32 values the int32 dot starts from MAGIC (the C operand), so
// as_float(d) - 1.5 * 2^23 is the dot; the formats with a scale per 16 values take each 16-value k-step's dot alone.
// Fragments (gfx11): A lane l holds the 16 k of row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l the 16 k of
// column l % 16, C/D lane l holds D[2i + l / 16][l % 16].  Weights are A (LDS, double-buffered) and activations B
// (read by each lane from memory, a stage ahead): lane l owns token column l % 16, element i weight row 2i + l / 16.
// A work item = NW_ROWS weight rows x a 64-row tile; 8 waves: 4 along the weight rows (32 each) x 2 along the tile
// (32 each), each 2 x 2 WMMA tiles.  Gate/up: local row r is feature r / 2, its gate (r even) or up (r odd) row, so a
// lane pair l, l + 16 holds gate and up of one feature.
// gfx12 (RDNA4, wave32) has the same instruction with another layout: A lane l holds A[l % 16][8 (l / 16) + j], B lane
// l B[8 (l / 16) + j][l % 16] (j = 0..7), C/D lane l holds D[8 (l / 16) + i][l % 16]; a feature's gate and up sit in
// one lane (elements 2j, 2j + 1).  NW_DROW is the D row of a lane's element i.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
#define ABYSS_NAT_W11 1
#else
#define ABYSS_NAT_W11 0
#endif
#if defined(__gfx1200__) || defined(__gfx1201__)
#define ABYSS_NAT_W12 1
#else
#define ABYSS_NAT_W12 0
#endif
#define ABYSS_NAT_WMMA (ABYSS_NAT_W11 || ABYSS_NAT_W12)

// a barrier over LDS only (Strata PR #3222): __syncthreads() is a fence over all memory, which on gfx11 / gfx12 also
// waits for every global load in flight (gfx12 also drops the L0 cache) - the stage loop's prefetched weights and
// activations would land before each barrier instead of under the stage's WMMAs.  The kernel's barriers order its LDS
// (wt, ws, srow, hs) only
__device__ __forceinline__ void nw_sync_lds() {
#if ABYSS_NAT_W12
    __asm__ volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
#elif ABYSS_NAT_W11
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();
#endif
}

#if ABYSS_NAT_W12
#define NW_DROW(i, hi) (8 * (hi) + (i))
#else
#define NW_DROW(i, hi) (2 * (i) + (hi))
#endif
typedef int nw_i4 __attribute__((ext_vector_type(4)));
typedef int nw_i2 __attribute__((ext_vector_type(2)));
typedef int nw_i8 __attribute__((ext_vector_type(8)));
constexpr int NW_ROWS = 128;
constexpr int NW_THREADS = 256;

// one A / B fragment: gfx11 16 int8 per lane (uint4), gfx12 8 (uint2)
#if ABYSS_NAT_W12
typedef uint2 nw_frag;
typedef nw_i2 nw_ab;
__device__ __forceinline__ nw_ab nw_v(uint2 v) { return nw_ab{(int) v.x, (int) v.y}; }
#else
typedef uint4 nw_frag;
typedef nw_i4 nw_ab;
__device__ __forceinline__ nw_ab nw_v(uint4 v) { return nw_ab{(int) v.x, (int) v.y, (int) v.z, (int) v.w}; }
#endif
__device__ __forceinline__ nw_i8 nw_wmma(nw_ab a, nw_ab b, nw_i8 c) {
#if ABYSS_NAT_W11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#elif ABYSS_NAT_W12
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}

// GU: gate/up + clipped SwiGLU -> int8 H (out); !GU: down -> FP32 rows (dm).  The H tile lives on the weight buffers
// (one extra barrier).  blobs[e]: expert e's blob (gate rows, up rows at geo.up_off, down rows at geo.down_off).
template <int WT, bool GU>
__global__ void __launch_bounds__(NW_THREADS) __attribute__((amdgpu_waves_per_eu(1)))
native_kernel(const unsigned long long* __restrict__ blobs, int ntiles, const int2* __restrict__ tiles,
              const int* __restrict__ bounds, const Geometry geo, const uint8_t* __restrict__ act,
              const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if ABYSS_NAT_WMMA
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;  // 64-value stages along K
    constexpr int NFB = 4096 / NW_ROWS;     // work items per tile
    constexpr int ACT_LD = NS * AB;
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    constexpr int GB = grid_bytes(WT) > 0 ? grid_bytes(WT) : 16;
    __shared__ __align__(16) uint8_t wtbuf[2 * NW_ROWS * WLD];
    uint8_t(*wt)[NW_ROWS][WLD] = reinterpret_cast<uint8_t(*)[NW_ROWS][WLD]>(wtbuf);
    __shared__ __align__(16) float ws[2][NW_ROWS][4];           // their scales per 16 values
    __shared__ __align__(16) uint8_t sgrid[GB];
    constexpr int SGN = grid_bytes(WT) > 0 ? sign_entries(WT) : 1;
    __shared__ __align__(16) uint4 ssign[SGN];                   // the signs of a sign byte, as byte masks
    __shared__ int srow[kTileRows];
    static_assert((size_t) kTileRows * 65 * 4 <= (size_t) 2 * NW_ROWS * WLD, "H tile must fit the weight buffers");
    float(*hs)[65] = reinterpret_cast<float(*)[65]>(wtbuf);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += NW_THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    if constexpr (SGN > 1) {
        for (int i = tid; i < SGN; i += NW_THREADS) ssign[i] = sign_entry(i, SGN == 128);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    const int ur = tid >> 1, uj = tid & 1;                       // this thread's decode unit: row ur, sub-block uj
    const int nwork = ntiles * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tile = tiles[w / NFB];
        const int fb = w % NFB, e = tile.x, row0 = tile.y;
        const int nrows = min(kTileRows, bounds[e + 1] - bounds[0] - row0);
        if (nrows <= 0) continue;   // uniform for the block; no barriers are skipped by individual lanes
        const uint8_t* blob = (const uint8_t*) blobs[e];
        const int rbase = fb * NW_ROWS;
        const uint8_t* wrow = GU ? blob + ((ur & 1) ? geo.up_off : 0) + (size_t) (fb * (NW_ROWS / 2) + (ur >> 1)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* {
            if (GU || WT == T_IQ2_S || WT == T_IQ3_XXS) return wrow + (s >> 2) * BS;
            return (WT == T_IQ4_NL) ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return (GU || WT == T_IQ2_S || WT == T_IQ3_XXS) ? 2 * (s & 3) + uj : uj; };
        auto put = [&](const uint32_t (&raw)[raw_words(WT)], int buf) {
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1, ssign);
            uint4* d = (uint4*) &wt[buf][ur][32 * uj];
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) &ws[buf][ur][2 * uj] = make_float2(s0, s1);
        };
        nw_sync_lds();                                            // the previous item is done with the buffers
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);             // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        uint32_t raw[raw_words(WT)] = {};
        load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (NS > 1) load_unit<WT>(unit(1), sub(1), raw);
        nw_sync_lds();                                            // srow, and stage 0's weights in buffer 0
        const bool on = 32 * wn < nrows;
        const uint8_t* brow[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) brow[nt] = act + (size_t) srow[32 * wn + 16 * nt + l16] * ACT_LD;
        float acc[2][2][8];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][nt][i] = 0.0f;
        nw_frag bq[2][4];        // B per 16-value k-step: bq[nt][2 h + kk] = k 32 h + 16 kk .. (gfx12: this lane's 8 of them)
        float2 bx[2], bsum[2];   // the activation scales, and (the formats with a minimum) the codes' sums per 32 values
        auto fetch_b = [&](int s, nw_frag (&q)[2][4], float2 (&x)[2], float2 (&sm)[2]) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#if ABYSS_NAT_W12
                const uint2* p = reinterpret_cast<const uint2*>(brow[nt] + s * AB + 8 * hi);   // k 16 j + 8 hi .. + 7
                q[nt][0] = p[0]; q[nt][1] = p[2]; q[nt][2] = p[4]; q[nt][3] = p[6];
#else
                const uint4* p = reinterpret_cast<const uint4*>(brow[nt] + s * AB);
                q[nt][0] = p[0]; q[nt][1] = p[1]; q[nt][2] = p[2]; q[nt][3] = p[3];
#endif
                x[nt] = *reinterpret_cast<const float2*>(brow[nt] + s * AB + 64);
                if constexpr (has_min(WT)) sm[nt] = *reinterpret_cast<const float2*>(brow[nt] + s * AB + 72);
                else sm[nt] = make_float2(0.0f, 0.0f);
            }
        };
        if (on) fetch_b(0, bq, bx, bsum);
        for (int s = 0; s < NS; ++s) {
            // (stage s's weights are in buffer s & 1: stage 0's by the barrier above, the next ones' by the barrier at
            // the end of the stage before - no barrier here, which only made the prefetch below wait (Strata #3222))
            nw_frag nbq[2][4];
            float2 nbx[2], nbsum[2];
            if (on && s + 1 < NS) fetch_b(s + 1, nbq, nbx, nbsum);
            if (on) {
                const int bf = s & 1;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int rb = 32 * wm + 16 * mt;
#if ABYSS_NAT_W12
                        const uint2* ap = reinterpret_cast<const uint2*>(&wt[bf][rb + l16][32 * h + 8 * hi]);
                        const nw_ab A0 = nw_v(ap[0]), A1 = nw_v(ap[2]);
#else
                        const uint4* ap = reinterpret_cast<const uint4*>(&wt[bf][rb + l16][32 * h]);
                        const nw_ab A0 = nw_v(ap[0]), A1 = nw_v(ap[1]);
#endif
                        float w0[8], w1[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float2 sw = *reinterpret_cast<const float2*>(&ws[bf][rb + NW_DROW(i, hi)][2 * h]);
                            w0[i] = sw.x; w1[i] = sw.y;
                        }
#pragma unroll
                        for (int nt = 0; nt < 2; ++nt) {
                            const float dx = h ? bx[nt].y : bx[nt].x;
                            const nw_ab B0 = nw_v(bq[nt][2 * h]), B1 = nw_v(bq[nt][2 * h + 1]);
                            const nw_i8 m = nw_i8{MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC};
                            if constexpr (K16) {
                                const nw_i8 d0 = nw_wmma(A0, B0, m), d1 = nw_wmma(A1, B1, m);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    const float v = fmaf(w1[i], dotf(d1[i]), w0[i] * dotf(d0[i]));
                                    acc[mt][nt][i] = fmaf(dx, v, acc[mt][nt][i]);
                                }
                            } else {
                                const nw_i8 d = nw_wmma(A1, B1, nw_wmma(A0, B0, m));
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    acc[mt][nt][i] = fmaf(w0[i] * dx, dotf(d[i]), acc[mt][nt][i]);
                                    if constexpr (has_min(WT)) acc[mt][nt][i] = fmaf(w1[i] * dx, h ? bsum[nt].y : bsum[nt].x, acc[mt][nt][i]);
                                }
                            }
                        }
                    }
                }
            }
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);                            // the other buffer: its readers passed this barrier
                nw_sync_lds();                                    // every thread's writes are in LDS before the next stage reads them
                if (s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
                if (on) {
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        bx[nt] = nbx[nt];
                        bsum[nt] = nbsum[nt];
#pragma unroll
                        for (int j = 0; j < 4; ++j) bq[nt][j] = nbq[nt][j];
                    }
                }
            }
        }
        if constexpr (GU) {
            nw_sync_lds();                                        // every wave is done reading the weight buffers (hs aliases them)
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
#if ABYSS_NAT_W12
                        // lane l: local rows rb + 8 hi + i = features 16 wm + 8 mt + 4 hi + j, gate (i = 2j) and up (2j + 1)
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float gt = fminf(acc[mt][nt][2 * j], geo.limit);
                            const float up = fminf(fmaxf(acc[mt][nt][2 * j + 1], -geo.limit), geo.limit);
                            hs[32 * wn + 16 * nt + l16][16 * wm + 8 * mt + 4 * hi + j] = gt / (1.0f + __expf(-gt)) * up;
                        }
#else
                        // even / odd weight rows put gate / up in paired lanes; the clamp as rows_swiglu_q8_kernel's
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float raw_up = __shfl_xor_sync(0xffffffffu, acc[mt][nt][i], 16, 32);
                            const float up = fminf(fmaxf(raw_up, -geo.limit), geo.limit);
                            const float gt = fminf(acc[mt][nt][i], geo.limit);
                            if (hi == 0) hs[32 * wn + 16 * nt + l16][16 * wm + 8 * mt + i] = gt / (1.0f + __expf(-gt)) * up;
                        }
#endif
                    }
            }
            nw_sync_lds();
            // H block fb (64 features) to int8 per 32: one thread per tile row and half
            if (tid < 2 * kTileRows) {
                const int r = tid >> 1, hh = tid & 1;
                if (r < nrows) {
                    float am = 0.0f;
#pragma unroll 8
                    for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(hs[r][32 * hh + j]));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    uint8_t* o = out + (size_t) (row0 + r) * (32 * AB) + (size_t) fb * AB;
                    uint32_t wd[8] = {};
                    int csum = 0;
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const int c = __float2int_rn(hs[r][32 * hh + j] * inv);
                        csum += c;
                        wd[j >> 2] |= (uint32_t) (uint8_t) (int8_t) c << (8 * (j & 3));
                    }
                    uint4* o4 = reinterpret_cast<uint4*>(o + 32 * hh);
                    o4[0] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                    o4[1] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
                    *reinterpret_cast<float*>(o + 64 + 4 * hh) = am / 127.0f;
                    *reinterpret_cast<float*>(o + 72 + 4 * hh) = (float) csum;   // (the formats with a minimum)
                }
            }
        } else {
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const int r = 32 * wn + 16 * nt + l16;
                        if (r < nrows) {
#if ABYSS_NAT_W12
                            float4* d = reinterpret_cast<float4*>(dm + (size_t) (row0 + r) * 4096 + rbase + 32 * wm + 16 * mt + 8 * hi);
                            d[0] = make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3]);
                            d[1] = make_float4(acc[mt][nt][4], acc[mt][nt][5], acc[mt][nt][6], acc[mt][nt][7]);
#else
                            float* d = dm + (size_t) (row0 + r) * 4096 + rbase + 32 * wm + 16 * mt + hi;
#pragma unroll
                            for (int i = 0; i < 8; ++i) d[2 * i] = acc[mt][nt][i];
#endif
                        }
                    }
            }
        }
    }
#else
    __builtin_trap();
#endif
}

struct DeviceInfo { bool ready = false, ok = false; int cus = 0; };
DeviceInfo device_info(int dev) {
    static DeviceInfo devices[32];
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    if (dev < 0 || dev >= 32) return {};
    auto& d = devices[dev];
    if (d.ready) return d;
    d.ready = true;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return d;
    // other architectures keep the row kernels until measured and validated
    if (std::strncmp(prop.gcnArchName, "gfx1151", 7) != 0) return d;
    hipFuncAttributes fa{};
    if (hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(native_kernel<T_IQ3_XXS, true>)) != hipSuccess) {
        (void) hipGetLastError();
        return d;
    }
    d.ok = fa.sharedSizeBytes >= (size_t) 2 * NW_ROWS * WLD;
    if (hipDeviceGetAttribute(&d.cus, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess) d.ok = false;
    return d;
}

}  // namespace

bool available(int dev) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GLM_IGPU_FUSED");
        return !(v != nullptr && v[0] == '0');
    }();
    if (!on) return false;
    return device_info(dev).ok;
}

bool supported(int gu_type, int d_type, int n_embd, int n_ff) {
    return gu_ok(gu_type) && dn_ok(d_type) && n_embd == GU_ROWS_K && n_ff == D_ROWS_K;
}

size_t act_bytes(size_t rows, int cols) { return rows * (size_t) (cols / 64) * AB; }

size_t row_bytes(int type, int k) {
    if (!gu_ok(type) && !dn_ok(type)) return 0;
    return (size_t) (k / block_values(type)) * (size_t) block_bytes(type);
}

void quantize(const float* x, size_t rows, int cols, void* xa, hipStream_t s) {
    if (rows == 0) return;
    const int64_t nblk = (int64_t) rows * (cols / 64);
    quant_act_nat_kernel<<<(unsigned) ((nblk + 7) / 8), 256, 0, s>>>(x, nblk, (uint8_t*) xa);
}

int make_tiles(const int* bounds, int ne, int2* tiles, int cap) {
    int n = 0;
    for (int e = 0; e < ne; ++e) {
        const int rows = bounds[e + 1] - bounds[e];
        for (int r = 0; r < rows; r += kTileRows) {
            if (n >= cap) return -1;
            tiles[n++] = make_int2(e, bounds[e] - bounds[0] + r);
        }
    }
    return n;
}

bool experts(const unsigned long long* blobs, int ne, const int* bounds, const int2* tiles, int ntiles,
             const Geometry& geo, const void* xa, const int* src, void* ha, float* out, hipStream_t s) {
    if (ne <= 0 || ntiles <= 0) return true;
    if (!gu_ok(geo.gu_type) || !dn_ok(geo.d_type) || ((uintptr_t) out & 15)) return false;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return false;
    const auto d = device_info(dev);
    if (!d.ok) return false;
    // blocks per CU (Maya's: 3 for the IQ2_S / IQ2_XS gate/up, else 4); ABYSS_IGPU_FUSED_GRID=<n> sets both
    static const int grid_env = [] { const char* v = std::getenv("ABYSS_IGPU_FUSED_GRID"); return v != nullptr ? std::atoi(v) : 0; }();
    const int gu_blocks = grid_env > 0 ? grid_env : geo.gu_type == T_IQ2_S || geo.gu_type == T_IQ2_XS ? 3 : 4;
    const int dn_blocks = grid_env > 0 ? grid_env : geo.d_type == T_IQ2_S ? 3 : 4;
    const unsigned grid_gu = (unsigned) std::min(ntiles * 32, d.cus * gu_blocks);
    const unsigned grid_dn = (unsigned) std::min(ntiles * 32, d.cus * dn_blocks);
#define GU(T) native_kernel<T, true><<<grid_gu, NW_THREADS, 0, s>>>(blobs, ntiles, tiles, bounds, geo, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr)
    switch (geo.gu_type) {
        case T_IQ2_XXS: GU(T_IQ2_XXS); break;
        case T_IQ2_XS: GU(T_IQ2_XS); break;
        case T_IQ2_S: GU(T_IQ2_S); break;
        case T_IQ3_XXS: GU(T_IQ3_XXS); break;
        case T_IQ3_S: GU(T_IQ3_S); break;
        case T_IQ4_XS: GU(T_IQ4_XS); break;
        default: return false;
    }
#undef GU
#define DN(T) native_kernel<T, false><<<grid_dn, NW_THREADS, 0, s>>>(blobs, ntiles, tiles, bounds, geo, (const uint8_t*) ha, src, nullptr, out)
    switch (geo.d_type) {
        case T_IQ4_NL: DN(T_IQ4_NL); break;
        case T_IQ2_S: DN(T_IQ2_S); break;
        case T_IQ3_XXS: DN(T_IQ3_XXS); break;
        default: return false;
    }
#undef DN
    return hipGetLastError() == hipSuccess;
}

}  // namespace abyss::igpu_fused
