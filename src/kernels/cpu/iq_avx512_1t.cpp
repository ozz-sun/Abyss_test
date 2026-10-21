// src/kernels/cpu/iq_avx512_1t.cpp - E-3: one-token IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS rows (a decode step's CPU-lane
// experts) in 512-bit lanes, the grids looked up in registers.
//
// ggml-cpu's AVX2 dots rebuild every weight from memory lookups: 16 grid words assembled into vectors for 128 IQ2_XS
// values, most of a row's time on a Zen 5, which runs these rows at ~5 GB/s a core (16 cores ~75 of the ~113 GB/s it
// reads).  The IQ2 grids' bytes are only 8 / 25 / 43 and iq3xxs_grid's 4, 12, ..., 52, 62, so an entry is a 16-bit
// code (2 bits a value; IQ3_XXS 3 bits, 12 used): the 256 IQ2_XXS / IQ3_XXS codes fill 8 registers, the 512 IQ2_XS
// codes 16, the 1024 IQ2_S codes 32 (read from L1), and vpermt2w looks 32 entries up at once.  vpmultishiftqb + vpshufb turn the codes into the grid bytes, vpermi2b turns
// the 7-bit sign indices into ksigns_iq2xs's bytes (a 64-bit mask a 64-value chunk), and maddubs + vpdpwssd take
// ggml's integer steps.  The result is BIT-IDENTICAL to ggml's dot: the int32 lanes hold the same products (16 lanes
// folded to ggml's 8, as in iq_avx512.cpp), the floats take ggml's steps (one fmadd a block with d_x * d_y, its
// hsum_float_8, the format's constant); native_expert.cpp checks it against ggml-cpu at every start.
//
// AVX-512 F / BW / VL / VNNI / VBMI (cpu_avx512_ok(): Ice Lake, Zen 4 and later).  This file is built with those
// flags: nothing in it runs before that check (no initializers at namespace scope; the tables are built on the
// first call).
#include "strata/kernels/cpu/iq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

// ggml's hsum_float_8: the same additions in the same order
inline float hsum_ggml(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// a block's 16 int32 lanes (values 4m..4m+3 of every 64-value chunk) -> ggml's 8 (values 4k..4k+3 of every 32-value
// chunk): lane k + lane k + 8, exact
inline __m256i fold(__m512i a) { return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1)); }

alignas(64) uint16_t code2xxs[256];  // iq2xxs_grid as 2-bit codes, value v at bits 2v: 0 = 8, 1 = 25, 2 = 43
alignas(64) uint16_t code2xs[512];   // iq2xs_grid, the same
alignas(64) uint16_t code2s[1024];   // iq2s_grid, the same
alignas(64) uint16_t code3xxs[256];  // iq3xxs_grid as 3-bit codes, value v at bits 3v: 4, 12, ..., 52, 62
alignas(64) uint8_t ksigns[128];     // ksigns_iq2xs: a 7-bit sign index with its parity as bit 7

// an IQ2 grid's entries as codes (false: a byte that is not 8 / 25 / 43 - never in ggml's grids)
bool codes2(const uint64_t* grid, int n, uint16_t* code) {
    for (int e = 0; e < n; ++e) {
        uint16_t c = 0;
        for (int v = 0; v < 8; ++v) {
            const uint8_t b = (uint8_t) (grid[e] >> (8 * v));
            const int k = b == 8 ? 0 : b == 25 ? 1 : b == 43 ? 2 : -1;
            if (k < 0) return false;
            c = (uint16_t) (c | (k << (2 * v)));
        }
        code[e] = c;
    }
    return true;
}

bool build_codes() {
    if (!codes2(iq2xxs_grid, 256, code2xxs) || !codes2(iq2xs_grid, 512, code2xs) || !codes2(iq2s_grid, 1024, code2s))
        return false;
    static const uint8_t lv3[8] = {4, 12, 20, 28, 36, 44, 52, 62};
    for (int e = 0; e < 256; ++e) {
        uint16_t c = 0;
        for (int v = 0; v < 4; ++v) {
            const uint8_t b = (uint8_t) (iq3xxs_grid[e] >> (8 * v));
            int k = -1;
            for (int l = 0; l < 8; ++l)
                if (lv3[l] == b) k = l;
            if (k < 0) return false;
            c = (uint16_t) (c | (k << (3 * v)));
        }
        code3xxs[e] = c;
    }
    std::memcpy(ksigns, ksigns_iq2xs, sizeof ksigns);
    return true;
}

// the codes of 32 entries (idx: 9-bit word indices) from code2xs's 16 registers: pair k holds entries 64k..64k+63
// (vpermt2w reads index bits 0-5), bits 6-8 pick the pair
inline __m512i look512(__m512i idx, const __m512i* T) {
    __m512i r0 = _mm512_permutex2var_epi16(T[0], idx, T[1]);
    __m512i r1 = _mm512_permutex2var_epi16(T[2], idx, T[3]);
    __m512i r2 = _mm512_permutex2var_epi16(T[4], idx, T[5]);
    __m512i r3 = _mm512_permutex2var_epi16(T[6], idx, T[7]);
    __m512i r4 = _mm512_permutex2var_epi16(T[8], idx, T[9]);
    __m512i r5 = _mm512_permutex2var_epi16(T[10], idx, T[11]);
    __m512i r6 = _mm512_permutex2var_epi16(T[12], idx, T[13]);
    __m512i r7 = _mm512_permutex2var_epi16(T[14], idx, T[15]);
    const __mmask32 b6 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(64));
    const __mmask32 b7 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(128));
    const __mmask32 b8 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(256));
    r0 = _mm512_mask_mov_epi16(r0, b6, r1);
    r2 = _mm512_mask_mov_epi16(r2, b6, r3);
    r4 = _mm512_mask_mov_epi16(r4, b6, r5);
    r6 = _mm512_mask_mov_epi16(r6, b6, r7);
    r0 = _mm512_mask_mov_epi16(r0, b7, r2);
    r4 = _mm512_mask_mov_epi16(r4, b7, r6);
    return _mm512_mask_mov_epi16(r0, b8, r4);
}

// ... of 32 IQ2_S entries (10-bit indices) from code2s's 32 registers' worth (bits 6-9 pick the pair)
inline __m512i look1024(__m512i idx, const __m512i* T) {
    __m512i r[16];
    for (int k = 0; k < 16; ++k) r[k] = _mm512_permutex2var_epi16(T[2 * k], idx, T[2 * k + 1]);
    for (int bit = 6, step = 1; bit <= 9; ++bit, step *= 2) {
        const __mmask32 m = _mm512_test_epi16_mask(idx, _mm512_set1_epi16((short) (1 << bit)));
        for (int k = 0; k + step < 16; k += 2 * step) r[k] = _mm512_mask_mov_epi16(r[k], m, r[k + step]);
    }
    return r[0];
}

// ... and of 32 IQ2_XXS / IQ3_XXS entries (8-bit indices) from code2xxs's / code3xxs's 8 registers
inline __m512i look256(__m512i idx, const __m512i* T) {
    __m512i r0 = _mm512_permutex2var_epi16(T[0], idx, T[1]);
    __m512i r1 = _mm512_permutex2var_epi16(T[2], idx, T[3]);
    __m512i r2 = _mm512_permutex2var_epi16(T[4], idx, T[5]);
    __m512i r3 = _mm512_permutex2var_epi16(T[6], idx, T[7]);
    const __mmask32 b6 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(64));
    const __mmask32 b7 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(128));
    r0 = _mm512_mask_mov_epi16(r0, b6, r1);
    r2 = _mm512_mask_mov_epi16(r2, b6, r3);
    return _mm512_mask_mov_epi16(r0, b7, r2);
}

// int16 lane l of a chunk's scale vector <- scale (4 / 2) j + l / 8 (IQ2_XS: a scale a 16 values) or l / 16 (IQ3_XXS:
// a scale a 32 values)
inline __m512i lane_groups(int per) {
    alignas(64) int16_t v[32];
    for (int l = 0; l < 32; ++l) v[l] = (int16_t) (l / per);
    return _mm512_load_si512(v);
}

float row_iq2xs(const uint8_t* row, int nb, const block_q8_K* y, int ahead) {
    const __m512i* T = (const __m512i*) code2xs;
    const __m512i ks0 = _mm512_load_si512(ksigns), ks1 = _mm512_load_si512(ksigns + 64);
    const __m512i m511 = _mm512_set1_epi16(511);
    const __m512i sh2 = _mm512_set1_epi64(0x0e0c0a0806040200ll);   // byte v of a qword <- its bits 2v..2v+7
    const __m512i three = _mm512_set1_epi8(3);
    const __m512i lut = _mm512_set4_epi32(0, 0, 0, 0x002b1908);     // codes 0 / 1 / 2 -> 8 / 25 / 43
    const __m512i zero = _mm512_setzero_si512();
    const __m512i g8 = lane_groups(8);
    const __m512i sp0 = g8, sp1 = _mm512_add_epi16(g8, _mm512_set1_epi16(4)),
                  sp2 = _mm512_add_epi16(g8, _mm512_set1_epi16(8)), sp3 = _mm512_add_epi16(g8, _mm512_set1_epi16(12));
    const __m128i m4 = _mm_set1_epi8(15), one8 = _mm_set1_epi8(1);
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = row + (size_t) i * 74;
        if (ahead > 0) {
            _mm_prefetch((const char*) b + ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) b + ahead + 64, _MM_HINT_T0);
        }
        const __m512i q = _mm512_loadu_si512(b + 2);   // 32 x (9-bit grid index | 7-bit sign index << 9)
        const __m512i codes = look512(_mm512_and_si512(q, m511), T);
        const __m512i s7 = _mm512_zextsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(q, 9)));
        alignas(32) uint64_t sm[4];   // the 32 sign bytes: chunk j's 64-bit mask
        _mm256_store_si256((__m256i*) sm, _mm512_castsi512_si256(_mm512_permutex2var_epi8(ks0, s7, ks1)));
        // ggml's scales: the 16 nibbles in order, 2s + 1, as int16
        const __m128i s8 = _mm_loadl_epi64((const __m128i*) (b + 66));
        const __m128i nib = _mm_unpacklo_epi8(_mm_and_si128(s8, m4), _mm_and_si128(_mm_srli_epi16(s8, 4), m4));
        const __m512i sc = _mm512_zextsi256_si512(_mm256_cvtepu8_epi16(_mm_add_epi8(_mm_add_epi8(nib, nib), one8)));
        const int8_t* yq = y[i].qs;
        __m512i acc = _mm512_setzero_si512();
#define CHUNK(j, sp)                                                                                                  \
        {                                                                                                             \
            const __m512i c = _mm512_cvtepu16_epi64(_mm512_extracti32x4_epi32(codes, j));                            \
            const __m512i g = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(sh2, c), three)); \
            const __m512i yv = _mm512_loadu_si512(yq + 64 * j);                                                      \
            const __m512i ys = _mm512_mask_sub_epi8(yv, _cvtu64_mask64(sm[j]), zero, yv);                            \
            acc = _mm512_dpwssd_epi32(acc, _mm512_maddubs_epi16(g, ys), _mm512_permutexvar_epi16(sp, sc));            \
        }
        CHUNK(0, sp0) CHUNK(1, sp1) CHUNK(2, sp2) CHUNK(3, sp3)
#undef CHUNK
        const float d = h2f(u16(b)) * y[i].d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(fold(acc)), accf);
    }
    return 0.125f * hsum_ggml(accf);
}

float row_iq2xxs(const uint8_t* row, int nb, const block_q8_K* y, int ahead) {
    const __m512i* T = (const __m512i*) code2xxs;
    const __m512i ks0 = _mm512_load_si512(ksigns), ks1 = _mm512_load_si512(ksigns + 64);
    const __m512i sh2 = _mm512_set1_epi64(0x0e0c0a0806040200ll);
    const __m512i three = _mm512_set1_epi8(3);
    const __m512i lut = _mm512_set4_epi32(0, 0, 0, 0x002b1908);
    const __m256i shs = _mm256_set1_epi64x(0x352e2720150e0700ll);
    const __m256i m7f = _mm256_set1_epi8(0x7f);
    const __m512i zero = _mm512_setzero_si512();
    // the 64 qs bytes hold 8 groups of 32 values: 4 grid indices, then a word of 4 sign indices and the scale << 28
    alignas(64) uint8_t gi[64];
    for (int k = 0; k < 64; ++k) gi[k] = (uint8_t) ((k / 4) * 8 + k % 4);
    const __m512i pick = _mm512_load_si512(gi);
    const __m512i g16 = lane_groups(16);
    const __m512i sp0 = g16, sp1 = _mm512_add_epi16(g16, _mm512_set1_epi16(2)),
                  sp2 = _mm512_add_epi16(g16, _mm512_set1_epi16(4)), sp3 = _mm512_add_epi16(g16, _mm512_set1_epi16(6));
    const __m256i one32 = _mm256_set1_epi32(1);
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = row + (size_t) i * 66;
        if (ahead > 0) {
            _mm_prefetch((const char*) b + ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) b + ahead + 64, _MM_HINT_T0);
        }
        const __m512i qs = _mm512_loadu_si512(b + 2);
        const __m512i idx = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutexvar_epi8(pick, qs)));
        const __m512i codes = look256(idx, T);
        const __m256i aux = _mm512_cvtepi64_epi32(_mm512_srli_epi64(qs, 32));   // the 8 sign / scale words
        const __m512i s7 = _mm512_zextsi256_si512(_mm256_and_si256(_mm256_multishift_epi64_epi8(shs, aux), m7f));
        alignas(32) uint64_t sm[4];
        _mm256_store_si256((__m256i*) sm, _mm512_castsi512_si256(_mm512_permutex2var_epi8(ks0, s7, ks1)));
        const __m256i ls = _mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aux, 28), 1), one32);
        const __m512i sc = _mm512_zextsi128_si512(_mm256_cvtepi32_epi16(ls));
        const int8_t* yq = y[i].qs;
        __m512i acc = _mm512_setzero_si512();
#define CHUNK(j, sp)                                                                                                  \
        {                                                                                                             \
            const __m512i c = _mm512_cvtepu16_epi64(_mm512_extracti32x4_epi32(codes, j));                            \
            const __m512i g = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(sh2, c), three)); \
            const __m512i yv = _mm512_loadu_si512(yq + 64 * j);                                                      \
            const __m512i ys = _mm512_mask_sub_epi8(yv, _cvtu64_mask64(sm[j]), zero, yv);                            \
            acc = _mm512_dpwssd_epi32(acc, _mm512_maddubs_epi16(g, ys), _mm512_permutexvar_epi16(sp, sc));            \
        }
        CHUNK(0, sp0) CHUNK(1, sp1) CHUNK(2, sp2) CHUNK(3, sp3)
#undef CHUNK
        const float d = h2f(u16(b)) * y[i].d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(fold(acc)), accf);
    }
    return 0.125f * hsum_ggml(accf);
}

float row_iq2s(const uint8_t* row, int nb, const block_q8_K* y, int ahead) {
    const __m512i* T = (const __m512i*) code2s;
    const __m512i sh2 = _mm512_set1_epi64(0x0e0c0a0806040200ll);
    const __m512i three = _mm512_set1_epi8(3);
    const __m512i lut = _mm512_set4_epi32(0, 0, 0, 0x002b1908);
    const __m512i zero = _mm512_setzero_si512();
    // entry e's grid index: qs[e] | ((qh[e / 4] >> 2 (e % 4)) & 3) << 8 - vpmultishiftqb brings bits 2e of qh (one
    // qword, in every lane) to the low byte of word e
    alignas(64) uint8_t hc[64];
    for (int k = 0; k < 64; ++k) hc[k] = (uint8_t) (k % 2 == 0 ? (k / 2) * 2 : 0);
    const __m512i hsh = _mm512_load_si512(hc);
    const __m512i h3 = _mm512_set1_epi16(3);
    const __m512i g8 = lane_groups(8);
    const __m512i sp0 = g8, sp1 = _mm512_add_epi16(g8, _mm512_set1_epi16(4)),
                  sp2 = _mm512_add_epi16(g8, _mm512_set1_epi16(8)), sp3 = _mm512_add_epi16(g8, _mm512_set1_epi16(12));
    const __m128i m4 = _mm_set1_epi8(15), one8 = _mm_set1_epi8(1);
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = row + (size_t) i * 82;
        if (ahead > 0) {
            _mm_prefetch((const char*) b + ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) b + ahead + 64, _MM_HINT_T0);
        }
        uint64_t qh;
        std::memcpy(&qh, b + 66, 8);
        const __m512i hi = _mm512_and_si512(_mm512_multishift_epi64_epi8(hsh, _mm512_set1_epi64((long long) qh)), h3);
        const __m512i idx = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (b + 2))),
                                            _mm512_slli_epi16(hi, 8));
        const __m512i codes = look1024(idx, T);
        const __m128i s8 = _mm_loadl_epi64((const __m128i*) (b + 74));
        const __m128i nib = _mm_unpacklo_epi8(_mm_and_si128(s8, m4), _mm_and_si128(_mm_srli_epi16(s8, 4), m4));
        const __m512i sc = _mm512_zextsi256_si512(_mm256_cvtepu8_epi16(_mm_add_epi8(_mm_add_epi8(nib, nib), one8)));
        const uint8_t* sg = b + 34;   // the sign bytes: chunk j's 64-bit mask at 8j
        const int8_t* yq = y[i].qs;
        __m512i acc = _mm512_setzero_si512();
#define CHUNK(j, sp)                                                                                                  \
        {                                                                                                             \
            uint64_t m;                                                                                               \
            std::memcpy(&m, sg + 8 * j, 8);                                                                           \
            const __m512i c = _mm512_cvtepu16_epi64(_mm512_extracti32x4_epi32(codes, j));                            \
            const __m512i g = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(sh2, c), three)); \
            const __m512i yv = _mm512_loadu_si512(yq + 64 * j);                                                      \
            const __m512i ys = _mm512_mask_sub_epi8(yv, _cvtu64_mask64(m), zero, yv);                                \
            acc = _mm512_dpwssd_epi32(acc, _mm512_maddubs_epi16(g, ys), _mm512_permutexvar_epi16(sp, sc));            \
        }
        CHUNK(0, sp0) CHUNK(1, sp1) CHUNK(2, sp2) CHUNK(3, sp3)
#undef CHUNK
        const float d = h2f(u16(b)) * y[i].d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(fold(acc)), accf);
    }
    return 0.125f * hsum_ggml(accf);
}

float row_iq3xxs(const uint8_t* row, int nb, const block_q8_K* y, int ahead) {
    const __m512i* T = (const __m512i*) code3xxs;
    const __m512i ks0 = _mm512_load_si512(ksigns), ks1 = _mm512_load_si512(ksigns + 64);
    // a qword's two codes -> bytes from bits 0, 3, 6, 9 and 32, 35, 38, 41
    const __m512i sh3 = _mm512_set1_epi64(0x2926232009060300ll);
    const __m512i seven = _mm512_set1_epi8(7);
    const __m512i lut = _mm512_set4_epi32(0, 0, 0x3e342c24, 0x1c140c04);   // codes 0..7 -> 4, 12, ..., 52, 62
    // a qword's two sign / scale words -> the 7-bit sign indices at bits 0, 7, 14, 21 and 32, 39, 46, 53
    const __m256i shs = _mm256_set1_epi64x(0x352e2720150e0700ll);
    const __m256i m7f = _mm256_set1_epi8(0x7f);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i g16 = lane_groups(16);
    const __m512i sp0 = g16, sp1 = _mm512_add_epi16(g16, _mm512_set1_epi16(2)),
                  sp2 = _mm512_add_epi16(g16, _mm512_set1_epi16(4)), sp3 = _mm512_add_epi16(g16, _mm512_set1_epi16(6));
    const __m256i one32 = _mm256_set1_epi32(1);
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = row + (size_t) i * 98;
        if (ahead > 0) {
            _mm_prefetch((const char*) b + ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) b + ahead + 64, _MM_HINT_T0);
        }
        const __m512i clo = look256(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (b + 2))), T);
        const __m512i chi = look256(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (b + 34))), T);
        const __m256i aux = _mm256_loadu_si256((const __m256i*) (b + 66));   // 8 x (4 sign indices | scale << 28)
        const __m512i s7 = _mm512_zextsi256_si512(_mm256_and_si256(_mm256_multishift_epi64_epi8(shs, aux), m7f));
        alignas(32) uint64_t sm[4];
        _mm256_store_si256((__m256i*) sm, _mm512_castsi512_si256(_mm512_permutex2var_epi8(ks0, s7, ks1)));
        const __m256i ls = _mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aux, 28), 1), one32);   // 2s + 1
        const __m512i sc = _mm512_zextsi128_si512(_mm256_cvtepi32_epi16(ls));
        const int8_t* yq = y[i].qs;
        __m512i acc = _mm512_setzero_si512();
#define CHUNK(j, cw, half, sp)                                                                                        \
        {                                                                                                             \
            const __m512i c = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(cw, half));                             \
            const __m512i g = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(sh3, c), seven)); \
            const __m512i yv = _mm512_loadu_si512(yq + 64 * j);                                                      \
            const __m512i ys = _mm512_mask_sub_epi8(yv, _cvtu64_mask64(sm[j]), zero, yv);                            \
            acc = _mm512_dpwssd_epi32(acc, _mm512_maddubs_epi16(g, ys), _mm512_permutexvar_epi16(sp, sc));            \
        }
        CHUNK(0, clo, 0, sp0) CHUNK(1, clo, 1, sp1) CHUNK(2, chi, 0, sp2) CHUNK(3, chi, 1, sp3)
#undef CHUNK
        const float d = h2f(u16(b)) * y[i].d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(fold(acc)), accf);
    }
    return 0.25f * hsum_ggml(accf);
}

}  // namespace

bool iq512_rows1_supported(int type) noexcept { return type == 16 || type == 17 || type == 18 || type == 22; }

bool iq512_rows1_ready() noexcept {
    static const bool ok = build_codes();
    return ok;
}

void iq512_rows1(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    // E-2's software prefetch distance (STRATA_IQ_PREFETCH, bytes; 0 = off)
    static const int ahead = [] {
        const char* v = std::getenv("STRATA_IQ_PREFETCH");
        return v ? std::atoi(v) : 2048;
    }();
    const int nb = n / QK_K;
    for (int t = 0; t < nt; ++t) {
        const block_q8_K* y = (const block_q8_K*) act[t];
        for (int r = r0; r < r1; ++r) {
            const uint8_t* row = w + (size_t) r * row_bytes;
            out[t][r] = type == 16   ? row_iq2xxs(row, nb, y, ahead)
                        : type == 17 ? row_iq2xs(row, nb, y, ahead)
                        : type == 22 ? row_iq2s(row, nb, y, ahead)
                                     : row_iq3xxs(row, nb, y, ahead);
        }
    }
}

}  // namespace strata::kernels::cpu
