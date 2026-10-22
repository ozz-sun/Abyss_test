// qwen35moe's verify-window attention (q35::attn_window: the keys in chunks, then merged) against the one-pass
// attn_prompt it replaces in the window, on random FP16 caches: windows of 1-8 rows at short and long positions, and
// the two kernels' times at the window's shape (4 rows, 32 heads over 2 KV heads, head_dim 256).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "strata/kernels/qwen35_fast.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

namespace q35 = strata::kernels::q35;

int main() {
    constexpr int kH = 32, kHkv = 2, kD = 256, kCtx = 24576, kRows = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<__half> kc((size_t) kCtx * kHkv * kD), vc(kc.size());
    for (auto& x : kc) x = __float2half(nd(rng));
    for (auto& x : vc) x = __float2half(nd(rng));
    std::vector<float> q((size_t) kRows * kH * kD);
    for (auto& x : q) x = nd(rng);

    __half *dk = nullptr, *dv = nullptr;
    float *dq = nullptr, *da = nullptr, *db = nullptr, *dpart = nullptr;
    const size_t part_n = q35::attn_window_floats(kH, kD, kCtx, kRows);
    CHECK(hipMalloc(&dk, kc.size() * 2));
    CHECK(hipMalloc(&dv, vc.size() * 2));
    CHECK(hipMalloc(&dq, q.size() * 4));
    CHECK(hipMalloc(&da, q.size() * 4));
    CHECK(hipMalloc(&db, q.size() * 4));
    CHECK(hipMalloc(&dpart, part_n * 4));
    CHECK(hipMemcpy(dk, kc.data(), kc.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dv, vc.data(), vc.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
    const auto* k16 = reinterpret_cast<const uint16_t*>(dk);
    const auto* v16 = reinterpret_cast<const uint16_t*>(dv);
    const float scale = 1.0f / std::sqrt((float) kD);

    int bad = 0;
    float worst_all = 0.0f;
    for (const int64_t p0 : {0LL, 1LL, 127LL, 128LL, 1000LL, 5000LL, 20000LL}) {
        for (int T = 1; T <= kRows; ++T) {
            if (p0 + T > kCtx) continue;
            q35::attn_prompt(dq, k16, v16, kH, kHkv, kD, p0, T, scale, da, nullptr);
            q35::attn_window(dq, k16, v16, kH, kHkv, kD, p0, T, scale, dpart, db, nullptr);
            CHECK(hipDeviceSynchronize());
            std::vector<float> a((size_t) T * kH * kD), b(a.size());
            CHECK(hipMemcpy(a.data(), da, a.size() * 4, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(b.data(), db, b.size() * 4, hipMemcpyDeviceToHost));
            float worst = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
                worst = std::max(worst, std::fabs(a[i] - b[i]) / std::max(1e-3f, std::fabs(a[i])));
            worst_all = std::max(worst_all, worst);
            if (!(worst < 1e-3f)) {
                std::printf("p0 %lld T %d: relative difference %.3g\n", (long long) p0, T, worst);
                ++bad;
            }
        }
    }
    std::printf("parity: %s, the largest relative difference %.3g\n",
                bad ? "FAIL" : "ok (windows of 1-8 rows, positions 0-20000)", worst_all);

    // times: 4 rows (a token and 3 drafts) at the positions a long answer reaches
    for (const int64_t p0 : {2000LL, 8000LL, 20000LL}) {
        double ms[2];
        for (int which = 0; which < 2; ++which) {
            auto run = [&] {
                if (which == 0) q35::attn_prompt(dq, k16, v16, kH, kHkv, kD, p0, 4, scale, da, nullptr);
                else q35::attn_window(dq, k16, v16, kH, kHkv, kD, p0, 4, scale, dpart, db, nullptr);
            };
            run();
            CHECK(hipDeviceSynchronize());
            const int reps = 20;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) run();
            CHECK(hipDeviceSynchronize());
            ms[which] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        }
        std::printf("p0 %6lld, 4 rows: attn_prompt %.3f ms, attn_window %.3f ms (a layer)\n", (long long) p0, ms[0], ms[1]);
    }
    hipFree(dk);
    hipFree(dv);
    hipFree(dq);
    hipFree(da);
    hipFree(db);
    hipFree(dpart);
    return bad ? 1 : 0;
}
