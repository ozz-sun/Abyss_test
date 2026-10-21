// src/core/dsv4_verify.cu - deepseek4's verify window: T (1..kVerifyMax) tokens through the fast path at once, for
// speculative decoding (the engine's n-gram drafts, src/program/generate.cpp).
//
// The layer loop of fast_layers with T columns.  What is shared by the window:
//   * every dense GEMV (attention projections, q_b, wo_a / wo_b, the router and the shared expert's gate/up, the
//     output head) is one gf::mvn launch over the T columns - the weights are read once (DeepSeek V4 Flash's 6.7 GB
//     of them: 4.99 ms for one token, 5.10 for two on an RTX 5090);
//   * the CPU lane's requests: every token's route of a layer is issued before the first answer is awaited, so the
//     host works through the window's RAM-tier experts while the device runs the resident ones.
// What stays per token: the mHC reads, the attention's prep / compress / q_prep / indexer / attend (in position order:
// token t sees exactly positions <= p0 + t - a later token's ring rows and compressed rows are not visible to it), a
// route each, the resident experts' kernels and the CPU-lane combine.
// STRATA_DS4_SPEC_PIPE (default 1): the PIPELINED window instead - each token on its own stream through the
// single-token kernels, token t + 1 a stage behind token t: its attention of a layer runs while the CPU lane computes
// token t's RAM-tier experts of that layer, and token t's next layer runs while the lane does token t + 1's.  The
// dense weights are read once a token (no GEMV sharing), but the host's expert work - the window's largest cost -
// overlaps the device's instead of adding to it.  The order that keeps it exact: a token's attention of a layer
// waits for the previous token's (ring rows, compressor state), and the routes run in (layer, token) order (the
// request counter, the tier tables, the host's in-order answers).
// Rejected tokens are dropped by spec_rewind (the position only): the window ring has kSpecSlack spare rows, and the
// compressor rings / compressed caches are rewritten in position order before anything reads them.
// A layer split: the parts one after another (verify_part), as a token's - each part's last layer writes its columns'
// complete residuals, which cross to the next part through a pinned host buffer, the copies ordered by an event; the
// last part runs the head and keeps the logits and the argmax (verify_argmax / _sample / _logits read its buffers).
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace ds = strata::kernels::dsv4;

namespace {
constexpr size_t kQ8 = 36;   // bytes of a q8_1 block (32 values)

ds::Rope layer_rope_v(const Glm5Geometry& g, bool comp) {
    if (!comp) return ds::make_rope(g.n_rot, g.rope_base, 1.0f, 0.0f, 0, 0.0f, 0.0f);
    return ds::make_rope(g.n_rot, g.comp_rope_base, 1.0f / g.yarn_factor, 1.0f, g.yarn_orig_ctx, g.yarn_beta_fast,
                         g.yarn_beta_slow);
}

// the window's buffers: [kVerifyMax] columns of each per-token activation, contiguous (column t at + t * width)
struct VerifyBuf {
    int TM = 0;
    float *R = nullptr;                       // [TM][2][hc * n_embd]: each token's residual pair
    float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    float* post_defer = nullptr;   // expert deferral: a column's FFN write weights of the layer whose part is deferred
    void* xq = nullptr;
    float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *iw = nullptr, *q = nullptr, *iq = nullptr;
    void* qr_q = nullptr;
    float *ckv[2] = {nullptr, nullptr}, *csc[2] = {nullptr, nullptr};
    float *o = nullptr, *oa = nullptr;
    void *oq = nullptr, *oaq = nullptr;
    float *rlog = nullptr, *sh_g = nullptr, *sh_u = nullptr, *sh_out = nullptr;
    void* sh_hq = nullptr;
    float *head_x = nullptr, *logits = nullptr, *emb = nullptr;
    void* head_xq = nullptr;
    int* tok = nullptr;                       // device argmax per column
    int* tok_h = nullptr;                     // pinned, host-mapped (tok_hd: its device address)
    int* tok_hd = nullptr;
    int* tok_in_h = nullptr;                  // the window's tokens, host-mapped for the embedding kernel
    int* tok_in_d = nullptr;
    float* emb_h = nullptr;                   // pinned [TM][n_embd]
    float* hop_h = nullptr;                   // a split: pinned [TM][hc * n_embd], the columns' residuals to the next part
    cudaEvent_t ev_hop = nullptr;
    std::vector<gf::MoeDev> md;               // a route plan per column (the shared tables, own plan arrays)
    gf::MoeColArgs* cols_d = nullptr;         // ... their plan / CPU-lane fields on the device (the column-batched experts)
    float* ffn_chk = nullptr;                 // STRATA_DS4_WIN_COLS_CHECK: the batched launch's FFN rows, then compared
    uint64_t chk_calls = 0, chk_cols = 0, chk_diff = 0;
    // per column, for the pipelined window's concurrent tokens: the mHC partials, the routed experts' h, the
    // indexer's scores / selection, the fetched experts' scratch slots (column 0: the token path's)
    float* hpart = nullptr;
    unsigned int* hcnt = nullptr;
    void* hq = nullptr;
    float* score = nullptr;
    int* sel = nullptr;
    void* scr = nullptr;
    size_t hpart_n = 0, hq_n = 0, score_n = 0;
    bool pipe = false;                        // the per-column scratch slots allocated: the pipelined window possible
    bool pipe_now = true;                     // this window pipelined (verify's choice, STRATA_DS4_SPEC_PIPE)
    // (the first part's) the kind kept on this machine, a width each (-1: still timing both): the kinds are close at
    // 3 columns and far apart at 6 (61 against 75 ms) - a choice made at one width does not hold at another
    int pipe_choice[Glm5Model::kVerifyMax + 1] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    // ... the timed windows' ms of each kind, by width (windows of different widths are not comparable: DSpark's
    // vary 2..6 columns, and 3 + 3 of mixed widths once kept the batched window at 78 ms against 56)
    std::vector<double> ms_pipe[Glm5Model::kVerifyMax + 1], ms_batch[Glm5Model::kVerifyMax + 1];
    cudaStream_t st[Glm5Model::kVerifyMax] = {};   // column t's stream (st[0]: the compute stream)
    cudaEvent_t ev_attn[Glm5Model::kVerifyMax] = {}, ev_route[Glm5Model::kVerifyMax] = {},
                ev_fin[Glm5Model::kVerifyMax] = {}, ev_start = nullptr;
    cudaEvent_t done = nullptr;
    int T = 0;                                // the last window's size
    // column t's CPU-lane answer of the last layer, left to its next attention-side hc3 read (verify_moe, as fast_moe's
    // STRATA_GLM_HC_CPU) - and the paced promotion copies, which go out after that read
    bool hc_cpu[Glm5Model::kVerifyMax] = {}, bg_after[Glm5Model::kVerifyMax] = {};
    // the pipelined window's scratch slot sets (8 VRAM slots a set): set 0 the token path's, sets 1..n_scr the
    // window's own - a group of columns runs on one stream, so its columns share their group's set
    static constexpr int kScrSets = 3;
    unsigned long long* scr_set[kScrSets + 1] = {};
    int n_scr = 0;
};
}  // namespace

bool Glm5Model::verify_ready() const {
    // every part of a split (the window runs part after part): the fast path, no draft block, prefetch or lookahead,
    // MoE layers; the head's weights on the last
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const FastState* F = m->fast_;
        if (F == nullptr || !m->g_.dsv4() || m->g_.v41 || m->mtp_il_ >= 0 || F->max_pf > 0 || F->n_ahead > 0) return false;
        for (int il = m->l0_; il < m->l1_; ++il)
            if (!F->L[(size_t) il].moe) return false;
    }
    const Glm5Model* tail = verify_tail();
    const auto it = tail->ws_map_.find("output.weight");
    return it != tail->ws_map_.end() && it->second.type != 0;
}

const Glm5Model* Glm5Model::verify_tail() const {
    const Glm5Model* t = this;
    while (t->split_next_) t = t->split_next_.get();
    return t;
}

bool Glm5Model::verify_alloc(std::string& err) {
    if (vb_ != nullptr) return true;
    const Glm5Geometry& g = g_;
    auto* V = new VerifyBuf();
    const int TM = kVerifyMax;
    V->TM = TM;
    const size_t E = (size_t) g.n_embd, D = (size_t) g.head_dim, HD = (size_t) g.n_head * D,
                 OL = (size_t) g.o_groups * g.o_lora, FF = (size_t) g.n_ff_exp * g.n_shared;
    bool ok = true;
    const auto f = [&](float*& p, size_t n) { ok = ok && cudaMalloc((void**) &p, n * TM * sizeof(float)) == cudaSuccess; };
    const auto b = [&](void*& p, size_t n) { ok = ok && cudaMalloc(&p, n * TM) == cudaSuccess; };
    ok = cudaMalloc((void**) &V->R, (size_t) TM * 2 * g.hc * E * sizeof(float)) == cudaSuccess;
    f(V->x, E); f(V->mixer, E); f(V->ffn, E); f(V->pre, 8); f(V->post, 8); f(V->comb, 16); f(V->post_defer, 8);
    b(V->xq, E / 32 * kQ8);
    f(V->qr_raw, g.q_lora); f(V->qr, g.q_lora); b(V->qr_q, (size_t) g.q_lora / 32 * kQ8);
    f(V->kv_raw, D); f(V->iw, g.idx_heads); f(V->q, HD); f(V->iq, (size_t) g.idx_heads * g.idx_key);
    for (int i = 0; i < 2; ++i) { f(V->ckv[i], 2 * D); f(V->csc[i], 2 * D); }
    f(V->o, HD); b(V->oq, HD / 32 * kQ8); f(V->oa, OL); b(V->oaq, OL / 32 * kQ8);
    f(V->rlog, g.n_expert); f(V->sh_g, FF); f(V->sh_u, FF); f(V->sh_out, E); b(V->sh_hq, FF / 32 * kQ8);
    f(V->head_x, E); b(V->head_xq, E / 32 * kQ8); f(V->logits, (size_t) g.n_vocab); f(V->emb, E);
    ok = ok && cudaMalloc((void**) &V->tok, 64 * sizeof(int)) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &V->tok_h, 64 * sizeof(int), cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &V->tok_in_h, 64 * sizeof(int), cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaHostGetDevicePointer((void**) &V->tok_hd, V->tok_h, 0) == cudaSuccess;
    ok = ok && cudaHostGetDevicePointer((void**) &V->tok_in_d, V->tok_in_h, 0) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &V->emb_h, (size_t) TM * E * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
    ok = ok && cudaEventCreateWithFlags(&V->done, cudaEventDisableTiming) == cudaSuccess;
    if (split_next_)
        ok = ok && cudaHostAlloc((void**) &V->hop_h, (size_t) TM * g.hc * E * sizeof(float), cudaHostAllocDefault) ==
                       cudaSuccess &&
             cudaEventCreateWithFlags(&V->ev_hop, cudaEventDisableTiming) == cudaSuccess;
    V->hpart_n = (size_t) (4 * E / 512) * 36 + 64;
    V->hq_n = (size_t) 8 * g.n_ff_exp / 32 * kQ8;
    V->score_n = (size_t) std::max<int64_t>(64, max_ctx_ / 4 + 64);
    ok = ok && cudaMalloc((void**) &V->hpart, V->hpart_n * TM * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &V->hcnt, (size_t) 64 * TM) == cudaSuccess;
    ok = ok && cudaMalloc(&V->hq, V->hq_n * TM) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &V->score, V->score_n * TM * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &V->sel, (size_t) g.idx_top_k * TM * sizeof(int)) == cudaSuccess;
    if (ok) cudaMemset(V->hcnt, 0, (size_t) 64 * TM);
    V->st[0] = fast_->cs;
    for (int t = 0; t < TM && ok; ++t) {
        if (t > 0) ok = cudaStreamCreateWithFlags(&V->st[t], cudaStreamNonBlocking) == cudaSuccess;
        for (cudaEvent_t* e : {&V->ev_attn[t], &V->ev_route[t], &V->ev_fin[t]})
            ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
    }
    ok = ok && cudaEventCreateWithFlags(&V->ev_start, cudaEventDisableTiming) == cudaSuccess;
    // the per-column route plans: everything the route / wait / fetch / gate_up / down / cpu_wait kernels write per
    // route; the tables, the request counter, the rings, the scratch slots and the route counts stay shared
    V->md.assign((size_t) TM, fast_->md);
    for (int t = 0; t < TM && ok; ++t) {
        gf::MoeDev& m = V->md[(size_t) t];
        uint8_t* blk = nullptr;
        const size_t nb = 4 * 8 * sizeof(unsigned long long) + 3 * 64 + E * sizeof(float);
        ok = cudaMalloc((void**) &blk, nb) == cudaSuccess;
        if (!ok) break;
        cudaMemset(blk, 0, nb);
        m.plan_ptr = (unsigned long long*) blk;
        m.fetch_src = (unsigned long long*) (blk + 64);
        m.plan_w = (float*) (blk + 128);
        m.plan_id = (int*) (blk + 192);
        m.wait_seq = (unsigned int*) (blk + 256);
        m.cpu_flag = (int*) (blk + 320);
        m.cpu_part = (float*) (blk + 448);
        if (fast_->md.cpu_seq != nullptr) m.cpu_seq = (unsigned int*) (blk + 384);
        m.pf_src = fast_->pf_buf[0];
        m.pf_dst = fast_->pf_buf[0] + 8;
        m.pf_n = fast_->pf_n_buf[0];
    }
    if (ok) {   // the columns' plans for the experts in one launch a group (moe_gate_up_cols / moe_down_cols)
        std::vector<gf::MoeColArgs> ca((size_t) TM);
        for (int t = 0; t < TM; ++t) {
            const gf::MoeDev& m = V->md[(size_t) t];
            ca[(size_t) t] = gf::MoeColArgs{m.plan_ptr, m.plan_w, m.cpu_flag, m.cpu_part};
        }
        ok = cudaMalloc((void**) &V->cols_d, ca.size() * sizeof(gf::MoeColArgs)) == cudaSuccess &&
             cudaMemcpy(V->cols_d, ca.data(), ca.size() * sizeof(gf::MoeColArgs), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    // the pipelined window's scratch slots (8 an extra column, an expert blob each): without them, the batched window
    size_t blob_max = 0;
    for (int il = l0_; il < l1_; ++il) blob_max = std::max(blob_max, fast_->L[(size_t) il].blob);
    const size_t sstride = (blob_max + 255u) & ~(size_t) 255u;
    // (sets for up to 4 groups: the 8-column window keeps 4 columns' worth, 192 MB a 8 MB blob)
    const int NS = std::min(TM - 1, (int) VerifyBuf::kScrSets);
    V->scr_set[0] = fast_->md.scratch;
    if (ok && sstride > 0 && cudaMalloc(&V->scr, (size_t) NS * 8 * sstride) == cudaSuccess) {
        for (int k = 1; k <= NS && ok; ++k) {
            unsigned long long sp[8];
            for (int i = 0; i < 8; ++i)
                sp[i] = (unsigned long long) ((uint8_t*) V->scr + ((size_t) (k - 1) * 8 + i) * sstride);
            unsigned long long* d = nullptr;
            ok = cudaMalloc((void**) &d, sizeof sp) == cudaSuccess &&
                 cudaMemcpy(d, sp, sizeof sp, cudaMemcpyHostToDevice) == cudaSuccess;
            V->scr_set[k] = d;
        }
        for (int t = 1; t < TM; ++t) V->md[(size_t) t].scratch = V->scr_set[std::min(t, NS)];
        V->n_scr = NS;
        V->pipe = ok;
    } else {
        cudaGetLastError();
        std::fprintf(stderr, "deepseek4 verify: no VRAM for the pipelined window's scratch slots (%.0f MB) - batched\n",
                     (double) NS * 8 * sstride / 1048576.0);
    }
    if (!ok) {
        cudaGetLastError();
        err = "deepseek4 verify: the window's buffers did not allocate";
        delete V;
        return false;
    }
    vb_ = V;
    return true;
}

bool Glm5Model::verify_attn(int il, int64_t p0, int tb, int T, cudaStream_t s, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int D = g.head_dim, r = Ly.ratio, E = g.n_embd;
    const ds::Rope rope = layer_rope_v(g, r != 0), crope = layer_rope_v(g, true);
    const int HD = g.n_head * D, OL = g.o_groups * g.o_lora;
    // the columns tb.. (the window's buffers from column tb on)
    const size_t q8e = (size_t) (E / 32) * kQ8;
    void* const xq = (uint8_t*) V->xq + tb * q8e;
    float* const x = V->x + (size_t) tb * E;
    float* const qr_raw = V->qr_raw + (size_t) tb * g.q_lora;
    float* const qr = V->qr + (size_t) tb * g.q_lora;
    void* const qr_qb = (uint8_t*) V->qr_q + (size_t) tb * (g.q_lora / 32) * kQ8;
    float* const kv_raw = V->kv_raw + (size_t) tb * D;
    float* const iwb = V->iw + (size_t) tb * g.idx_heads;
    float* const ckv0 = V->ckv[0] + (size_t) tb * 2 * D;
    float* const csc0 = V->csc[0] + (size_t) tb * 2 * D;
    float* const ckv1 = V->ckv[1] + (size_t) tb * 2 * D;
    float* const csc1 = V->csc[1] + (size_t) tb * 2 * D;
    float* const qb = V->q + (size_t) tb * HD;
    float* const iqb = V->iq + (size_t) tb * g.idx_heads * g.idx_key;
    float* const ob = V->o + (size_t) tb * HD;
    void* const oqb = (uint8_t*) V->oq + (size_t) tb * (HD / 32) * kQ8;
    float* const oab = V->oa + (size_t) tb * OL;
    void* const oaqb = (uint8_t*) V->oaq + (size_t) tb * (OL / 32) * kQ8;
    float* const mixer = V->mixer + (size_t) tb * E;
    // 1. the projections of x, every column at once
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 0;
    const int cw = r == 4 ? 2 * D : D;
    j[nj++] = {Ly.q_a.q, xq, x, qr_raw, nullptr, 1.0f, Ly.q_a.type, E, g.q_lora};
    j[nj++] = {Ly.kv.q, xq, x, kv_raw, nullptr, 1.0f, Ly.kv.type, E, D};
    if (r != 0) {
        j[nj++] = {Ly.comp_kv.q, xq, x, ckv0, nullptr, 1.0f, Ly.comp_kv.type, E, cw, 0, 2 * D};
        j[nj++] = {Ly.comp_gate.q, xq, x, csc0, nullptr, 1.0f, Ly.comp_gate.type, E, cw, 0, 2 * D};
    }
    if (r == 4) {
        j[nj++] = {Ly.idx_comp_kv.q, xq, x, ckv1, nullptr, 1.0f, Ly.idx_comp_kv.type, E, 2 * g.idx_key, 0, 2 * D};
        j[nj++] = {Ly.idx_comp_gate.q, xq, x, csc1, nullptr, 1.0f, Ly.idx_comp_gate.type, E, 2 * g.idx_key, 0, 2 * D};
        j[nj++] = {Ly.idx_proj_f, nullptr, x, iwb, nullptr, 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads)),
                   gf::kTypeF32, E, g.idx_heads};
    }
    if (!gf::mvn(j, nj, T, s)) { err = "deepseek4 verify: attention projections"; return false; }
    // 2. per token, in position order: norms, RoPE, the ring writes, a block end's compression
    for (int t = 0; t < T; ++t) {
        const int p = (int) (p0 + tb + t);
        ds::PrepArgs a;
        a.qr_raw = qr_raw + (size_t) t * g.q_lora;
        a.q_a_norm = Ly.q_a_norm;
        a.qr = qr + (size_t) t * g.q_lora;
        a.qr_q = (uint8_t*) qr_qb + (size_t) t * (g.q_lora / 32) * kQ8;
        a.q_lora = g.q_lora;
        a.kv_raw = kv_raw + (size_t) t * D;
        a.kv_norm = Ly.kv_a_norm;
        a.raw = Ly.raw;
        a.ring = ds_ring_;
        a.head_dim = D;
        a.n_rot = g.n_rot;
        a.rope = rope;
        a.comp_rope = crope;
        a.p = p;
        a.eps = g.norm_eps;
        if (r != 0) {
            ds::Comp& c = a.comp[a.n_comp];
            c.ring_kv = Ly.ring_kv;
            c.ring_sc = Ly.ring_sc;
            c.cache = Ly.comp_cache;
            c.ape = Ly.comp_ape;
            c.norm = Ly.comp_norm;
            c.D = D;
            c.ratio = r;
            c.overlap = r == 4;
            a.comp_kv[a.n_comp] = ckv0 + (size_t) t * 2 * D;
            a.comp_sc[a.n_comp] = csc0 + (size_t) t * 2 * D;
            ++a.n_comp;
        }
        if (r == 4) {
            ds::Comp& c = a.comp[a.n_comp];
            c.ring_kv = Ly.iring_kv;
            c.ring_sc = Ly.iring_sc;
            c.cache = Ly.idx_cache;
            c.ape = Ly.idx_comp_ape;
            c.norm = Ly.idx_comp_norm;
            c.D = g.idx_key;
            c.ratio = 4;
            c.overlap = 1;
            a.comp_kv[a.n_comp] = ckv1 + (size_t) t * 2 * D;
            a.comp_sc[a.n_comp] = csc1 + (size_t) t * 2 * D;
            ++a.n_comp;
        }
        ds::prep(a, s);
        if (r != 0 && (p + 1) % r == 0) ds::compress(a, s);
    }
    // 3. q_b (and the indexer's q), every column at once
    gf::MvJob j2[2];
    int nj2 = 0;
    j2[nj2++] = {Ly.q_b.q, qr_qb, qr, qb, nullptr, 1.0f, Ly.q_b.type, g.q_lora, HD};
    if (r == 4)
        j2[nj2++] = {Ly.idx_q_b_q.q, qr_qb, qr, iqb, nullptr, 1.0f, Ly.idx_q_b_q.type, g.q_lora,
                     g.idx_heads * g.idx_key};
    if (!gf::mvn(j2, nj2, T, s)) { err = "deepseek4 verify: q_b"; return false; }
    // 4. per token: the per-head norm + RoPE, the indexer's top-k, the attention
    for (int t = 0; t < T; ++t) {
        const int c = tb + t, p = (int) (p0 + c);
        float* q = qb + (size_t) t * HD;
        float* iq = iqb + (size_t) t * g.idx_heads * g.idx_key;
        ds::q_prep(q, g.n_head, D, g.n_rot, rope, r == 4 ? iq : nullptr, g.idx_heads, g.idx_key, crope, p, g.norm_eps,
                   s);
        const int n_raw = std::min(p + 1, g.n_swa);
        int n_comp = 0;
        const int* sel = nullptr;
        if (r != 0) {
            const int n_vis = (p + 1) / r;
            static const bool idx_all = getenv("STRATA_DS4_IDX_ALL") != nullptr && std::atoi(getenv("STRATA_DS4_IDX_ALL")) != 0;
            if (r == 4 && n_vis > 0 && (idx_all || n_vis > g.idx_top_k)) {   // (as the token path: all rows below top_k)
                int* cs = V->sel + (size_t) c * g.idx_top_k;
                ds::indexer(iq, Ly.idx_cache, iwb + (size_t) t * g.idx_heads, g.idx_heads, g.idx_key, n_vis,
                            g.idx_top_k, V->score + (size_t) c * V->score_n, cs, s);
                n_comp = std::min(n_vis, g.idx_top_k);
                sel = cs;
            } else {
                n_comp = n_vis;
            }
        }
        ds::attend(q, Ly.raw, ds_ring_, n_raw, Ly.comp_cache, sel, n_comp, Ly.sinks, g.n_head, D, g.n_rot, rope, p,
                   ob + (size_t) t * HD, s, c, (uint8_t*) oqb + (size_t) t * (HD / 32) * kQ8);
    }
    // 5. the grouped output LoRA, every column at once (a group's rows of column t at o + t * HD + gi * gin)
    const int gin = g.n_head / g.o_groups * D;
    const size_t wrow = gf::row_bytes(Ly.out_a.type, gin);
    if (wrow == 0 || g.o_groups > gf::kMaxMvJobs) { err = "deepseek4 verify: wo_a type"; return false; }
    gf::MvJob jo[gf::kMaxMvJobs];
    for (int gi = 0; gi < g.o_groups; ++gi)
        jo[gi] = {(const uint8_t*) Ly.out_a.q + (size_t) gi * g.o_lora * wrow,
                  (const uint8_t*) oqb + (size_t) gi * (gin / 32) * kQ8, ob + (size_t) gi * gin,
                  oab + (size_t) gi * g.o_lora, nullptr, 1.0f, Ly.out_a.type, gin, g.o_lora, HD / 32, OL};
    if (!gf::mvn(jo, g.o_groups, T, s)) { err = "deepseek4 verify: wo_a"; return false; }
    gf::quantize_q8_1(oab, oaqb, T * OL, s);
    gf::MvJob jb = {Ly.out_b.q, oaqb, oab, mixer, nullptr, 1.0f, Ly.out_b.type, OL, E};
    if (!gf::mvn(&jb, 1, T, s)) { err = "deepseek4 verify: wo_b"; return false; }
    return true;
}

bool Glm5Model::verify_moe(int il, int tb, int T, const std::vector<int32_t>& toks, cudaStream_t s,
                           cudaEvent_t wait_ev, cudaEvent_t done_ev, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared, k = g.n_exp_used;
    const size_t q8e = (size_t) (E / 32) * kQ8;
    gf::MvJob j[3];
    j[0] = {Ly.router, nullptr, V->x + (size_t) tb * E, V->rlog + (size_t) tb * g.n_expert, nullptr, 1.0f,
            gf::kTypeBF16, E, g.n_expert};
    j[1] = {Ly.sh_gate.q, (uint8_t*) V->xq + tb * q8e, nullptr, V->sh_g + (size_t) tb * FFs, nullptr, 1.0f,
            Ly.sh_gate.type, E, FFs};
    j[2] = {Ly.sh_up.q, (uint8_t*) V->xq + tb * q8e, nullptr, V->sh_u + (size_t) tb * FFs, nullptr, 1.0f,
            Ly.sh_up.type, E, FFs};
    // the shared expert's gate/up after the routes, as the token path does (fast_moe, STRATA_GLM_SH_LATE)
    static const bool sh_late = [] {
        const char* v = getenv("STRATA_GLM_SH_LATE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool late = sh_late && FFs > 0;
    if (!gf::mvn(j, late ? 1 : 3, T, s)) { err = "deepseek4 verify: router / shared expert"; return false; }
    const bool lane = F->cpu_plan != 0ull && F->cpu_fmt[(size_t) il].n_ff > 0;
    static const int promote_min = [] {
        const char* v = getenv("STRATA_GLM_PROMOTE_MIN");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    static const int pf_rank = [] {
        const char* v = getenv("STRATA_GLM_PREFETCH_RANK");
        return v != nullptr ? std::max(1, std::atoi(v)) : 0;
    }();
    // every token's route first: the CPU lane has the whole window's RAM-tier experts queued before any wait
    if (wait_ev != nullptr) cudaStreamWaitEvent(s, wait_ev, 0);
    for (int t = tb; t < tb + T; ++t) {
        gf::moe_route(V->rlog + (size_t) t * g.n_expert, Ly.router_bias, g.n_expert, k, g.w_scale, g.norm_w != 0, il,
                      V->x + (size_t) t * E, E, V->md[(size_t) t], late ? nullptr : V->sh_g + (size_t) t * FFs,
                      late ? nullptr : V->sh_u + (size_t) t * FFs, g.swiglu_shexp, late ? 0 : FFs,
                      (uint8_t*) V->sh_hq + (size_t) t * (FFs / 32) * kQ8,
                      s, nullptr, nullptr, 0, nullptr, nullptr, 0, fast_skip_from(il), lane ? F->cpu_plan : 0ull, promote_min, pf_rank,
                      g.gating, Ly.tid2eid != nullptr ? Ly.tid2eid + (size_t) toks[(size_t) t] * k : nullptr,
                      fast_skip_w(il), 0, nullptr, tb + T - 1 - t);   // (the columns after this one: the CPU lane
                                                                      // answers a batched window's columns together)
        ++F->expected;
    }
    if (late) {
        if (!gf::mvn(j + 1, 2, T, s)) { err = "deepseek4 verify: shared expert"; return false; }
        gf::sh_swiglu(V->sh_g + (size_t) tb * FFs, V->sh_u + (size_t) tb * FFs, g.swiglu_shexp, T * FFs,
                      (uint8_t*) V->sh_hq + (size_t) tb * (FFs / 32) * kQ8, s);
    }
    // ... then each token's resident experts (its own plan and h) ...
    static const bool wait_fetch = [] {   // STRATA_GLM_WAIT_FETCH, as fast_moe's
        const char* v = getenv("STRATA_GLM_WAIT_FETCH");
#if defined(STRATA_USE_HIP)
        return v == nullptr || std::atoi(v) != 0;
#else
        return v != nullptr && std::atoi(v) != 0;
#endif
    }();
    // STRATA_DS4_WIN_COLS (default 1): the columns' resident experts in one gate/up and one down launch (the same
    // blocks as a launch a column: the same sums) - a column's ~4 resident experts a layer left the GPU half idle
    static const bool win_cols = [] {
        const char* v = getenv("STRATA_DS4_WIN_COLS");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool cols = win_cols && T > 1 && V->cols_d != nullptr && !F->prof_on;
    // STRATA_DS4_WIN_COLS_CHECK=1: every 16th group of columns also through a launch a column, compared bit by bit
    static const bool cols_chk = getenv("STRATA_DS4_WIN_COLS_CHECK") != nullptr && std::atoi(getenv("STRATA_DS4_WIN_COLS_CHECK")) != 0;
    const bool chk = cols && cols_chk && (V->chk_calls++ % 16) == 0;
    if (chk && V->ffn_chk == nullptr && cudaMalloc((void**) &V->ffn_chk, (size_t) kVerifyMax * E * sizeof(float)) != cudaSuccess) {
        err = "deepseek4 verify: the check buffer";
        return false;
    }
    for (int t = tb; t < tb + T; ++t) {
        const gf::MoeDev& md = V->md[(size_t) t];
        float* sh_out = V->sh_out + (size_t) t * E;
        void* hq = (uint8_t*) V->hq + (size_t) t * V->hq_n;
        if (wait_fetch && !F->prof_on) {   // (one launch: gf::moe_wait_fetch, as fast_moe)
            gf::moe_wait_fetch(md, E, k, Ly.blob, s);
        } else {
            gf::moe_wait(md, E, s);
            gf::moe_fetch(md, k, Ly.blob, s);
        }
        // the next route may find this one's promotions resident: it goes after their copies have landed
        if (done_ev != nullptr && t == tb + T - 1) cudaEventRecord(done_ev, s);
        if (cols) continue;   // (the batched launches below; the check runs these after them)
        gf::moe_gate_up(Ly.gu_type, md, k, E, g.n_ff_exp, g.swiglu_exp, (const uint8_t*) V->xq + (size_t) t * q8e,
                        hq, Ly.sh_down.q, Ly.sh_down.type, (const uint8_t*) V->sh_hq + (size_t) t * (FFs / 32) * kQ8,
                        FFs, sh_out, s);
        gf::moe_down(Ly.d_type, md, k, E, g.n_ff_exp, Ly.down_off, hq, Ly.sh_down.q != nullptr ? sh_out : nullptr,
                     V->ffn + (size_t) t * E, s);
    }
    if (cols) {
        const size_t shq = (size_t) (FFs / 32) * kQ8;
        float* sh_out = V->sh_out + (size_t) tb * E;
        void* hq = (uint8_t*) V->hq + (size_t) tb * V->hq_n;
        gf::moe_gate_up_cols(Ly.gu_type, V->cols_d + tb, T, k, E, g.n_ff_exp, g.swiglu_exp,
                             (const uint8_t*) V->xq + (size_t) tb * q8e, q8e, hq, V->hq_n, Ly.sh_down.q, Ly.sh_down.type,
                             (const uint8_t*) V->sh_hq + (size_t) tb * shq, shq, FFs, sh_out, E, s);
        gf::moe_down_cols(Ly.d_type, V->cols_d + tb, T, k, E, g.n_ff_exp, Ly.down_off, hq, V->hq_n,
                          Ly.sh_down.q != nullptr ? sh_out : nullptr, chk ? V->ffn_chk : V->ffn + (size_t) tb * E, E, s);
        if (chk) {   // the same inputs through a launch a column into V->ffn, then the rows compared
            for (int t = tb; t < tb + T; ++t) {
                const gf::MoeDev& md = V->md[(size_t) t];
                float* sh_o = V->sh_out + (size_t) t * E;
                void* hq1 = (uint8_t*) V->hq + (size_t) t * V->hq_n;
                gf::moe_gate_up(Ly.gu_type, md, k, E, g.n_ff_exp, g.swiglu_exp, (const uint8_t*) V->xq + (size_t) t * q8e,
                                hq1, Ly.sh_down.q, Ly.sh_down.type, (const uint8_t*) V->sh_hq + (size_t) t * shq, FFs, sh_o, s);
                gf::moe_down(Ly.d_type, md, k, E, g.n_ff_exp, Ly.down_off, hq1, Ly.sh_down.q != nullptr ? sh_o : nullptr,
                             V->ffn + (size_t) t * E, s);
            }
            std::vector<float> a((size_t) T * E), b((size_t) T * E);
            cudaMemcpyAsync(a.data(), V->ffn_chk, a.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(b.data(), V->ffn + (size_t) tb * E, b.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            for (int t = 0; t < T; ++t) {
                ++V->chk_cols;
                if (std::memcmp(a.data() + (size_t) t * E, b.data() + (size_t) t * E, (size_t) E * sizeof(float)) != 0) ++V->chk_diff;
            }
            if ((V->chk_calls & 1023) == 1)
                std::fprintf(stderr, "deepseek4 verify: column-batched experts checked - %llu columns, %llu differ\n",
                             (unsigned long long) V->chk_cols, (unsigned long long) V->chk_diff);
        }
    }
    // ... and the CPU lane's answers, in request order.  Deferred (F->defer, as fast_ffn): a column's part is
    // written into its streams after the next layer's attention (verify's hc_defer_add) with these write weights,
    // saved now; the trunk's (or this part's) last layer waits for its own
    static const bool hc_cpu_env = [] {   // STRATA_GLM_HC_CPU, as fast_moe's
        const char* v = getenv("STRATA_GLM_HC_CPU");
#if defined(STRATA_USE_HIP)
        return v == nullptr || std::atoi(v) != 0;
#else
        return v != nullptr && std::atoi(v) != 0;
#endif
    }();
    const bool hc_cpu = lane && hc_cpu_env && !F->defer && !Ly.mtp && !F->prof_on && il != l1_ - 1 && gf::hc_fuses_cpu(E);
    if (F->defer && Ly.moe && !Ly.mtp && il != l1_ - 1) {
        if (lane)
            for (int t = tb; t < tb + T; ++t) gf::copy_floats(V->post_defer + (size_t) t * 8, V->post + (size_t) t * 8, 8, s);
    } else if (hc_cpu) {
        for (int t = tb; t < tb + T; ++t) V->hc_cpu[t] = true;
    } else if (lane) {
        for (int t = tb; t < tb + T; ++t) gf::moe_cpu_wait(V->md[(size_t) t], E, V->ffn + (size_t) t * E, s);
    }
    if (hc_cpu) V->bg_after[tb] = true;
    else F->bg_drain(s, F->bg_pace);   // the paced promotions' next bytes, as fast_moe
    return true;
}

bool Glm5Model::verify(const std::vector<int32_t>& toks, std::string& err) {
    const int T = (int) toks.size();
    if (T < 1 || T > kVerifyMax) {
        err = "deepseek4 verify: 1.." + std::to_string(kVerifyMax) + " tokens";
        return false;
    }
    if (!verify_ready()) {
        err = "deepseek4 verify: not available (deepseek4 fast path, no MTP / prefetch / lookahead)";
        return false;
    }
    if (pos_ + T > max_ctx_) {
        err = "deepseek4 verify: the window exceeds max_ctx";
        return false;
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!m->verify_alloc(err)) return false;
    }
    cudaSetDevice(dev_);
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    // the window's kind: STRATA_DS4_SPEC_PIPE=1 pipelined, 0 batched (2: pipelined on one stream, debug), auto: the
    // faster on this machine - the first windows alternate, 3 of each, then the kind with the lower median time a token
    // stays.  Unset: auto on HIP, pipelined on NVIDIA (the timing is not checked on an NVIDIA card yet).  An RTX 5090 +
    // iGPU lane: pipelined (90 against 73 tok/s on swe-bench prompts).  An AI PRO R9700 + iGPU lane: batched (2-token
    // windows 30.6 against 44.5 ms, swe-bench 6 tasks 42 -> 46-50 tok/s) - HIP put every stream in one hardware queue,
    // so the pipelined window's tokens ran one after another, each reading the dense weights
    static const int pipe_env = [] {
        const char* v = getenv("STRATA_DS4_SPEC_PIPE");
        if (v != nullptr) return std::strcmp(v, "auto") == 0 ? -1 : std::atoi(v);
#if defined(STRATA_USE_HIP)
        return -1;
#else
        return 1;
#endif
    }();
    bool timed = pipe_env < 0 && V->pipe_choice[T] < 0 && T > 1 && !F->prof_on;
    const bool use_pipe = pipe_env >= 0 ? pipe_env != 0
                        : V->pipe_choice[T] >= 0 ? V->pipe_choice[T] != 0
                        : V->ms_pipe[T].size() <= V->ms_batch[T].size();   // (timing: alternate a width, pipelined first)
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        auto* VM = (VerifyBuf*) m->vb_;
        VM->pipe_now = use_pipe;
        timed = timed && VM->pipe;   // (no scratch slots: every window is batched)
    }
    dspark_settle();
    const int64_t p0 = pos_;
    pos_ += T;
    dspark_window(p0, T);   // (DSpark: the kept columns into its context at spec_rewind)
    const bool gp = F->gap_on && split_next_ == nullptr;   // (as the token path: one device's events)
    if (gp) {
        cudaEventRecord(F->ev_g0, s);
        if (F->gap_live)
            F->gap_host += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - F->t_ready).count();
    }
    // between windows (every device idle): finished promotions go live in the device tables
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    const auto t0 = std::chrono::steady_clock::now();
    // ---- the embedding rows -> each token's 4 streams (the first part's columns)
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "deepseek4 verify: no embedding dequantizer";
        return false;
    }
    const size_t E = (size_t) g.n_embd, HCE = (size_t) g.hc * E;
    if (F->emb_tab_d != nullptr) {
        for (int t = 0; t < T; ++t) V->tok_in_h[t] = toks[(size_t) t];
        gf::embed_rows_q(pack_emb_type_, F->emb_tab_d, V->tok_in_d, T, g.n_embd, V->emb, s);
    } else {
        for (int t = 0; t < T; ++t)
            tt->to_float(pack_emb_src_ + (size_t) toks[(size_t) t] * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd),
                         V->emb_h + (size_t) t * E, g.n_embd);
        cudaMemcpyAsync(V->emb, V->emb_h, (size_t) T * E * sizeof(float), cudaMemcpyHostToDevice, s);
    }
    for (int t = 0; t < T; ++t) gf::embed_streams(V->emb + (size_t) t * E, V->R + (size_t) t * 2 * HCE, g.n_embd, s);
    if (gp) {
        cudaEventRecord(F->ev_gs, s);
        F->gap_emb += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    // ---- the parts in order (one part: this one's layers and the head)
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        m->pos_ = pos_;
        ((VerifyBuf*) m->vb_)->T = T;
        if (!m->verify_part(toks, p0, err)) return false;
    }
    cudaSetDevice(dev_);
    const Glm5Model* tail = verify_tail();
    auto* VT = (VerifyBuf*) tail->vb_;
    if (gp) cudaEventRecord(F->ev_ge[F->gap_par], s);
    const auto tw = std::chrono::steady_clock::now();
    cudaError_t done;
    while ((done = cudaEventQuery(VT->done)) == cudaErrorNotReady) {
        if (F->token_spin) glmfast::cpu_relax(); else std::this_thread::yield();
    }
    if (!glmfast::cuda_ok(done, "deepseek4 verify completion", err)) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (!m->fast_->route_ok(err)) return false;
    if (gf::launch_errors() > 0) {
        err = "deepseek4 verify: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    const double wms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    F->ms += wms;
    if (timed) {   // (the window kind still being timed: 3 of each at one width decide)
        (use_pipe ? V->ms_pipe[T] : V->ms_batch[T]).push_back(wms);
        if (V->ms_pipe[T].size() >= 3 && V->ms_batch[T].size() >= 3) {
            const auto median = [](std::vector<double> v) {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };
            const double mp = median(V->ms_pipe[T]), mb = median(V->ms_batch[T]);
            V->pipe_choice[T] = mp <= mb ? 1 : 0;
            std::fprintf(stderr, "deepseek4 verify: the %s window here at %d columns (%.1f ms against %.1f %s) - kept "
                                 "(STRATA_DS4_SPEC_PIPE=0 batched, 1 pipelined)\n",
                         V->pipe_choice[T] ? "pipelined" : "batched", T, std::min(mp, mb), std::max(mp, mb),
                         V->pipe_choice[T] ? "batched" : "pipelined");
        }
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_->prof_on) {
            cudaSetDevice(m->dev_);
            m->fast_->collect();
        }
    cudaSetDevice(dev_);
    last_tok_ = VT->tok_h[T - 1];
    if (gp) fast_gap_mark(tw);
    return true;
}

bool Glm5Model::verify_part(const std::vector<int32_t>& toks, int64_t p0, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int T = V->T;
    const bool last = split_next_ == nullptr;
    const size_t E = (size_t) g.n_embd, HCE = (size_t) g.hc * E;
    std::vector<float*> Rc((size_t) T), Ro((size_t) T);
    for (int t = 0; t < T; ++t) {
        Rc[(size_t) t] = V->R + (size_t) t * 2 * HCE;
        Ro[(size_t) t] = Rc[(size_t) t] + HCE;
    }
    const auto hc_read = [&](int il, int t, const float* block_out, bool attn, cudaStream_t hs) {
        const auto& Ly = F->L[(size_t) il];
        gf::HcArgs h;
        h.block_out = block_out;
        h.R_old = Rc[(size_t) t];
        h.R_new = Ro[(size_t) t];
        h.post_in = V->post + (size_t) t * 8;
        h.comb_in = V->comb + (size_t) t * 16;
        h.pre = V->pre + (size_t) t * 8;
        h.post = V->post + (size_t) t * 8;
        h.comb = V->comb + (size_t) t * 16;
        h.w_fn = attn ? Ly.hc_attn_fn : Ly.hc_ffn_fn;
        h.w_scale = attn ? Ly.hc_attn_scale : Ly.hc_ffn_scale;
        h.w_base = attn ? Ly.hc_attn_base : Ly.hc_ffn_base;
        h.norm_w = attn ? Ly.attn_norm : Ly.ffn_norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = g.n_embd;
        h.x = V->x + (size_t) t * E;
        h.xq = (uint8_t*) V->xq + (size_t) t * (E / 32) * kQ8;
        h.part = V->hpart + (size_t) t * V->hpart_n;
        h.counter = V->hcnt + (size_t) t * 16;
        const bool cpu = attn && block_out != nullptr && V->hc_cpu[t];   // the column's last MoE answer, added here
        if (cpu) {
            h.cpu_seq = V->md[(size_t) t].cpu_seq;
            h.cpu_ans = V->md[(size_t) t].cpu_ans;
            V->hc_cpu[t] = false;
        }
        gf::hc(h, hs);
        if (block_out != nullptr) std::swap(Rc[(size_t) t], Ro[(size_t) t]);
        if (attn && V->bg_after[t]) {
            V->bg_after[t] = false;
            F->bg_drain(hs, F->bg_pace);
        }
    };
    // expert deferral (F->defer, as fast_layers): the layer before's CPU-lane part of column t into its streams after
    // this layer's attention, with the weights verify_moe saved (its own md: the last route is the layer before's)
    const auto defer_add = [&](int il, int t, cudaStream_t hs) {
        if (!F->defer || il <= g.dense_lead || il <= l0_ || !F->L[(size_t) il].moe) return;
        gf::hc_defer_add(V->md[(size_t) t], V->post_defer + (size_t) t * 8, g.n_embd, Rc[(size_t) t], hs);
    };
    // STRATA_GLM_PROF (as the token path's): the window's phases, marked v<T>_* so the two kinds stay apart
    // the pipelined window (verify's choice, V->pipe_now; not with PROF - its marks are one stream's)
    static const bool pipe_env1 = getenv("STRATA_DS4_SPEC_PIPE") != nullptr && std::atoi(getenv("STRATA_DS4_SPEC_PIPE")) == 2;
    const bool pipe = V->pipe_now && V->pipe && T > 1 && !F->prof_on;
    // STRATA_DS4_SPEC_GROUP=<n>: the pipelined window's columns a group; default two groups of ceil(T / 2) (the two
    // hardware queues), so a 2-column window keeps a column a group
    static const int grp_env = [] {
        const char* v = getenv("STRATA_DS4_SPEC_GROUP");
        return v != nullptr ? std::max(1, std::atoi(v)) : 0;
    }();
    int grp = !pipe ? T : grp_env > 0 ? std::min(grp_env, T) : (T + 1) / 2;
    while ((T + grp - 1) / grp > 1 + V->n_scr) ++grp;   // (a scratch set a group)
    const int n_grp = (T + grp - 1) / grp;
    if (pipe)
        for (int t = 0; t < T; ++t) V->md[(size_t) t].scratch = V->scr_set[t / grp];
    const bool prof = F->prof_on;
    const std::string vt = "v" + std::to_string(T) + "_";
    static std::vector<std::string> names;   // (mark keeps the pointer)
    if (prof && names.empty())
        for (int n = 1; n <= kVerifyMax; ++n)
            for (const char* ph : {"hc_attn", "attn", "hc_ffn", "moe"}) names.push_back("v" + std::to_string(n) + "_" + ph);
    const auto mark = [&](int ph) { if (prof) F->mark(names[(size_t) (T - 1) * 4 + ph].c_str()); };
    if (prof) F->mark("v_start");
    for (int t = 0; t < kVerifyMax; ++t) V->hc_cpu[t] = V->bg_after[t] = false;
    if (pipe) {
        // group g of the columns (gs a group, the last one the rest) on stream g, after the embedding; per layer, its
        // attention after group g - 1's (ring rows, compressor state) and its routes after group g - 1's (group 0:
        // after the last group's of the layer before).  Inside a group the columns go together, as the batched
        // window's (the dense weights read once a group); across groups the host's experts of one group are computed
        // while the device runs another's attention.  gs = 1: a column a stream (the window before groups)
        cudaEventRecord(V->ev_start, s);
        for (int gi = 1; gi < n_grp; ++gi) cudaStreamWaitEvent(V->st[gi], V->ev_start, 0);
        for (int il = l0_; il < l1_; ++il)
            for (int gi = 0; gi < n_grp; ++gi) {
                const int c0 = gi * grp, nc = std::min(grp, T - c0);
                cudaStream_t st = pipe_env1 ? s : V->st[gi];
                if (gi > 0) cudaStreamWaitEvent(st, V->ev_attn[gi - 1], 0);
                for (int t = c0; t < c0 + nc; ++t) hc_read(il, t, il == l0_ ? nullptr : V->ffn + (size_t) t * E, true, st);
                if (!verify_attn(il, p0, c0, nc, st, err)) return false;
                cudaEventRecord(V->ev_attn[gi], st);
                for (int t = c0; t < c0 + nc; ++t) defer_add(il, t, st);
                if (dsp_ != nullptr && il > l0_)   // DSpark's features: the layer before's complete output
                    for (int t = c0; t < c0 + nc; ++t) dspark_capture(il - 1, Rc[(size_t) t], t, st);
                for (int t = c0; t < c0 + nc; ++t) hc_read(il, t, V->mixer + (size_t) t * E, false, st);
                cudaEvent_t after = gi > 0 ? V->ev_route[gi - 1] : il > l0_ ? V->ev_route[n_grp - 1] : nullptr;
                if (!verify_moe(il, c0, nc, toks, st, after, V->ev_route[gi], err)) return false;
            }
    } else {
        for (int il = l0_; il < l1_; ++il) {
            for (int t = 0; t < T; ++t) hc_read(il, t, il == l0_ ? nullptr : V->ffn + (size_t) t * E, true, s);
            mark(0);
            if (!verify_attn(il, p0, 0, T, s, err)) return false;
            mark(1);
            for (int t = 0; t < T; ++t) defer_add(il, t, s);
            if (dsp_ != nullptr && il > l0_)
                for (int t = 0; t < T; ++t) dspark_capture(il - 1, Rc[(size_t) t], t, s);
            for (int t = 0; t < T; ++t) hc_read(il, t, V->mixer + (size_t) t * E, false, s);
            mark(2);
            if (!verify_moe(il, 0, T, toks, s, nullptr, nullptr, err)) return false;
            mark(3);
        }
    }
    // (verify_moe leaves no answer to a read past the part's last layer; the check is for safety)
    for (int t = 0; t < T; ++t) {
        cudaStream_t st = pipe && !pipe_env1 ? V->st[t / grp] : s;
        if (V->hc_cpu[t]) {
            gf::moe_cpu_wait(V->md[(size_t) t], g.n_embd, V->ffn + (size_t) t * E, st);
            V->hc_cpu[t] = false;
        }
        if (V->bg_after[t]) {
            V->bg_after[t] = false;
            F->bg_drain(st, F->bg_pace);
        }
    }
    // the last layer's write half; a part before the last: the columns' residuals to the next part (as a token's hop);
    // the last: the head, the output rows (every column at once) and each column's argmax
    if (!last) {
        Glm5Model* B = split_next_.get();
        auto* VB = (VerifyBuf*) B->vb_;
        for (int t = 0; t < T; ++t) {
            cudaStream_t st = pipe && !pipe_env1 ? V->st[t / grp] : s;
            gf::hc_post(V->ffn + (size_t) t * E, Rc[(size_t) t], V->post + (size_t) t * 8, V->comb + (size_t) t * 16,
                        g.n_embd, Ro[(size_t) t], st);
            if (pipe && t / grp > 0) {
                cudaEventRecord(V->ev_fin[t / grp], st);
                cudaStreamWaitEvent(s, V->ev_fin[t / grp], 0);
            }
        }
        for (int t = 0; t < T; ++t)
            cudaMemcpyAsync(V->hop_h + (size_t) t * HCE, Ro[(size_t) t], HCE * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaEventRecord(V->ev_hop, s);
        cudaSetDevice(B->dev_);
        cudaStream_t sb = B->fast_->cs;
        cudaStreamWaitEvent(sb, V->ev_hop, 0);
        for (int t = 0; t < T; ++t)
            cudaMemcpyAsync(VB->R + (size_t) t * 2 * HCE, V->hop_h + (size_t) t * HCE, HCE * sizeof(float),
                            cudaMemcpyHostToDevice, sb);
        return true;
    }
    const auto fn = w16_.find("output_hc_fn.weight");
    const auto sc = w_.find("output_hc_scale.weight");
    const auto bs = w_.find("output_hc_base.weight");
    const auto nw = w_.find("output_norm.weight");
    if (fn == w16_.end() || sc == w_.end() || bs == w_.end() || nw == w_.end()) {
        err = "deepseek4 verify: the head's tensors";
        return false;
    }
    for (int t = 0; t < T; ++t) {
        cudaStream_t st = pipe && !pipe_env1 ? V->st[t / grp] : s;
        gf::hc_post(V->ffn + (size_t) t * E, Rc[(size_t) t], V->post + (size_t) t * 8, V->comb + (size_t) t * 16,
                    g.n_embd, Ro[(size_t) t], st);
        if (dsp_ != nullptr) dspark_capture(l1_ - 1, Ro[(size_t) t], t, st);
        ds::head_prep(Ro[(size_t) t], fn->second, sc->second, bs->second, g.hc_eps, nw->second, g.norm_eps, g.n_embd,
                      V->head_x + (size_t) t * E, (uint8_t*) V->head_xq + (size_t) t * (E / 32) * kQ8, st);
        if (pipe && t / grp > 0) {
            cudaEventRecord(V->ev_fin[t / grp], st);
            cudaStreamWaitEvent(s, V->ev_fin[t / grp], 0);
        }
    }
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, V->head_xq, V->head_x, V->logits, nullptr, 1.0f, ow.type, g.n_embd, g.n_vocab};
    if (gf::mvn_supported(ow.type)) {
        if (!gf::mvn(&o, 1, T, s)) { err = "deepseek4 verify: output head"; return false; }
    } else {
        for (int t = 0; t < T; ++t) {
            gf::MvJob ot = o;
            ot.xq = (const uint8_t*) V->head_xq + (size_t) t * (E / 32) * kQ8;
            ot.xf = V->head_x + (size_t) t * E;
            ot.y = V->logits + (size_t) t * g.n_vocab;
            if (!gf::mv(&ot, 1, s)) { err = "deepseek4 verify: output head (type " + std::to_string(ow.type) + ")"; return false; }
        }
    }
    if (F->zero_copy) {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok_hd + t, s);
    } else {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok + t, s);
        cudaMemcpyAsync(V->tok_h, V->tok, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost, s);
    }
    cudaEventRecord(V->done, s);
    return true;
}

int Glm5Model::verify_argmax(int i) const {
    const auto* V = (const VerifyBuf*) verify_tail()->vb_;
    return V != nullptr && i >= 0 && i < V->T ? V->tok_h[i] : -1;
}

bool Glm5Model::verify_logits(int i, std::vector<float>& out, std::string& err) {
    const Glm5Model* tail = verify_tail();
    auto* V = (VerifyBuf*) tail->vb_;
    if (V == nullptr || i < 0 || i >= V->T) {
        err = "deepseek4 verify: no such column";
        return false;
    }
    cudaSetDevice(tail->dev_);
    out.resize((size_t) g_.n_vocab);
    const bool ok = cudaMemcpy(out.data(), V->logits + (size_t) i * g_.n_vocab, out.size() * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaSetDevice(dev_);
    if (!ok) {
        err = "deepseek4 verify: logits copy";
        return false;
    }
    return true;
}

int Glm5Model::verify_sample(int i, strata::kernels::SamplerParams& sp, std::string& err) {
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    auto* V = (VerifyBuf*) tail->vb_;
    if (V == nullptr || i < 0 || i >= V->T) {
        err = "deepseek4 verify: no such column";
        return -1;
    }
    if (sp.greedy) return V->tok_h[i];
    cudaSetDevice(tail->dev_);
    cudaMemcpyAsync(tail->sc_ + tail->sc_logits, V->logits + (size_t) i * g_.n_vocab,
                    (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, tail->fast_->cs);
    const int tok = tail->fast_sample(sp, err);
    cudaSetDevice(dev_);
    return tok;
}

}  // namespace strata::core
