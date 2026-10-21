// src/core/dspark.cu - DSpark, DeepSeek V4 Flash 0731's block drafter (bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF,
// arch deepseek4-dspark), loaded with STRATA_DS4_DSPARK=<the GGUF>.  Three DSV4 stages read the target's layer 40, 41
// and 42 outputs and propose the next 5 tokens of a verify window at once.  The arithmetic (worklog/2026-10-21_dspark.md,
// taken from antirez/ds4 and llama.cpp's dflash; the numpy reference is Abyss-tools bench/dspark/dspark_probe.py):
//   * the context: g[p] = rms(main_proj [h40; h41; h42]) * main_norm, h_L the mean of layer L's 4 output streams at
//     position p; each stage's K = V row rope_p(rms(attn_kv g[p]) * attn_kv_a_norm), kept in a ring of kRing
//     positions (FP16, as the target's window keys); a draft reads the last kCtx of them;
//   * the block: [y, noise x 4] (y the pending token, the noise token's embedding after it) at positions pos ..
//     pos + 4 through the stages - the mHC read / write halves, attention over the context and the whole block (no
//     mask), the MoE (sqrt-softplus router + bias, 6 of 256 experts, weights x 1.5, the shared expert; the SwiGLU's
//     clamp 10) - then the hc head, the norm and the TARGET's output head (row i proposes position pos + i + 1);
//   * the Markov chain: draft i = argmax(logits_i + markov_w2 . markov_w1[draft i - 1]) (draft -1 = y), its
//     confidence sigmoid(conf_proj . [h_i; markov_w1[draft i - 1]]) (h_i before the norm).  The decode stops the
//     window at the first draft below the threshold.
// The context follows the sequence by itself: the token path, the verify window (the kept columns, at spec_rewind)
// and the prompt path (a chunk's last kCtx rows) leave their layer 40-42 outputs, dspark_commit turns them into ring
// rows.  The routed experts (2.1 GB a stage, Q2_K) live in a VRAM cache, STRATA_DS4_DSPARK_EXPERTS a stage (default
// all 256): an expert without a slot is left out of its row (the kept weights renormalized - a draft only proposes,
// the window decides), and after a draft the most used missing ones are copied in (dspark_upload, while the window
// runs), the least used out.
#include "glm_fast_state.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/glm_fast.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace ds = strata::kernels::dsv4;

namespace {
constexpr size_t kQ8 = 36;      // bytes of a q8_1 block (32 values)
constexpr int kB = 5;           // the block: y + 4 noise rows -> 5 drafts
constexpr int kRing = 256;      // the context ring's rows (positions mod kRing): kCtx + kB < kRing
constexpr int kCtx = 128;       // the context a draft reads
constexpr int kStages = 3;
constexpr int kL0 = 40;         // the target layers whose outputs are the features (40, 41, 42)
constexpr int kQ2K = 10, kQ8_0 = 8, kF16 = 1, kF32 = 0;
constexpr float kEps = 1e-6f, kClamp = 10.0f, kWScale = 1.5f;

struct DspStage {
    WSlot q_a, q_b, kv, out_a, out_b, sh_gate, sh_up, sh_down;   // Q8_0, on the device
    float *q_a_norm = nullptr, *kv_a_norm = nullptr, *sinks = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr;
    float *gate_inp = nullptr, *probs_b = nullptr;
    float *hc_attn_scale = nullptr, *hc_attn_base = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
    uint16_t *hc_attn_fn = nullptr, *hc_ffn_fn = nullptr;       // BF16 [24][4 n_embd]
    uint16_t* ring = nullptr;                                     // [kRing][head_dim] FP16 K = V rows
    unsigned long long* tab = nullptr;                            // [n_expert] the expert's VRAM slot (0: none)
    uint8_t* slots = nullptr;                                     // [n_slots] blobs [gate | up | down]
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;   // the GGUF's expert tensors (mapped)
    std::vector<int> slot_of, expert_in;                          // expert -> slot, slot -> expert (-1: none)
    std::vector<float> score;                                     // decayed use counts (the cache's policy)
};

struct DspUp {
    int s, e, slot, evict;
};

struct DSparkState {
    std::unique_ptr<strata::GgufFile> gguf;
    DspStage st[kStages];
    WSlot main_proj, markov_w2;
    float *main_norm = nullptr, *conf_proj = nullptr, *hc_head_scale = nullptr, *hc_head_base = nullptr,
          *norm = nullptr;
    uint16_t *markov_w1 = nullptr, *hc_head_fn = nullptr;
    int n_slots = 256, n_up = 4;
    size_t blob = 0, gu_rb = 0, dn_rb = 0, down_off = 0, dense_bytes = 0;
    ds::Rope rope;
    // features, [3][n_embd] a row: the token's / the window's columns (feat) and a prompt chunk's last kCtx rows
    float *feat = nullptr, *feat_p = nullptr;
    void* featq = nullptr;
    float *gpre = nullptr, *gv = nullptr, *kvc = nullptr;         // the commit: main_proj's rows, g, the stages' kv
    void* gq = nullptr;
    // the draft ([kB] rows)
    float *emb_noise = nullptr, *emb_y = nullptr, *R = nullptr, *x = nullptr, *pre = nullptr, *post = nullptr,
          *comb = nullptr, *hpart = nullptr, *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *q = nullptr,
          *o = nullptr, *oa = nullptr, *mixer = nullptr, *rlog = nullptr, *sh_g = nullptr, *sh_u = nullptr,
          *sh_out = nullptr, *ffn = nullptr, *head_h = nullptr, *head_x = nullptr, *logits = nullptr,
          *biased = nullptr, *conf = nullptr;
    void *xq = nullptr, *qr_q = nullptr, *oq = nullptr, *oaq = nullptr, *sh_hq = nullptr, *hq = nullptr,
         *head_xq = nullptr;
    unsigned int* hcnt = nullptr;
    size_t hpart_n = 0, hq_n = 0;
    unsigned long long* plan_ptr = nullptr;                       // [kB][8]
    float* plan_w = nullptr;
    int* zero = nullptr;                                          // the plans' cpu_flag (always 0)
    float* cpu_part = nullptr;
    int *ids_d = nullptr, *dtok = nullptr, *iota = nullptr;       // [3][kB][k] routes; [kB + 1] the chain; [2 kRing]
    std::vector<gf::MoeDev> md;
    // pinned host: the token in (host-mapped), the results, the cache's staging and table values
    int *y_h = nullptr, *y_d = nullptr, *res_tok = nullptr, *res_ids = nullptr;
    float *res_conf = nullptr, *emb_h = nullptr;
    uint8_t* stage_h = nullptr;
    unsigned long long* tabv_h = nullptr;
    cudaStream_t up = nullptr;
    cudaEvent_t ev_up = nullptr;
    std::vector<DspUp> ups;
    std::vector<void*> allocs;                                    // every device allocation (freed by dspark_destroy)
    int launch_err0 = 0;
    // the context: positions [ctx_lo, ctx_end) have ring rows; a verify window's columns wait for spec_rewind
    int64_t ctx_lo = 0, ctx_end = 0, win_p0 = -1;
    int win_T = 0;
    // statistics; STRATA_DS4_DSPARK_PROF=1: the device time of a draft's phases (events: start, each stage's attention
    // and MoE, the head, the Markov chain)
    uint64_t drafts = 0, uses = 0, misses = 0, uploads = 0, commits = 0;
    double ms_draft = 0.0;
    bool prof = false;
    static constexpr int kPh = 2 * kStages + 2;
    cudaEvent_t ev[kPh + 1] = {};
    double ms_ph[kPh] = {};
};

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);   // NaN stays NaN
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
}  // namespace

bool Glm5Model::dspark_load(std::string& err) {
    const char* path = getenv("STRATA_DS4_DSPARK");
    if (path == nullptr || !path[0]) return true;
    const Glm5Geometry& g = g_;
    const auto refuse = [&](const std::string& why) {
        std::fprintf(stderr, "dspark: not loaded - %s\n", why.c_str());
        dspark_destroy();
        return true;   // the target runs on without the drafter
    };
    if (!g.dsv4() || g.v41 || n_parts_ != 1 || l0_ != 0 || l1_ != g.n_layers || mtp_il_ >= 0)
        return refuse("DeepSeek V4 Flash on one device only (no split, no MTP)");
    if (g.n_layers <= kL0 + 2 || g.n_embd > 4096 || g.head_dim != 512 || g.n_expert > 512 || g.n_exp_used > 8)
        return refuse("the target's geometry (n_layers " + std::to_string(g.n_layers) + ", n_embd " +
                      std::to_string(g.n_embd) + ")");
    auto* D = new DSparkState();
    dsp_ = D;
    try {
        D->gguf = std::make_unique<strata::GgufFile>(path);
    } catch (const std::exception& e) {
        return refuse(std::string("cannot open ") + path + ": " + e.what());
    }
    const strata::GgufFile& G = *D->gguf;
    const auto num = [&](const char* key, double def) {
        const strata::MetaValue* v = G.get(key);
        return v != nullptr && v->is_num() ? v->num() : def;
    };
    {
        const strata::MetaValue* a = G.get("general.architecture");
        if (a == nullptr || a->s != "deepseek4-dspark") return refuse("not a deepseek4-dspark GGUF");
        const strata::MetaValue* tl = G.get("deepseek4.dspark.target_layers");
        bool layers_ok = tl != nullptr && tl->items.size() == 3;
        for (int i = 0; layers_ok && i < 3; ++i) layers_ok = (int) tl->items[(size_t) i].num() == kL0 + i;
        if (!layers_ok || (int) num("deepseek4.dspark.layer_count", 0) != kStages ||
            (int) num("deepseek4.dspark.block_size", 0) != kB || (int) num("deepseek4.dspark.markov_rank", 0) != 256 ||
            (int) num("deepseek4.dspark.expert_count", 0) != g.n_expert)
            return refuse("the drafter's layout (3 stages over layers 40-42, block 5, Markov rank 256, the target's "
                          "expert count)");
    }
    const int noise = (int) num("deepseek4.dspark.noise_token_id", -1);
    if (noise < 0 || noise >= g.n_vocab) return refuse("the noise token id");
    const int E = g.n_embd, NE = g.n_expert, K = g.n_exp_used, FF = g.n_ff_exp, V = g.n_vocab;
    const int HD = g.n_head * g.head_dim, OL = g.o_groups * g.o_lora, QL = g.q_lora, DH = g.head_dim;
    std::string why;
    const auto find = [&](const std::string& n, int type, std::initializer_list<uint64_t> shape) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = G.find(n);
        if (t == nullptr) { if (why.empty()) why = n + " missing"; return nullptr; }
        bool ok = (int) t->type == type && t->shape.size() == shape.size();
        size_t i = 0;
        for (uint64_t d : shape) ok = ok && t->shape[i++] == d;
        if (!ok && why.empty()) why = n + " has another type or shape";
        return ok ? t : nullptr;
    };
    bool ok = true;
    const auto dev = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; cudaGetLastError(); return nullptr; }
        D->allocs.push_back(p);
        D->dense_bytes += bytes;
        return p;
    };
    const auto raw = [&](const std::string& n, int type, int64_t n_in, int64_t n_out, WSlot& w) {
        const strata::TensorInfo* t = find(n, type, {(uint64_t) n_in, (uint64_t) n_out});
        if (t == nullptr) { ok = false; return; }
        const size_t bytes = gf::row_bytes(type, n_in) * (size_t) n_out;
        void* p = dev(bytes);
        if (p == nullptr) return;
        ok = cudaMemcpy(p, G.tensor_data(*t), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        w = WSlot{nullptr, type, p, n_in, n_out};
    };
    const auto f32 = [&](const std::string& n, std::initializer_list<uint64_t> shape) -> float* {
        const strata::TensorInfo* t = find(n, kF32, shape);
        if (t == nullptr) { ok = false; return nullptr; }
        float* p = (float*) dev(t->elements() * sizeof(float));
        if (p != nullptr)
            ok = cudaMemcpy(p, G.tensor_data(*t), t->elements() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        return p;
    };
    const auto bf16 = [&](const std::string& n, std::initializer_list<uint64_t> shape) -> uint16_t* {
        const strata::TensorInfo* t = find(n, kF32, shape);
        if (t == nullptr) { ok = false; return nullptr; }
        std::vector<uint16_t> h((size_t) t->elements());
        const float* src = (const float*) G.tensor_data(*t);
        for (size_t i = 0; i < h.size(); ++i) h[i] = to_bf16(src[i]);
        uint16_t* p = (uint16_t*) dev(h.size() * 2);
        if (p != nullptr) ok = cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess;
        return p;
    };
    const uint64_t uE = (uint64_t) E, u4E = 4 * uE;
    for (int s = 0; s < kStages && ok; ++s) {
        DspStage& S = D->st[s];
        const std::string P = "dspark." + std::to_string(s) + ".";
        raw(P + "attn_q_a.weight", kQ8_0, E, QL, S.q_a);
        raw(P + "attn_q_b.weight", kQ8_0, QL, HD, S.q_b);
        raw(P + "attn_kv.weight", kQ8_0, E, DH, S.kv);
        raw(P + "attn_output_a.weight", kQ8_0, (int64_t) HD / g.o_groups, OL, S.out_a);
        raw(P + "attn_output_b.weight", kQ8_0, OL, E, S.out_b);
        raw(P + "ffn_gate_shexp.weight", kQ8_0, E, FF, S.sh_gate);
        raw(P + "ffn_up_shexp.weight", kQ8_0, E, FF, S.sh_up);
        raw(P + "ffn_down_shexp.weight", kQ8_0, FF, E, S.sh_down);
        S.q_a_norm = f32(P + "attn_q_a_norm.weight", {(uint64_t) QL});
        S.kv_a_norm = f32(P + "attn_kv_a_norm.weight", {(uint64_t) DH});
        S.sinks = f32(P + "attn_sinks.weight", {(uint64_t) g.n_head});
        S.attn_norm = f32(P + "attn_norm.weight", {uE});
        S.ffn_norm = f32(P + "ffn_norm.weight", {uE});
        S.gate_inp = f32(P + "ffn_gate_inp.weight", {uE, (uint64_t) NE});
        S.probs_b = f32(P + "exp_probs_b.bias", {(uint64_t) NE});
        S.hc_attn_fn = bf16(P + "hc_attn_fn.weight", {u4E, 24});
        S.hc_attn_scale = f32(P + "hc_attn_scale.weight", {3});
        S.hc_attn_base = f32(P + "hc_attn_base.weight", {24});
        S.hc_ffn_fn = bf16(P + "hc_ffn_fn.weight", {u4E, 24});
        S.hc_ffn_scale = f32(P + "hc_ffn_scale.weight", {3});
        S.hc_ffn_base = f32(P + "hc_ffn_base.weight", {24});
        const strata::TensorInfo* tg = find(P + "ffn_gate_exps.weight", kQ2K, {uE, (uint64_t) FF, (uint64_t) NE});
        const strata::TensorInfo* tu = find(P + "ffn_up_exps.weight", kQ2K, {uE, (uint64_t) FF, (uint64_t) NE});
        const strata::TensorInfo* td = find(P + "ffn_down_exps.weight", kQ2K, {(uint64_t) FF, uE, (uint64_t) NE});
        if (tg == nullptr || tu == nullptr || td == nullptr) { ok = false; break; }
        S.src_g = G.tensor_data(*tg);
        S.src_u = G.tensor_data(*tu);
        S.src_d = G.tensor_data(*td);
        S.ring = (uint16_t*) dev((size_t) kRing * DH * sizeof(uint16_t));
        S.tab = (unsigned long long*) dev((size_t) NE * sizeof(unsigned long long));
    }
    raw("dspark.main_proj.weight", kQ8_0, 3 * E, E, D->main_proj);
    raw("dspark.markov_w2.weight", kQ8_0, 256, V, D->markov_w2);
    D->main_norm = f32("dspark.main_norm.weight", {uE});
    D->conf_proj = f32("dspark.conf_proj.weight", {uE + 256, 1});
    D->hc_head_fn = bf16("dspark.hc_head_fn.weight", {u4E, 4});
    D->hc_head_scale = f32("dspark.hc_head_scale.weight", {1});
    D->hc_head_base = f32("dspark.hc_head_base.weight", {4});
    D->norm = f32("dspark.norm.weight", {uE});
    if (ok) {
        const strata::TensorInfo* t = find("dspark.markov_w1.weight", kF16, {256, (uint64_t) V});
        if (t == nullptr) ok = false;
        D->markov_w1 = (uint16_t*) dev((size_t) V * 256 * 2);
        if (ok) ok = cudaMemcpy(D->markov_w1, G.tensor_data(*t), (size_t) V * 256 * 2, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) return refuse(why.empty() ? "the weights did not upload (VRAM?)" : why);
    if (!gf::moe_supported(kQ2K) || !gf::mvn_supported(kQ8_0)) return refuse("the Q2_K / Q8_0 kernels");
    // the noise token's embedding, once
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) return refuse("the target's embedding");
    {
        std::vector<float> e((size_t) E);
        tt->to_float(pack_emb_src_ + (size_t) noise * ggml_row_size((ggml_type) pack_emb_type_, E), e.data(), E);
        D->emb_noise = (float*) dev((size_t) E * sizeof(float));
        if (ok) ok = cudaMemcpy(D->emb_noise, e.data(), (size_t) E * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    // the buffers
    const auto fb = [&](float*& p, size_t n) { p = (float*) dev(n * sizeof(float)); };
    const auto bb = [&](void*& p, size_t n) { p = dev(n); };
    const size_t HCE = 4 * (size_t) E;
    fb(D->feat, (size_t) kVerifyMax * 3 * E);
    fb(D->feat_p, (size_t) kCtx * 3 * E);
    bb(D->featq, (size_t) kCtx * 3 * E / 32 * kQ8);
    fb(D->gpre, (size_t) kCtx * E);
    fb(D->gv, (size_t) kCtx * E);
    bb(D->gq, (size_t) kCtx * E / 32 * kQ8);
    fb(D->kvc, (size_t) kStages * kCtx * DH);
    fb(D->emb_y, E);
    fb(D->R, (size_t) kB * 2 * HCE);
    fb(D->x, (size_t) kB * E);
    bb(D->xq, (size_t) kB * E / 32 * kQ8);
    fb(D->pre, kB * 8);
    fb(D->post, kB * 8);
    fb(D->comb, kB * 16);
    D->hpart_n = (size_t) (4 * E / 512) * 36 + 64;
    fb(D->hpart, D->hpart_n * kB);
    D->hcnt = (unsigned int*) dev((size_t) kB * 64);
    fb(D->qr_raw, (size_t) kB * QL);
    fb(D->qr, (size_t) kB * QL);
    bb(D->qr_q, (size_t) kB * QL / 32 * kQ8);
    fb(D->kv_raw, (size_t) kB * DH);
    fb(D->q, (size_t) kB * HD);
    fb(D->o, (size_t) kB * HD);
    bb(D->oq, (size_t) kB * HD / 32 * kQ8);
    fb(D->oa, (size_t) kB * OL);
    bb(D->oaq, (size_t) kB * OL / 32 * kQ8);
    fb(D->mixer, (size_t) kB * E);
    fb(D->rlog, (size_t) kB * NE);
    fb(D->sh_g, (size_t) kB * FF);
    fb(D->sh_u, (size_t) kB * FF);
    bb(D->sh_hq, (size_t) kB * FF / 32 * kQ8);
    D->hq_n = (size_t) 8 * FF / 32 * kQ8;
    bb(D->hq, D->hq_n * kB);
    fb(D->sh_out, (size_t) kB * E);
    fb(D->ffn, (size_t) kB * E);
    fb(D->head_h, (size_t) kB * E);
    fb(D->head_x, (size_t) kB * E);
    bb(D->head_xq, (size_t) kB * E / 32 * kQ8);
    fb(D->logits, (size_t) kB * V);
    fb(D->biased, (size_t) V);
    fb(D->conf, 8);
    D->plan_ptr = (unsigned long long*) dev((size_t) kB * 8 * sizeof(unsigned long long));
    fb(D->plan_w, kB * 8);
    D->zero = (int*) dev(64);
    fb(D->cpu_part, E);
    D->ids_d = (int*) dev((size_t) kStages * kB * 8 * sizeof(int));
    D->dtok = (int*) dev(16 * sizeof(int));
    D->iota = (int*) dev(2 * kRing * sizeof(int));
    if (!ok) return refuse("the buffers did not allocate (VRAM?)");
    cudaMemset(D->hcnt, 0, (size_t) kB * 64);
    cudaMemset(D->zero, 0, 64);
    cudaMemset(D->plan_ptr, 0, (size_t) kB * 8 * sizeof(unsigned long long));
    {
        std::vector<int> io(2 * kRing);
        for (int i = 0; i < 2 * kRing; ++i) io[(size_t) i] = i % kRing;
        cudaMemcpy(D->iota, io.data(), io.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    D->md.assign(kB, gf::MoeDev{});
    for (int i = 0; i < kB; ++i) {
        gf::MoeDev& m = D->md[(size_t) i];
        m.plan_ptr = D->plan_ptr + i * 8;
        m.plan_w = D->plan_w + i * 8;
        m.cpu_flag = D->zero;
        m.cpu_part = D->cpu_part;
    }
    // the routed experts: a VRAM slot cache a stage
    D->gu_rb = gf::row_bytes(kQ2K, E);
    D->dn_rb = gf::row_bytes(kQ2K, FF);
    D->down_off = 2 * (size_t) FF * D->gu_rb;
    D->blob = (D->down_off + (size_t) E * D->dn_rb + 255) & ~(size_t) 255;
    if (const char* v = getenv("STRATA_DS4_DSPARK_EXPERTS")) D->n_slots = std::max(8, std::min(NE, std::atoi(v)));
    else D->n_slots = NE;
    if (const char* v = getenv("STRATA_DS4_DSPARK_UPLOADS")) D->n_up = std::max(0, std::min(8, std::atoi(v)));
    const auto blob_of = [&](const DspStage& S, int e, uint8_t* dst, cudaStream_t cs, bool async) {
        const size_t gu = (size_t) FF * D->gu_rb, dn = (size_t) E * D->dn_rb;
        const uint8_t* src[3] = {S.src_g + (size_t) e * gu, S.src_u + (size_t) e * gu, S.src_d + (size_t) e * dn};
        const size_t off[3] = {0, gu, D->down_off}, len[3] = {gu, gu, dn};
        for (int i = 0; i < 3; ++i) {
            if (async) cudaMemcpyAsync(dst + off[i], src[i], len[i], cudaMemcpyHostToDevice, cs);
            else if (cudaMemcpy(dst + off[i], src[i], len[i], cudaMemcpyHostToDevice) != cudaSuccess) return false;
        }
        return true;
    };
    std::vector<unsigned long long> tab((size_t) NE);
    for (int s = 0; s < kStages && ok; ++s) {
        DspStage& S = D->st[s];
        S.slots = (uint8_t*) dev((size_t) D->n_slots * D->blob);
        if (!ok) break;
        S.slot_of.assign((size_t) NE, -1);
        S.expert_in.assign((size_t) D->n_slots, -1);
        S.score.assign((size_t) NE, 0.0f);
        std::fill(tab.begin(), tab.end(), 0ull);
        for (int e = 0; e < D->n_slots && ok; ++e) {   // (the first n_slots experts until the uses say otherwise)
            uint8_t* dst = S.slots + (size_t) e * D->blob;
            ok = blob_of(S, e, dst, nullptr, false);
            S.slot_of[(size_t) e] = e;
            S.expert_in[(size_t) e] = e;
            tab[(size_t) e] = (unsigned long long) dst;
        }
        if (ok) ok = cudaMemcpy(S.tab, tab.data(), tab.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) return refuse("the expert cache did not allocate (" + std::to_string(D->n_slots) + " a stage)");
    // pinned host buffers, the upload stream
    const size_t n_stage = (size_t) std::max(1, D->n_up) * kStages;
    ok = cudaHostAlloc((void**) &D->y_h, 64, cudaHostAllocMapped) == cudaSuccess &&
         cudaHostGetDevicePointer((void**) &D->y_d, D->y_h, 0) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_tok, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_conf, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_ids, (size_t) kStages * kB * 8 * sizeof(int), cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->emb_h, (size_t) E * sizeof(float), cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->tabv_h, 2 * n_stage * sizeof(unsigned long long), cudaHostAllocDefault) == cudaSuccess &&
         (D->n_slots >= NE ||
          cudaHostAlloc((void**) &D->stage_h, n_stage * D->blob, cudaHostAllocDefault) == cudaSuccess) &&
         cudaStreamCreateWithFlags(&D->up, cudaStreamNonBlocking) == cudaSuccess &&
         cudaEventCreateWithFlags(&D->ev_up, cudaEventDisableTiming) == cudaSuccess;
    if (!ok) return refuse("the pinned buffers");
    cudaEventRecord(D->ev_up, D->up);
    D->launch_err0 = gf::launch_errors();
    D->prof = getenv("STRATA_DS4_DSPARK_PROF") != nullptr && std::atoi(getenv("STRATA_DS4_DSPARK_PROF")) != 0;
    for (int i = 0; D->prof && i <= DSparkState::kPh; ++i) cudaEventCreate(&D->ev[i]);
    D->rope = ds::make_rope(g.n_rot, 10000.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f);   // (the drafter's: base 10000, no YaRN)
    std::fprintf(stderr, "dspark: %s - %d stages over target layers %d-%d, %d drafts a step; %.2f GB on CUDA%d "
                         "(dense %.2f, experts %d of %d a stage %.2f)\n",
                 path, kStages, kL0, kL0 + 2, kB, (double) (D->dense_bytes) / 1073741824.0, dev_,
                 (double) (D->dense_bytes - (size_t) kStages * D->n_slots * D->blob) / 1073741824.0, D->n_slots, NE,
                 (double) kStages * D->n_slots * D->blob / 1073741824.0);
    return true;
}

void Glm5Model::dspark_destroy() {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) return;
    dsp_ = nullptr;
    if (D->up) cudaStreamSynchronize(D->up);
    for (cudaEvent_t e : D->ev)
        if (e) cudaEventDestroy(e);
    for (void* p : D->allocs) cudaFree(p);
    if (D->up) cudaStreamDestroy(D->up);
    if (D->ev_up) cudaEventDestroy(D->ev_up);
    for (void* p : {(void*) D->y_h, (void*) D->res_tok, (void*) D->res_conf, (void*) D->res_ids, (void*) D->emb_h,
                    (void*) D->tabv_h, (void*) D->stage_h})
        if (p) cudaFreeHost(p);
    delete D;
}

size_t Glm5Model::dspark_vram_bytes() const {
    const auto* D = (const DSparkState*) dsp_;
    return D != nullptr ? D->dense_bytes : 0;
}

void Glm5Model::dspark_report() const {
    const auto* D = (const DSparkState*) dsp_;
    if (D == nullptr || D->drafts == 0) return;
    std::fprintf(stderr, "dspark: %llu drafts, %.2f ms each (host to results); routed experts %llu, %.1f%% without a "
                         "slot (%d a stage), %llu uploads; %llu context commits\n",
                 (unsigned long long) D->drafts, D->ms_draft / (double) D->drafts, (unsigned long long) D->uses,
                 100.0 * (double) D->misses / (double) std::max<uint64_t>(1, D->uses), D->n_slots,
                 (unsigned long long) D->uploads, (unsigned long long) D->commits);
    if (D->prof) {
        static const char* names[DSparkState::kPh] = {"embed", "s0 attn", "s0 moe", "s1 attn", "s1 moe", "s2 attn",
                                                     "s2 moe", "head+markov"};
        std::fprintf(stderr, "dspark: device ms a draft -");
        double t = 0.0;
        for (int i = 0; i < DSparkState::kPh; ++i) {
            std::fprintf(stderr, " %s %.3f", names[i], D->ms_ph[i] / (double) D->drafts);
            t += D->ms_ph[i];
        }
        std::fprintf(stderr, " | total %.3f\n", t / (double) D->drafts);
    }
}

void Glm5Model::dspark_reset() {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) return;
    D->ctx_lo = D->ctx_end = pos_;
    D->win_p0 = -1;
}

void Glm5Model::dspark_capture(int il_out, const float* R, int col, cudaStream_t s) {
    auto* D = (DSparkState*) dsp_;
    const int k = il_out - kL0;
    if (D == nullptr || k < 0 || k > 2 || col < 0 || col >= kVerifyMax) return;
    const int E = g_.n_embd;
    ds::stream_mean(R, 1, 0, E, D->feat + ((size_t) col * 3 + k) * E, 0, s);
}

void Glm5Model::dspark_capture_rows(int il_out, const float* R, int T, cudaStream_t s) {
    auto* D = (DSparkState*) dsp_;
    const int k = il_out - kL0;
    if (D == nullptr || k < 0 || k > 2 || T <= 0) return;
    const int E = g_.n_embd, n = std::min(T, kCtx);
    ds::stream_mean(R + (size_t) (T - n) * 4 * E, n, (size_t) 4 * E, E, D->feat_p + (size_t) k * E, (size_t) 3 * E, s);
}

bool Glm5Model::dspark_commit(int64_t p0, int n, bool prompt, cudaStream_t s) {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr || n <= 0) return true;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, DH = g.head_dim;
    if (n > kCtx) {   // (only the last kCtx rows can be read)
        p0 += n - kCtx;
        n = kCtx;
    }
    const float* feat = prompt ? D->feat_p : D->feat;   // (the prompt rows: the chunk's last n, from row 0)
    gf::quantize_q8_1(feat, D->featq, n * 3 * E, s);
    gf::MvJob j = {D->main_proj.q, D->featq, feat, D->gpre, nullptr, 1.0f, D->main_proj.type, 3 * E, E};
    if (!gf::mvn(&j, 1, n, s)) return false;
    for (int c = 0; c < n; ++c)
        gf::rms_q8(D->gpre + (size_t) c * E, nullptr, D->main_norm, kEps, E, D->gv + (size_t) c * E,
                   (uint8_t*) D->gq + (size_t) c * (E / 32) * kQ8, s);
    gf::MvJob jk[kStages];
    for (int st = 0; st < kStages; ++st)
        jk[st] = {D->st[st].kv.q, D->gq, D->gv, D->kvc + (size_t) st * kCtx * DH, nullptr, 1.0f, D->st[st].kv.type, E, DH};
    if (!gf::mvn(jk, kStages, n, s)) return false;
    for (int st = 0; st < kStages; ++st)
        ds::kv_rows(D->kvc + (size_t) st * kCtx * DH, n, D->st[st].kv_a_norm, DH, g.n_rot, D->rope, (int) p0,
                    D->st[st].ring, kRing, kEps, s);
    if (p0 > D->ctx_end || p0 < D->ctx_lo) D->ctx_lo = p0;   // (a gap: the context starts over)
    D->ctx_end = p0 + n;
    ++D->commits;
    return true;
}

void Glm5Model::dspark_settle() {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr || D->win_p0 < 0) return;
    const int64_t p0 = D->win_p0;
    const int n = (int) std::max<int64_t>(0, std::min<int64_t>(D->win_T, pos_ - p0));
    D->win_p0 = -1;
    dspark_commit(p0, n, false, fast_->cs);
}

void Glm5Model::dspark_window(int64_t p0, int T) {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) return;
    D->win_p0 = p0;
    D->win_T = T;
}

void Glm5Model::spec_rewind(int64_t pos) {
    pos_ = pos;
    dspark_settle();
}

void Glm5Model::dspark_upload() {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr || D->ups.empty()) return;
    const int E = g_.n_embd, FF = g_.n_ff_exp;
    cudaEventSynchronize(D->ev_up);   // (the staging buffers of the last uploads; long done)
    const size_t gu = (size_t) FF * D->gu_rb, dn = (size_t) E * D->dn_rb;
    for (size_t i = 0; i < D->ups.size(); ++i) {
        const DspUp& u = D->ups[i];
        DspStage& S = D->st[u.s];
        uint8_t* slot = S.slots + (size_t) u.slot * D->blob;
        uint8_t* stg = D->stage_h + i * D->blob;
        // the evicted expert's table entry off first, then the bytes, then the new entry
        D->tabv_h[2 * i] = 0ull;
        D->tabv_h[2 * i + 1] = (unsigned long long) slot;
        if (u.evict >= 0)
            cudaMemcpyAsync(S.tab + u.evict, &D->tabv_h[2 * i], sizeof(unsigned long long), cudaMemcpyHostToDevice, D->up);
        std::memcpy(stg, S.src_g + (size_t) u.e * gu, gu);
        std::memcpy(stg + gu, S.src_u + (size_t) u.e * gu, gu);
        std::memcpy(stg + D->down_off, S.src_d + (size_t) u.e * dn, dn);
        cudaMemcpyAsync(slot, stg, D->down_off + dn, cudaMemcpyHostToDevice, D->up);
        cudaMemcpyAsync(S.tab + u.e, &D->tabv_h[2 * i + 1], sizeof(unsigned long long), cudaMemcpyHostToDevice, D->up);
        ++D->uploads;
    }
    D->ups.clear();
    cudaEventRecord(D->ev_up, D->up);
}

bool Glm5Model::dspark_draft(int32_t y, int32_t* drafts, float* conf, std::string& err) {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) { err = "dspark: not loaded"; return false; }
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto t0 = std::chrono::steady_clock::now();
    dspark_settle();
    dspark_upload();   // (a draft before the last one's uploads went out)
    const int64_t pos = pos_;
    const int E = g.n_embd, NE = g.n_expert, K = g.n_exp_used, FF = g.n_ff_exp, V = g.n_vocab, DH = g.head_dim;
    const int HD = g.n_head * DH, OL = g.o_groups * g.o_lora, QL = g.q_lora;
    const size_t HCE = 4 * (size_t) E, q8e = (size_t) (E / 32) * kQ8;
    // the context rows [lo, pos) when the ring holds every one of them, else none
    int64_t lo = std::max<int64_t>(D->ctx_lo, pos - kCtx);
    int n_ctx = D->ctx_end == pos ? (int) (pos - lo) : 0;
    if (n_ctx <= 0) {
        n_ctx = 0;
        lo = pos;
    }
    cudaStreamWaitEvent(s, D->ev_up, 0);   // the cache's last copies and table edits
    int ph = 0;
    const auto mark = [&] { if (D->prof) cudaEventRecord(D->ev[ph++], s); };
    mark();
    // the rows' streams: y's embedding, then the noise token's
    D->y_h[0] = y;
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (F->emb_tab_d != nullptr) {
        gf::embed_rows_q(pack_emb_type_, F->emb_tab_d, D->y_d, 1, E, D->emb_y, s);
    } else {
        tt->to_float(pack_emb_src_ + (size_t) y * ggml_row_size((ggml_type) pack_emb_type_, E), D->emb_h, E);
        cudaMemcpyAsync(D->emb_y, D->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    }
    cudaMemcpyAsync(D->dtok, D->y_h, sizeof(int), cudaMemcpyHostToDevice, s);
    float* Rc[kB];
    float* Ro[kB];
    for (int i = 0; i < kB; ++i) {
        Rc[i] = D->R + (size_t) i * 2 * HCE;
        Ro[i] = Rc[i] + HCE;
        gf::embed_streams(i == 0 ? D->emb_y : D->emb_noise, Rc[i], E, s);
    }
    const auto hc_read = [&](const DspStage& S, int i, const float* block_out, bool attn) {
        gf::HcArgs h;
        h.block_out = block_out;
        h.R_old = Rc[i];
        h.R_new = Ro[i];
        h.post_in = D->post + i * 8;
        h.comb_in = D->comb + i * 16;
        h.pre = D->pre + i * 8;
        h.post = D->post + i * 8;
        h.comb = D->comb + i * 16;
        h.w_fn = attn ? S.hc_attn_fn : S.hc_ffn_fn;
        h.w_scale = attn ? S.hc_attn_scale : S.hc_ffn_scale;
        h.w_base = attn ? S.hc_attn_base : S.hc_ffn_base;
        h.norm_w = attn ? S.attn_norm : S.ffn_norm;
        h.norm_eps = kEps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = E;
        h.x = D->x + (size_t) i * E;
        h.xq = (uint8_t*) D->xq + (size_t) i * q8e;
        h.part = D->hpart + (size_t) i * D->hpart_n;
        h.counter = D->hcnt + (size_t) i * 16;
        gf::hc(h, s);
        if (block_out != nullptr) std::swap(Rc[i], Ro[i]);
    };
    for (int st = 0; st < kStages; ++st) {
        const DspStage& S = D->st[st];
        mark();
        // ---- attention: the block's rows over the context and the whole block
        for (int i = 0; i < kB; ++i) hc_read(S, i, st == 0 ? nullptr : D->ffn + (size_t) i * E, true);
        gf::MvJob j[2] = {{S.q_a.q, D->xq, D->x, D->qr_raw, nullptr, 1.0f, S.q_a.type, E, QL},
                          {S.kv.q, D->xq, D->x, D->kv_raw, nullptr, 1.0f, S.kv.type, E, DH}};
        if (!gf::mvn(j, 2, kB, s)) { err = "dspark: q_a / kv"; return false; }
        for (int i = 0; i < kB; ++i) {
            ds::PrepArgs a;
            a.qr_raw = D->qr_raw + (size_t) i * QL;
            a.q_a_norm = S.q_a_norm;
            a.qr = D->qr + (size_t) i * QL;
            a.qr_q = (uint8_t*) D->qr_q + (size_t) i * (QL / 32) * kQ8;
            a.q_lora = QL;
            a.kv_raw = D->kv_raw + (size_t) i * DH;
            a.kv_norm = S.kv_a_norm;
            a.raw = S.ring;   // (row (pos + i) % kRing: past the context's rows, rewritten by later commits)
            a.ring = kRing;
            a.head_dim = DH;
            a.n_rot = g.n_rot;
            a.rope = D->rope;
            a.comp_rope = D->rope;
            a.p = (int) (pos + i);
            a.eps = kEps;
            ds::prep(a, s);
        }
        gf::MvJob jq = {S.q_b.q, D->qr_q, D->qr, D->q, nullptr, 1.0f, S.q_b.type, QL, HD};
        if (!gf::mvn(&jq, 1, kB, s)) { err = "dspark: q_b"; return false; }
        for (int i = 0; i < kB; ++i) {
            float* q = D->q + (size_t) i * HD;
            ds::q_prep(q, g.n_head, DH, g.n_rot, D->rope, nullptr, 0, 0, D->rope, (int) (pos + i), kEps, s);
            ds::attend(q, S.ring, kRing, 0, S.ring, D->iota + (lo % kRing), n_ctx + kB, S.sinks, g.n_head, DH, g.n_rot,
                       D->rope, (int) (pos + i), D->o + (size_t) i * HD, s, 0,
                       (uint8_t*) D->oq + (size_t) i * (HD / 32) * kQ8);
        }
        const int gin = g.n_head / g.o_groups * DH;
        const size_t wrow = gf::row_bytes(S.out_a.type, gin);
        gf::MvJob jo[gf::kMaxMvJobs];
        for (int gi = 0; gi < g.o_groups; ++gi)
            jo[gi] = {(const uint8_t*) S.out_a.q + (size_t) gi * g.o_lora * wrow,
                      (const uint8_t*) D->oq + (size_t) gi * (gin / 32) * kQ8, D->o + (size_t) gi * gin,
                      D->oa + (size_t) gi * g.o_lora, nullptr, 1.0f, S.out_a.type, gin, g.o_lora, HD / 32, OL};
        if (!gf::mvn(jo, g.o_groups, kB, s)) { err = "dspark: wo_a"; return false; }
        gf::quantize_q8_1(D->oa, D->oaq, kB * OL, s);
        gf::MvJob jb = {S.out_b.q, D->oaq, D->oa, D->mixer, nullptr, 1.0f, S.out_b.type, OL, E};
        if (!gf::mvn(&jb, 1, kB, s)) { err = "dspark: wo_b"; return false; }
        // ---- the MoE
        mark();
        for (int i = 0; i < kB; ++i) hc_read(S, i, D->mixer + (size_t) i * E, false);
        gf::MvJob jm[3] = {{S.gate_inp, nullptr, D->x, D->rlog, nullptr, 1.0f, gf::kTypeF32, E, NE},
                           {S.sh_gate.q, D->xq, nullptr, D->sh_g, nullptr, 1.0f, S.sh_gate.type, E, FF},
                           {S.sh_up.q, D->xq, nullptr, D->sh_u, nullptr, 1.0f, S.sh_up.type, E, FF}};
        if (!gf::mvn(jm, 3, kB, s)) { err = "dspark: router / shared expert"; return false; }
        if (!ds::dspark_route(D->rlog, S.probs_b, NE, K, kWScale, S.tab, kB, D->plan_ptr, D->plan_w,
                              D->ids_d + (size_t) st * kB * K, s)) {
            err = "dspark: router";
            return false;
        }
        gf::sh_swiglu(D->sh_g, D->sh_u, kClamp, kB * FF, D->sh_hq, s);
        for (int i = 0; i < kB; ++i) {
            const gf::MoeDev& md = D->md[(size_t) i];
            void* hq = (uint8_t*) D->hq + (size_t) i * D->hq_n;
            float* sh_out = D->sh_out + (size_t) i * E;
            gf::moe_gate_up(kQ2K, md, K, E, FF, kClamp, (const uint8_t*) D->xq + (size_t) i * q8e, hq, S.sh_down.q,
                            S.sh_down.type, (const uint8_t*) D->sh_hq + (size_t) i * (FF / 32) * kQ8, FF, sh_out, s);
            gf::moe_down(kQ2K, md, K, E, FF, D->down_off, hq, sh_out, D->ffn + (size_t) i * E, s);
        }
    }
    // ---- the head: the last write half, the hc head, the target's output rows, the Markov chain
    mark();
    for (int i = 0; i < kB; ++i) {
        gf::hc_post(D->ffn + (size_t) i * E, Rc[i], D->post + i * 8, D->comb + i * 16, E, Ro[i], s);
        ds::dspark_head(Ro[i], D->hc_head_fn, D->hc_head_scale, D->hc_head_base, g.hc_eps, D->norm, kEps, E,
                        D->head_h + (size_t) i * E, D->head_x + (size_t) i * E,
                        (uint8_t*) D->head_xq + (size_t) i * q8e, s);
    }
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, D->head_xq, D->head_x, D->logits, nullptr, 1.0f, ow.type, E, V};
    if (gf::mvn_supported(ow.type)) {
        if (!gf::mvn(&o, 1, kB, s)) { err = "dspark: output head"; return false; }
    } else {
        for (int i = 0; i < kB; ++i) {
            gf::MvJob ot = o;
            ot.xq = (const uint8_t*) D->head_xq + (size_t) i * q8e;
            ot.xf = D->head_x + (size_t) i * E;
            ot.y = D->logits + (size_t) i * V;
            if (!gf::mv(&ot, 1, s)) { err = "dspark: output head (type " + std::to_string(ow.type) + ")"; return false; }
        }
    }
    for (int i = 0; i < kB; ++i) {
        ds::dspark_markov(D->logits + (size_t) i * V, D->markov_w2.q, D->markov_w1, V, D->dtok + i, D->conf_proj,
                          D->head_h + (size_t) i * E, E, D->biased, D->conf + i, s);
        gf::argmax(D->biased, V, D->dtok + i + 1, s);
    }
    mark();
    cudaMemcpyAsync(D->res_tok, D->dtok + 1, kB * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(D->res_conf, D->conf, kB * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(D->res_ids, D->ids_d, (size_t) kStages * kB * K * sizeof(int), cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess || gf::launch_errors() != D->launch_err0) {
        err = std::string("dspark: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    for (int i = 0; i < kB; ++i) {
        drafts[i] = D->res_tok[i];
        conf[i] = D->res_conf[i];
    }
    for (int i = 0; D->prof && i < DSparkState::kPh; ++i) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, D->ev[i], D->ev[i + 1]);
        D->ms_ph[i] += ms;
    }
    // ---- the cache: decayed use counts; this draft's missing experts with the most uses in, the coldest out
    for (int st = 0; st < kStages; ++st) {
        DspStage& S = D->st[st];
        for (float& v : S.score) v *= 0.95f;
        std::vector<int> miss;
        for (int i = 0; i < kB * K; ++i) {
            const int e = D->res_ids[(size_t) st * kB * K + i];
            if (e < 0 || e >= NE) continue;
            S.score[(size_t) e] += 1.0f;
            ++D->uses;
            if (S.slot_of[(size_t) e] < 0) {
                ++D->misses;
                if (std::find(miss.begin(), miss.end(), e) == miss.end()) miss.push_back(e);
            }
        }
        if (D->n_slots >= NE || D->n_up == 0) continue;
        std::sort(miss.begin(), miss.end(), [&](int a, int b) { return S.score[(size_t) a] > S.score[(size_t) b]; });
        int n = 0;
        for (int e : miss) {
            if (n >= D->n_up) break;
            int slot = -1, cold = -1;
            for (int k = 0; k < D->n_slots; ++k) {
                const int c = S.expert_in[(size_t) k];
                if (c < 0) { slot = k; cold = -1; break; }
                if (slot < 0 || S.score[(size_t) c] < S.score[(size_t) cold]) { slot = k; cold = c; }
            }
            if (slot < 0 || (cold >= 0 && S.score[(size_t) cold] >= S.score[(size_t) e])) break;
            if (cold >= 0) S.slot_of[(size_t) cold] = -1;
            S.slot_of[(size_t) e] = slot;
            S.expert_in[(size_t) slot] = e;
            D->ups.push_back({st, e, slot, cold});
            ++n;
        }
    }
    ++D->drafts;
    D->ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::core
