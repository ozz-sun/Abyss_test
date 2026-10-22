// src/core/qwen35_fast_path.cu - the qwen35moe (Qwen3.5-MoE: Ornith-1.5-122B) pieces of the fast decode path
// (Glm5Model members).  A layer: x = rms(R) * attn_norm -> the mixer (a gated delta-net layer: gdn.hpp's recurrence
// with SiLU on the output gate; or a full-attention layer: GQA over an FP16 KV cache, q / k norms, NEOX RoPE on the
// first n_rot dims, a sigmoid gate from the q projection) -> R += mixer -> x = rms(R) * post_attention_norm -> the
// MoE (fast_moe: a softmax top-k router, the routed experts through the tiers and the lanes, the shared expert behind
// its sigmoid gate) -> R += ffn.  No hyper-connections: the residual R is state_ (n_embd floats).  The math follows
// llama.cpp's src/models/qwen35moe.cpp and delta-net-base.cpp.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/qwen35_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace q35 = strata::kernels::q35;

namespace {
size_t al(size_t b) { return (b + 255u) & ~(size_t) 255u; }
}  // namespace

void Glm5Model::qwen35_bind(int il, std::string& missing) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto& Ly = F->L[(size_t) il];
    const std::string P = "blk." + std::to_string(il) + ".";
    const auto f32 = [&](const char* n) -> const float* {
        auto it = w_.find(P + n);
        if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
        return it->second;
    };
    const auto q = [&](const char* n, int64_t n_in, int64_t n_out) -> WSlot {
        auto it = ws_map_.find(P + n);
        if (it == ws_map_.end() || it->second.type == 0 || !gf::mv_supported(it->second.type)) {
            missing += P + n + " (quant) ";
            return WSlot{};
        }
        if ((it->second.n_in != 0 && it->second.n_in != n_in) || (it->second.n_out != 0 && it->second.n_out != n_out))
            missing += P + n + " (shape " + std::to_string(it->second.n_in) + "x" + std::to_string(it->second.n_out) +
                       ", expected " + std::to_string(n_in) + "x" + std::to_string(n_out) + ") ";
        return it->second;
    };
    const int E = g.n_embd, C = g.ssm_conv_dim(), V = g.ssm_inner, D = g.head_dim;
    Ly.attn_norm = f32("attn_norm.weight");
    Ly.post_norm = f32("post_attention_norm.weight");
    if (Ly.recr) {
        Ly.qkv = q("attn_qkv.weight", E, C);
        Ly.zg = q("attn_gate.weight", E, V);
        Ly.ssm_o = q("ssm_out.weight", V, E);
        Ly.alpha_w = q("ssm_alpha.weight", E, g.ssm_v_heads);
        Ly.beta_w = q("ssm_beta.weight", E, g.ssm_v_heads);
        Ly.conv1d = f32("ssm_conv1d.weight");
        Ly.dt_bias = f32("ssm_dt.bias");
        Ly.ssm_a = f32("ssm_a");
        Ly.ssm_norm = f32("ssm_norm.weight");
    } else {
        Ly.aq = q("attn_q.weight", E, (int64_t) g.n_head * 2 * D);
        Ly.ak = q("attn_k.weight", E, (int64_t) g.attn_head_kv * D);
        Ly.av = q("attn_v.weight", E, (int64_t) g.attn_head_kv * D);
        Ly.out = q("attn_output.weight", (int64_t) g.n_head * D, E);
        Ly.q_norm = f32("attn_q_norm.weight");
        Ly.k_norm = f32("attn_k_norm.weight");
    }
    if (Ly.moe && g.n_shared > 0) Ly.sh_gate_w = f32("ffn_gate_inp_shexp.weight");
    if (Ly.mtp) {   // the NextN block: eh_proj in (2 n_embd -> n_embd) and its two norms; the head's norm the trunk's
        Ly.eh = q("nextn.eh_proj.weight", 2 * E, E);
        Ly.enorm = f32("nextn.enorm.weight");
        Ly.hnorm = f32("nextn.hnorm.weight");
    }
}

bool Glm5Model::qwen35_alloc(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const size_t S = (size_t) g.ssm_state, HV = (size_t) g.ssm_v_heads, C = (size_t) g.ssm_conv_dim(), V = (size_t) g.ssm_inner;
    const size_t D = (size_t) g.head_dim, HD = (size_t) g.n_head * D, KV = (size_t) g.attn_head_kv * D;
    const size_t ctx = (size_t) std::max<int64_t>(max_ctx_, 1);
    const size_t state_b = al(S * HV * S * 4), conv_b = al(C * (size_t) (g.d_conv - 1) * 4), cache_b = al(ctx * KV * 2);
    int n_rec = 0, n_attn = 0;
    for (int il = l0_; il < lt_; ++il) (F->L[(size_t) il].recr ? n_rec : n_attn) += 1;   // (lt_: the NextN block too)
    const size_t part = q35::attn_part_floats(g.n_head, g.head_dim, (int) ctx);
    size_t bytes = (size_t) n_rec * (state_b + conv_b) + (size_t) n_attn * 2 * cache_b;
    bytes += al(C * 4) + al(V * 4) + al(C * 4) + al(2 * HV * 4) + al(HV * 4) + al(V * 4) + al(V * 4) + al(V / 32 * 36) +
             al(HD * 2 * 4) + al(KV * 4) + al(KV * 4) + al(HD * 4) + al(HD * 4) + al(HD / 32 * 36) + al(part * 4) + 256;
    const size_t E = (size_t) g.n_embd, TM = (size_t) kVerifyMax;
    const bool mtp = mtp_il_ >= 0;
    if (mtp)   // the drafter: the pending hidden, <= kVerifyMax rows of scratch, the chain's tokens and confidences
        bytes += al(E * 4) + al(TM * E * 4) * 3 + al(TM * (E / 32) * 36) + al(TM * (2 * E / 32) * 36) + al(TM * KV * 4) * 2 +
                 al(TM * 4) * 2;
    if (cudaMalloc(&qwen35_state_, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "qwen35moe: the states and caches (" + std::to_string(bytes >> 20) + " MB for " + std::to_string(ctx) +
              " tokens) did not allocate";
        return false;
    }
    cudaMemset(qwen35_state_, 0, bytes);
    uint8_t* at = (uint8_t*) qwen35_state_;
    const auto take = [&](size_t b) {
        uint8_t* p = at;
        at += al(b);
        return p;
    };
    for (int il = l0_; il < lt_; ++il) {
        auto& Ly = F->L[(size_t) il];
        if (Ly.recr) {
            Ly.gdn_state = (float*) take(S * HV * S * 4);
            Ly.gdn_conv = (float*) take(C * (size_t) (g.d_conv - 1) * 4);
        } else {
            Ly.kc = (uint16_t*) take(ctx * KV * 2);
            Ly.vc = (uint16_t*) take(ctx * KV * 2);
        }
    }
    F->q35_qkv = (float*) take(C * 4);
    F->q35_z = (float*) take(V * 4);
    F->q35_h = (float*) take(C * 4);
    F->q35_ab = (float*) take(2 * HV * 4);
    F->q35_gate = (float*) take(HV * 4);
    F->q35_o = (float*) take(V * 4);
    F->q35_y = (float*) take(V * 4);
    F->q35_yq = take(V / 32 * 36);
    F->q35_qf = (float*) take(HD * 2 * 4);
    F->q35_k = (float*) take(KV * 4);
    F->q35_v = (float*) take(KV * 4);
    F->q35_q = (float*) take(HD * 4);
    F->q35_ao = (float*) take(HD * 4);
    F->q35_aoq = take(HD / 32 * 36);
    F->q35_part = (float*) take(part * 4);
    F->q35_shg = (float*) take(256);
    if (mtp) {
        F->q35_pend_h = (float*) take(E * 4);
        F->q35_mt_emb = (float*) take(TM * E * 4);
        F->q35_mt_hid = (float*) take(TM * E * 4);
        F->q35_mt_x = (float*) take(TM * E * 4);
        F->q35_mt_xq = take(TM * (E / 32) * 36);
        F->q35_mt_cat = take(TM * (2 * E / 32) * 36);
        F->q35_mt_k = (float*) take(TM * KV * 4);
        F->q35_mt_v = (float*) take(TM * KV * 4);
        F->q35_mt_conf = (float*) take(TM * 4);
        F->q35_mt_tok = (int*) take(TM * 4);
        if (cudaHostAlloc((void**) &F->q35_mt_conf_h, 256, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &F->q35_mt_conf_d, F->q35_mt_conf_h, 0) != cudaSuccess) {
            err = "qwen35moe: the drafter's host-mapped word did not allocate";
            return false;
        }
    }
    std::fprintf(stderr, "qwen35moe: CUDA%d %d delta-net layers (states %.0f MB), %d attention layers (KV caches %.2f GB "
                         "for %lld tokens)\n", dev_, n_rec, (double) n_rec * (state_b + conv_b) / 1048576.0, n_attn,
                 (double) n_attn * 2 * cache_b / 1073741824.0, (long long) ctx);
    return true;
}

void Glm5Model::qwen35_reset() {
    FastState* F = fast_;
    if (F == nullptr || qwen35_state_ == nullptr) return;
    const Glm5Geometry& g = g_;
    const size_t S = (size_t) g.ssm_state, HV = (size_t) g.ssm_v_heads, C = (size_t) g.ssm_conv_dim();
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.recr) continue;
        cudaMemsetAsync(Ly.gdn_state, 0, S * HV * S * 4, F->cs);
        cudaMemsetAsync(Ly.gdn_conv, 0, C * (size_t) (g.d_conv - 1) * 4, F->cs);
    }
    q35_pend_pos_ = q35_row_done_ = -1;
    cudaStreamSynchronize(F->cs);
}

// ---------------------------------------------------------------- the NextN block as the drafter
// The block (llama.cpp's graph_mtp): h = the trunk's output_norm'd hidden at a position, e = the NEXT token's
// embedding -> eh_proj([enorm(e) | hnorm(h)]) -> an attention layer (its own KV cache, a row a position) -> the MoE ->
// output_norm -> the trunk's head: the logits of the token after the next.  A cache row needs only the position's
// (h, next token) through eh_proj and k / v, so a position waits (q35_pend_pos_) until its next token arrives
// (fast_token, a window's first token, or a draft), then its row is written; a draft step at the pending position
// writes the row itself.  Chained drafts feed the block its own normed output as h and its argmax as the token.
bool Glm5Model::q35_mtp_on() const {
    return g_.qwen35() && mtp_il_ >= 0 && fast_ != nullptr && fast_->emb_tab_d != nullptr && fast_->q35_pend_h != nullptr;
}

void Glm5Model::q35_mtp_pend_set(int64_t p, const float* h) {
    FastState* F = fast_;
    cudaMemcpyAsync(F->q35_pend_h, h, (size_t) g_.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, F->cs);
    q35_pend_pos_ = p;
}

bool Glm5Model::q35_mtp_pend_commit(int32_t next, std::string& err) {
    if (q35_pend_pos_ < 0 || q35_pend_pos_ <= q35_row_done_) {   // (none, or a draft step wrote it)
        q35_pend_pos_ = -1;
        return true;
    }
    const int64_t p = q35_pend_pos_;
    q35_pend_pos_ = -1;
    return q35_mtp_rows(p, 1, fast_->q35_pend_h, 0, &next, err);
}

// the cache rows of positions p0 .. p0 + n - 1 (n <= kVerifyMax) from their normed hiddens (h_stride floats apart) and
// their next tokens: eh_proj over the rows in one mvn launch, the attention norm a row, k / v in one launch
bool Glm5Model::q35_mtp_rows(int64_t p0, int n, const float* h_rows, size_t h_stride, const int32_t* next, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    if (n <= 0) return true;
    if (n > kVerifyMax) { err = "qwen35 mtp: " + std::to_string(n) + " rows (at most " + std::to_string(kVerifyMax) + ")"; return false; }
    const int il = mtp_il_, E = g.n_embd, D = g.head_dim, KV = g.attn_head_kv * D;
    const size_t q8e = (size_t) (E / 32) * 36, cat_b = (size_t) (2 * E / 32) * 36;
    const auto& Ly = F->L[(size_t) il];
    if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "qwen35 mtp rows", err)) return false;   // (tok_in_h may feed an earlier copy)
    for (int t = 0; t < n; ++t) F->tok_in_h[t] = next[t];
    if (!gf::embed_rows_q(pack_emb_type_, F->emb_tab_d, F->tok_in_d, n, E, F->q35_mt_emb, s)) { err = "qwen35 mtp: the embedding rows"; return false; }
    for (int t = 0; t < n; ++t)
        gf::mtp_in(F->q35_mt_emb + (size_t) t * E, h_rows + (size_t) t * h_stride, Ly.enorm, Ly.hnorm, g.norm_eps, E,
                   (uint8_t*) F->q35_mt_cat + (size_t) t * cat_b, s);
    gf::MvJob eh = {Ly.eh.q, F->q35_mt_cat, nullptr, F->q35_mt_hid, nullptr, 1.0f, Ly.eh.type, 2 * E, E};
    if (!gf::mvn(&eh, 1, n, s)) { err = "qwen35 mtp: eh_proj rows"; return false; }
    for (int t = 0; t < n; ++t)
        gf::rms_q8(F->q35_mt_hid + (size_t) t * E, nullptr, Ly.attn_norm, g.norm_eps, E, F->q35_mt_x + (size_t) t * E,
                   (uint8_t*) F->q35_mt_xq + (size_t) t * q8e, s);
    gf::MvJob j[2];
    j[0] = {Ly.ak.q, F->q35_mt_xq, F->q35_mt_x, F->q35_mt_k, nullptr, 1.0f, Ly.ak.type, E, KV};
    j[1] = {Ly.av.q, F->q35_mt_xq, F->q35_mt_x, F->q35_mt_v, nullptr, 1.0f, Ly.av.type, E, KV};
    if (!gf::mvn(j, 2, n, s)) { err = "qwen35 mtp: k / v rows"; return false; }
    q35::attn_kv_rows(F->q35_mt_k, F->q35_mt_v, Ly.k_norm, g.norm_eps, g.attn_head_kv, D, g.n_rot, g.rope_base, p0, n,
                      Ly.kc, Ly.vc, s);
    q35_row_done_ = std::max(q35_row_done_, p0 + n - 1);
    return true;
}

// one draft step at position p: the block whole (its row written on the way), the head, the argmax (tok_out, on the
// device) and its probability (conf_out); F->head_x holds the block's normed output for a chained step
bool Glm5Model::q35_mtp_step(int64_t p, const int* tok_dev, const float* h, float* conf_out, int* tok_out, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int il = mtp_il_, E = g.n_embd;
    const auto& Ly = F->L[(size_t) il];
    if (!gf::embed_rows_q(pack_emb_type_, F->emb_tab_d, tok_dev, 1, E, F->emb, s)) { err = "qwen35 mtp: the embedding"; return false; }
    gf::mtp_in(F->emb, h, Ly.enorm, Ly.hnorm, g.norm_eps, E, F->mtp_catq, s);
    gf::MvJob eh = {Ly.eh.q, F->mtp_catq, nullptr, F->mtp_h, nullptr, 1.0f, Ly.eh.type, 2 * E, E};
    if (!gf::mv(&eh, 1, s)) { err = "qwen35 mtp: eh_proj"; return false; }
    gf::rms_q8(F->mtp_h, nullptr, Ly.attn_norm, g.norm_eps, E, F->x, F->xq, s);
    if (!fast_qwen35_attn(il, p, err)) return false;
    gf::rms_q8(F->mtp_h, F->mixer, Ly.post_norm, g.norm_eps, E, F->x, F->xq, s);
    bool pf_pending = false;
    if (!fast_moe(il, pf_pending, err)) return false;
    gf::rms_q8(F->mtp_h, F->ffn, w_.at("output_norm.weight"), g.norm_eps, E, F->head_x, F->head_xq, s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, F->head_xq, F->head_x, F->mtp_logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, s)) { err = "qwen35 mtp: the head"; return false; }
    gf::argmax(F->mtp_logits, g.n_vocab, F->mtp_tok, s);
    q35::top_prob(F->mtp_logits, g.n_vocab, F->mtp_tok, conf_out, s);
    cudaMemcpyAsync(tok_out, F->mtp_tok, sizeof(int), cudaMemcpyDeviceToDevice, s);
    return true;
}

// the drafts after y (the token at position(), pending): up to STRATA_Q35_MTP_K (1..5, default 3) chained steps from
// the pending position, each the block over the previous step's argmax and normed output; a draft's confidence is
// its softmax probability, read after every step (host-mapped) - the chain stops at the first below
// STRATA_Q35_MTP_MIN (default STRATA_DS4_DSPARK_THR, else the decode's 0.5 for this drafter: it drops the rest anyway)
bool Glm5Model::q35_mtp_draft(int32_t y, int32_t* drafts, float* conf, std::string& err) {
    FastState* F = fast_;
    for (int k = 0; k < 5; ++k) {
        drafts[k] = -1;
        conf[k] = 0.0f;
    }
    if (q35_pend_pos_ < 0) return true;   // (no position to draft from)
    static const int kmax = [] {
        const char* v = getenv("STRATA_Q35_MTP_K");
        return v != nullptr ? std::max(1, std::min(5, std::atoi(v))) : 3;
    }();
    static const float stop_below = [] {
        const char* v = getenv("STRATA_Q35_MTP_MIN");
        if (v == nullptr) v = getenv("STRATA_DS4_DSPARK_THR");
        return v != nullptr ? (float) std::atof(v) : 0.5f;
    }();
    cudaStream_t s = F->cs;
    const int64_t p = q35_pend_pos_;
    if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "qwen35 mtp draft", err)) return false;   // (tok_in_h)
    F->tok_in_h[0] = y;
    int n = 0;
    for (int k = 0; k < kmax; ++k) {
        if (!q35_mtp_step(p + k, k == 0 ? F->tok_in_d : F->mtp_tok, k == 0 ? F->q35_pend_h : F->head_x,
                          F->q35_mt_conf_d, F->q35_mt_tok + k, err))
            return false;
        if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "qwen35 mtp draft step", err)) return false;
        conf[k] = F->q35_mt_conf_h[0];
        n = k + 1;
        if (conf[k] < stop_below) break;
    }
    int toks[5];
    cudaMemcpy(toks, F->q35_mt_tok, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost);
    if (!F->route_ok(err)) return false;
    for (int k = 0; k < n; ++k) drafts[k] = toks[k];
    q35_row_done_ = std::max(q35_row_done_, p);   // (the step at p wrote its row from the real h and y)
    q35_pend_pos_ = -1;
    return true;
}

// GLM_CB_DIR seam dumps under llama.cpp's cb names (<name>-<layer>.f32; with GLM_CB_POS=1 <name>-<layer>@<pos>.f32,
// else every position overwrites): synchronous, debug only - the diff script bisects the two engines by op
static int64_t q35_dump_pos = 0;
static void q35_dump(cudaStream_t s, const char* name, int il, const float* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    static const bool per_pos = getenv("GLM_CB_POS") != nullptr;
    if (dir == nullptr || !dir[0]) return;
    cudaStreamSynchronize(s);
    std::vector<float> buf((size_t) n);
    cudaMemcpy(buf.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    const std::string fn = std::string(dir) + "/" + name + "-" + std::to_string(il) +
                           (per_pos ? "@" + std::to_string(q35_dump_pos) : "") + ".f32";
    if (FILE* f = std::fopen(fn.c_str(), "wb")) {
        std::fwrite(buf.data(), 4, (size_t) n, f);
        std::fclose(f);
    }
}

bool Glm5Model::fast_qwen35_gdn(int il, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, C = g.ssm_conv_dim(), V = g.ssm_inner, S = g.ssm_state, HK = g.ssm_k_heads, HV = g.ssm_v_heads;
    // 1. the projections: qkv (C), z (V), alpha and beta (h_v each) in one launch
    gf::MvJob j[4];
    j[0] = {Ly.qkv.q, F->xq, F->x, F->q35_qkv, nullptr, 1.0f, Ly.qkv.type, E, C};
    j[1] = {Ly.zg.q, F->xq, F->x, F->q35_z, nullptr, 1.0f, Ly.zg.type, E, V};
    j[2] = {Ly.alpha_w.q, F->xq, F->x, F->q35_ab, nullptr, 1.0f, Ly.alpha_w.type, E, HV};
    j[3] = {Ly.beta_w.q, F->xq, F->x, F->q35_ab + HV, nullptr, 1.0f, Ly.beta_w.type, E, HV};
    if (!gf::mv(j, 4, s)) { err = "qwen35moe: the delta-net projections"; return false; }
    if (F->prof_on) F->mark("q35_gdn_mv");
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (dumps) {
        q35_dump(s, "linear_attn_qkv_mixed", il, F->q35_qkv, C);
        q35_dump(s, "z", il, F->q35_z, V);
        q35_dump(s, "alpha", il, F->q35_ab, HV);
        q35_dump(s, "beta", il, F->q35_ab + HV, HV);
    }
    // 2. the conv over [state | x] (the state slides), SiLU on the whole output, q / k L2-normed, q scaled by 1/sqrt(S)
    q35::conv_silu_step(Ly.gdn_conv, F->q35_qkv, Ly.conv1d, C, g.d_conv, F->q35_h, s);
    if (dumps) {   // (llama.cpp's q_conv / k_conv are the views before the L2 norm, *_predelta after it)
        q35_dump(s, "conv_output_silu", il, F->q35_h, C);
        q35_dump(s, "q_conv", il, F->q35_h, (int64_t) HK * S);
        q35_dump(s, "k_conv", il, F->q35_h + (size_t) HK * S, (int64_t) HK * S);
        q35_dump(s, "v_conv", il, F->q35_h + (size_t) 2 * HK * S, (int64_t) HV * S);
    }
    // the q and k heads L2-normed, q scaled, the gates (softplus(alpha + dt) * A, sigmoid(beta)) in one launch
    // (the gate lands over alpha: F->q35_ab [0, HV), beta's sigmoid over beta: [HV, 2 HV))
    q35::gdn_prep_gates(F->q35_h, 1, C, HK, S, g.norm_eps, F->q35_ab, F->q35_ab + HV, Ly.dt_bias, Ly.ssm_a, HV, s);
    if (dumps) {
        q35_dump(s, "q_conv_predelta", il, F->q35_h, (int64_t) HK * S);
        q35_dump(s, "k_conv_predelta", il, F->q35_h + (size_t) HK * S, (int64_t) HK * S);
    }
    // 3. the recurrence (the state in place), the gated norm, the output projection
    if (dumps) {
        q35_dump(s, "gate", il, F->q35_ab, HV);
        q35_dump(s, "beta_sigmoid", il, F->q35_ab + HV, HV);
    }
    // the recurrence: the prompt path's scan over one row (a block a value head, the state column in registers: the
    // state read and written once, 8 MB a layer) - gdn.hpp's two-pass kernel behind STRATA_Q35_GDN_STEP=1 (the A/B)
    static const bool old_step = getenv("STRATA_Q35_GDN_STEP") != nullptr;
    if (old_step) {
        strata::kernels::GdnShapes sh;
        sh.S = S;
        sh.h_k = HK;
        sh.h_v = HV;
        strata::kernels::gdn_step(Ly.gdn_state, F->q35_h, F->q35_h + (size_t) HK * S, F->q35_h + (size_t) 2 * HK * S,
                                  F->q35_ab, F->q35_ab + HV, F->q35_o, sh, (void*) s);
    } else {
        q35::gdn_scan(Ly.gdn_state, F->q35_h, 1, C, F->q35_ab, F->q35_ab + HV, S, HK, HV, F->q35_o, s);
    }
    if (dumps) q35_dump(s, "gdn_o", il, F->q35_o, V);
    strata::kernels::gdn_out_norm_silu(F->q35_o, F->q35_z, Ly.ssm_norm, F->q35_y, HV, S, g.norm_eps, (void*) s);
    if (F->prof_on) F->mark("q35_gdn");
    if (dumps) q35_dump(s, "final_output", il, F->q35_y, V);
    gf::quantize_q8_1(F->q35_y, F->q35_yq, V, s);
    gf::MvJob o = {Ly.ssm_o.q, F->q35_yq, F->q35_y, F->mixer, nullptr, 1.0f, Ly.ssm_o.type, V, E};
    if (!gf::mv(&o, 1, s)) { err = "qwen35moe: ssm_out"; return false; }
    if (F->prof_on) F->mark("q35_gdn_out");
    if (dumps) q35_dump(s, "linear_attn_out", il, F->mixer, E);
    return true;
}

bool Glm5Model::fast_qwen35_attn(int il, int64_t p64, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, D = g.head_dim, HD = g.n_head * D, KV = g.attn_head_kv * D, p = (int) p64;
    gf::MvJob j[3];
    j[0] = {Ly.aq.q, F->xq, F->x, F->q35_qf, nullptr, 1.0f, Ly.aq.type, E, 2 * HD};
    j[1] = {Ly.ak.q, F->xq, F->x, F->q35_k, nullptr, 1.0f, Ly.ak.type, E, KV};
    j[2] = {Ly.av.q, F->xq, F->x, F->q35_v, nullptr, 1.0f, Ly.av.type, E, KV};
    if (!gf::mv(j, 3, s)) { err = "qwen35moe: the attention projections"; return false; }
    if (F->prof_on) F->mark("q35_attn_mv");
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (dumps) {
        q35_dump(s, "Qcur_full", il, F->q35_qf, 2 * HD);
        q35_dump(s, "Kcur_raw", il, F->q35_k, KV);
        q35_dump(s, "Vcur", il, F->q35_v, KV);
    }
    q35::attn_prep(F->q35_qf, F->q35_k, F->q35_v, Ly.q_norm, Ly.k_norm, g.norm_eps, g.n_head, g.attn_head_kv, D, g.n_rot,
                   g.rope_base, p, Ly.kc, Ly.vc, F->q35_q, s);
    q35::attn_decode(F->q35_q, Ly.kc, Ly.vc, g.n_head, g.attn_head_kv, D, p + 1, 1.0f / std::sqrt((float) D), F->q35_part,
                     F->q35_ao, s);
    if (dumps) {
        q35_dump(s, "Qcur", il, F->q35_q, HD);
        q35_dump(s, "attn_pregate", il, F->q35_ao, HD);
    }
    q35::attn_gate(F->q35_ao, F->q35_qf, g.n_head, D, s);
    if (F->prof_on) F->mark("q35_attend");
    if (dumps) q35_dump(s, "attn_gated", il, F->q35_ao, HD);
    gf::quantize_q8_1(F->q35_ao, F->q35_aoq, HD, s);
    gf::MvJob o = {Ly.out.q, F->q35_aoq, F->q35_ao, F->mixer, nullptr, 1.0f, Ly.out.type, HD, E};
    if (!gf::mv(&o, 1, s)) { err = "qwen35moe: attn_output"; return false; }
    if (F->prof_on) F->mark("q35_attn_out");
    if (dumps) q35_dump(s, "attn_output", il, F->mixer, E);
    return true;
}

bool Glm5Model::fast_qwen35_layers(int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    bool pf_pending = false;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    q35_dump_pos = p;
    if (F->prof_on) F->mark("start");
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        // the residual R (state_) takes the last FFN's output, then x = rms(R) * attn_norm (and its q8_1)
        gf::rms_q8(state_, il == l0_ ? nullptr : F->ffn, Ly.attn_norm, g.norm_eps, g.n_embd, F->x, F->xq, s);
        if (F->prof_on) F->mark("q35_norm");
        if (dumps) {
            if (il > l0_) q35_dump(s, "l_out", il - 1, state_, g.n_embd);
            else q35_dump(s, "inp_embd", il, state_, g.n_embd);
            q35_dump(s, "attn_norm", il, F->x, g.n_embd);
        }
        if (Ly.recr) {
            if (!fast_qwen35_gdn(il, err)) return false;
        } else {
            if (!fast_qwen35_attn(il, p, err)) return false;
        }
        gf::rms_q8(state_, F->mixer, Ly.post_norm, g.norm_eps, g.n_embd, F->x, F->xq, s);
        if (F->prof_on) F->mark("q35_norm");
        if (dumps) {
            q35_dump(s, "attn_residual", il, state_, g.n_embd);
            q35_dump(s, "attn_post_norm", il, F->x, g.n_embd);
        }
        if (!Ly.moe) { err = "qwen35moe: a dense FFN layer (not supported)"; return false; }
        if (!fast_moe(il, pf_pending, err)) return false;
        if (dumps) {
            q35_dump(s, "ffn_shexp_gated", il, F->sh_out, g.n_embd);
            q35_dump(s, "ffn_out", il, F->ffn, g.n_embd);
        }
    }
    // the head: R += the last FFN, x = rms(R) * output_norm -> head_x / head_xq (fast_token runs the output rows)
    const auto nw = w_.find("output_norm.weight");
    if (nw == w_.end()) { err = "qwen35moe: output_norm.weight"; return false; }
    gf::rms_q8(state_, F->ffn, nw->second, g.norm_eps, g.n_embd, F->head_x, F->head_xq, s);
    if (F->prof_on) F->mark("q35_head");
    if (dumps) {
        q35_dump(s, "l_out", l1_ - 1, state_, g.n_embd);
        q35_dump(s, "result_norm", l1_ - 1, F->head_x, g.n_embd);
    }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("qwen35moe: ") + cudaGetErrorString(e); return false; } }
    return true;
}

}  // namespace strata::core
