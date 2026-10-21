# DeepSeek V4 Flash (`deepseek4`) forward-pass spec for the Strata port

Source: llama.cpp (commit bed0a8566), read 2026-10-11.
Key files:
- `src/models/deepseek4.cpp` (loader, graph, MTP graph)
- `src/llama-kv-cache-dsv4.{h,cpp}` (dedicated memory class)
- `src/llama-graph.cpp`: DSV4 inputs/masks 747-1056 and 3493-3515, `build_moe_ffn` 2015-2389, `build_ffn` clamp 1849-1870, `build_attn_mha` 2624-2760
- `src/llama-kv-cache.cpp`: Hadamard 20-50, `attn_rot_k` 321-332
- `ggml/include/ggml.h`: `ggml_rope_set_offset` 2057, `ggml_lightning_indexer` 2673, `ggml_dsv4_hc_comb/pre/post` 2690-2727, `ggml_swiglu_clamp` 1382
- `ggml/src/ggml-cpu/ops.cpp`: swiglu_clamp 3571, softmax+sinks 5854-5908, rope/YaRN 6179-6460, hc_comb 11336-11470, lightning indexer 12352-12430
- CUDA reference kernels: `ggml/src/ggml-cuda/dsv4-hc.cu`, `lightning-indexer.cu`
- `src/models/dflash.cpp`, `common/speculative.cpp` 981-1340 (DSpark)
- `conversion/deepseek.py` 525-1035 (HF -> GGUF)

## 1. Hyperparameters (prefix `deepseek4.`)

| key | value | note |
| --- | --- | --- |
| block_count | 43 | includes MTP layers if present |
| nextn_predict_layers | 0/1 | loader probes `blk.{n_main}.nextn.eh_proj.weight` |
| embedding_length | 4096 | |
| attention.head_count / head_count_kv | 64 / 1 | single shared latent K=V |
| attention.key_length = value_length | 512 | |
| rope.dimension_count | 64 | applied to the LAST 64 dims [448,512), NORM (adjacent pairs) |
| rope.freq_base | 10000 | only ratio-0 layers |
| rope.scaling | yarn, factor 16, orig ctx 65536, beta 32/1 | |
| layer_norm_rms_epsilon | 1e-6 | every RMSNorm incl. weightless |
| expert_count / used | 256 / 6 | no group routing |
| expert_gating_func | 4 = SQRT_SOFTPLUS | |
| q_lora_rank | 1024 | |
| sliding_window | 128 | raw-token window |
| expert_feed_forward_length | 2048 | |
| expert_shared_count | 1 | |
| expert_weights_scale / norm | 1.5 / true | |
| swiglu_clamp_exp / shexp | 10.0 per layer | |
| indexer head_count / key_length / top_k | 64 / 128 / 512 | |
| output_group_count / output_lora_rank | 8 / 1024 | |
| compress_rope_freq_base | 160000 | |
| compress_ratios | [0,0,4,128,4,128,...,4,128,4,0,0,0] | L0-1: SWA only; even 2..42: CSA(4); odd 3..41: HCA(128); 43-45 MTP |
| hyper_connection count / sinkhorn iters / eps | 4 / 20 / 1e-6 | |
| hash_layer_count | 3 | layers 0-2 route by token id |

All layers are MoE + 1 shared expert (no dense FFN layers). rope_type NORM.
Tokenizer: gpt2 BPE, pre `joyai-llm`, vocab 129280, bos 0, eos 1, pad 2.

## 2. Tensors (ggml order {in, out})

Global: `token_embd` {4096,129280}, `output_norm` {4096}, `output` {4096,129280},
`output_hc_fn` {16384,4}, `output_hc_base` {4}, `output_hc_scale` {1}.

Per layer `blk.N.`:
- HC: `hc_attn_fn` {16384,24}, `hc_attn_base` {24}, `hc_attn_scale` {3}; same for `hc_ffn_*`
- `attn_norm` {4096}, `ffn_norm` {4096}
- `attn_sinks` {64}
- `attn_q_a` {4096,1024}, `attn_q_a_norm` {1024}, `attn_q_b` {1024,32768}
- `attn_kv` {4096,512}, `attn_kv_a_norm` {512}
- `attn_output_a` file {4096,8192} -> reshaped {4096,1024,8} (group g = output rows g*1024..), `attn_output_b` {8192,4096}
- ratio != 0 (coff = 2 if ratio 4 else 1): `attn_compressor_kv` {4096,coff*512}, `attn_compressor_gate` {4096,coff*512}, `attn_compressor_ape` {coff*512,ratio}, `attn_compressor_norm` {512}
- ratio == 4: `indexer.proj` {4096,64}, `indexer.attn_q_b` {1024,8192}, `indexer_compressor_kv` {4096,256}, `indexer_compressor_gate` {4096,256}, `indexer_compressor_ape` {256,4}, `indexer_compressor_norm` {128}
- MoE: `ffn_gate_inp` {4096,256}; layers <3: `ffn_gate_tid2eid` I32 {6,129280} (no bias); else `exp_probs_b.bias` {256}; optional `exp_probs_b_vl.bias`
- `ffn_{gate,up}_exps` {4096,2048,256}, `ffn_down_exps` {2048,4096,256}
- `ffn_{gate,up}_shexp` {4096,2048}, `ffn_down_shexp` {2048,4096}
- MTP block: `nextn.eh_proj` {8192,4096}, `nextn.enorm`, `nextn.hnorm` {4096}

## 3. Forward pass

`X` = residual [4096, 4 streams, T]; `rms(v) = v/sqrt(mean(v^2)+1e-6)`; `RMS(v,w) = rms(v)*w`.

### 3.0 Input
`e = token_embd[tok]` (no scale); `X[:,h,t] = e` for all 4 streams.

### 3.1 HC-pre (deepseek4.cpp 353-409; unfused 289-351)
```
flat = reshape(X,[16384,T])              # flat[h*4096+i]
m    = fn^T @ rms(flat)                  # [24,T], no norm weight
pre  = sigmoid(m[0:4]*scale[0] + base[0:4]) + eps
post = 2*sigmoid(m[4:8]*scale[1] + base[4:8])
L[d,s] = m[8+d+4*s]*scale[2] + base[8+d+4*s]       # d = dst (fastest), s = src
C[:,s] = softmax_d(L[:,s]) + eps
C[d,:] /= (eps + sum_s C[d,s])
repeat 19x: C[:,s] /= (eps + sum_d C[d,s]); C[d,:] /= (eps + sum_s C[d,s])
x = sum_h pre[h]*X[:,h]
```
### 3.2 HC-post
`X'[:,d] = post[d]*f + sum_s C[d,s]*X[:,s]`

### 3.3 Layer
```
r=X; x=RMS(HCpre_attn(X), attn_norm); a=Attn(x); X=HCpost(a,r)
r=X; x=RMS(HCpre_ffn(X),  ffn_norm);  f=MoE(x)+Shared(x); X=HCpost(f,r)
```
### 3.4 Head
```
m = output_hc_fn^T @ rms(flat(X))   # [4]
pre = sigmoid(m*scale[0] + base) + eps
h = sum pre[h] X[:,h]
logits = output @ RMS(h, output_norm)
```

### 3.5 Attention
RoPE params:
- ratio != 0: base 160000, freq_scale 1/16, ext_factor 1, n_ctx_orig 65536, beta 32/1, attn_factor = 1/(1+0.1 ln 16) (cancels ggml YaRN mscale -> net magnitude 1)
- ratio 0: base 10000, plain rope (fs 1, ext 0)

ggml YaRN: theta_scale = base^(-2/64); pair i0: th_extrap = p*theta_scale^(i0/2), th_interp = fs*th_extrap;
ramp = 1 - clamp((i0/2 - lo)/max(0.001, hi-lo), 0, 1); th = th_interp*(1-ramp*ext) + th_extrap*ramp*ext;
lo = max(0, floor(cd(32))), hi = min(63, ceil(cd(1))), cd(b) = 64*ln(n_ctx_orig/(b*2pi))/(2 ln base).
NORM: rotate (x[o+2k], x[o+2k+1]), o = 448.

```
qr = RMS(wq_a @ x, attn_q_a_norm)                 # [1024]
q  = reshape(wq_b @ qr,[512,64]); q = rms(q) per head (NO weight)
q  = rope(q,pos) dims [448,512)
kv = RMS(wkv @ x, attn_kv_a_norm)                 # [512], K = V
kv = rope(kv,pos) dims [448,512)
<compressor updates 3.6>
o  = attn(keys = raw window U compressed, V=K, scale 1/sqrt(512), sinks)
o  = rope_back(o, pos_q) dims [448,512)           # inverse rotation
o  = reshape(o,[4096, 8 groups])                  # group g = heads 8g..8g+7
oa[g] = wo_a[:,:,g]^T @ o[g]                      # 4096 -> 1024
out = wo_b @ concat_g(oa)                         # 8192 -> 4096
```
Sinks: per head, raw sink logit s_h joins the max and adds exp(s_h - max) to the denominator, no value.
Hadamard k_rot only when KV is quantized (orthonormal, can be skipped consistently).

Variants:
- (a) ratio 0 (also MTP, DSpark): raw only, key visible iff p_q - p_k < 128 (causal)
- (b) HCA 128: raw window + compressed rows; row j visible iff j < (p+1)/128; no top-k
- (c) CSA 4: raw window + compressed rows restricted to indexer top-k AND j < (p+1)/4

Lightning indexer:
```
iq = reshape(indexer.attn_q_b @ qr,[128,64])      # qr = normed q_a output
iq = rope(iq,pos) dims [64,128)                    # compress-layer YaRN, base 160000
iq = H128 @ iq
w  = (indexer.proj @ x) / sqrt(128*64)             # [64], x = attn_norm output
score[j] = sum_h relu(iq_h . K_lid[j]) * w_h + mask[j]
topk = top_k(score, min(512, n_kv_graph))          # n_kv_graph padded to mult of 256, min 256
```

### 3.6 Compressors
Per token: `skv = Wkv@x`, `ssc = Wgate@x + ape[:, pos % ratio]`. Widths CSA 1024, HCA 512, LID 256.
F32 ring: state_size = 2*ratio (8) for CSA/LID, 128 for HCA; row pos % state_size.
On block completion ((pos+1) % ratio == 0), b = pos/ratio, s0 = b*ratio:
- HCA: V,S over tokens [s0, s0+127]; c[d] = sum_t softmax_t(S[t,d]) * V[t,d] (per channel)
- CSA/LID: prev window [s0-4,s0-1] uses FIRST half of skv/ssc, current [s0,s0+3] uses SECOND half; 8 rows, same per-channel softmax sum; positions < 0 -> kv 0, score -inf
- then c = RMS(c, compressor_norm); c = rope(c, pos=s0) last 64 dims (compress YaRN); LID only: H128; write row b
CSA layers run both attn compressor (kv_csa) and indexer compressor (kv_lid); HCA writes kv_hca.

## 4. MoE
```
logits = ffn_gate_inp @ x               # F32
p = sqrt(softplus(logits))              # [256]
il < 3 (text): sel = tid2eid[token]     # no bias
else:          sel = top6(p + exp_probs_b)
w = p[sel]; w /= max(sum w, 6.1035e-5); w *= 1.5
expert(x) = down( silu(min(g,10)) * clamp(u,-10,10) )
f = sum_k w_k expert_sel_k(x) + shared(x)   # shared: same clamp
```

## 5. Prefill vs decode
Same math. Prefill: blocks completed in the ubatch are compressed before attention, later tokens see them.
Decode: compressed row only on boundary tokens. Top-k = min(512, padded n_kv).
Spec rollback: ring state keeps n_rs_seq snapshot planes.

## 6. KV / memory
K-only (K = V).
- raw: per layer K[512] per cell, size pad256(min(ctx, 128 + n_ubatch))
- kv_csa: pad256(ceil(ctx/4)) x 512 (CSA layers)
- kv_hca: pad256(ceil(ctx/128)) x 512 (HCA layers)
- kv_lid: pad256(ceil(ctx/4)) x 128 (CSA layers, Hadamard-rotated)
- F32 rings: csa [1024,8], hca [512,128], lid [256,8] per layer

## 7. MTP
`e = RMS(embed(tok), enorm)`; `h = RMS(h_prev, hnorm)`; `X = eh_proj @ concat(e,h)` per stream; one DSV4 block (SWA only) + hc_head + head.

## 8. DSpark (arch `dflash`, hc > 0)
N DSV4 stages (SWA only). Target features = hc_mean of inputs of target_layers -> `g = RMS(fc @ feat, enc_norm)` -> per-stage K injected.
Draft block [id_last, MASK x (B-1)] non-causal; markov head `logits_i += w2 @ w1[prev]`, greedy chain; confidence head `sigmoid(proj . [hidden; w1[prev]] + b)`.

As implemented in Abyss (v1.3.43, `src/core/dspark.cu`; bleysg's `deepseek4-dspark` GGUF, ds4's layout: 3 stages,
target layers 40/41/42, block 5, Markov rank 256, the target's 256 experts):
- **Context:** `g[p] = rms(main_proj [h40; h41; h42]) * main_norm`, h_L = the mean of layer L's 4 output streams at
  position p. Each stage keeps `K = V = rope_p(rms(attn_kv g[p]) * attn_kv_a_norm)` in a ring of 256 positions (FP16,
  like the target's window keys); a draft reads the last 128. The token path, the verify window's kept columns (at
  `spec_rewind`) and a prompt chunk's last 128 rows leave their layer 40-42 outputs; `dspark_commit` turns them into
  ring rows (the prompt path's last-layer tail skip is off while the drafter is loaded).
- **Block:** `[y, noise x 4]` (y = the pending token, `noise_token_id`'s embedding after it) at positions pos .. pos + 4
  through the stages: the mHC read / write halves, attention over the context and the whole block (no mask; the
  target's attend kernel with the rows as compressed entries), the MoE (sqrt-softplus router + bias, top 6 of 256,
  weights x 1.5, the shared expert, the SwiGLU clamped at 10), then `hc_head`, the norm and the target's output head
  (row i proposes position pos + i + 1).
- **Chain:** draft i = argmax(logits_i + markov_w2 . markov_w1[draft i-1]) with draft -1 = y; its confidence
  sigmoid(conf_proj . [h_i; markov_w1[draft i-1]]) with h_i before the norm. The decode keeps the leading drafts at
  confidence >= `STRATA_DS4_DSPARK_THR` (0.6) and verifies them in one window; the target's samples decide.
- The same numbers as the numpy reference (Abyss-tools `bench/dspark/dspark_probe.py`): 4.68 against 4.705 tokens a
  step on a swe-bench answer.

## 9. Quant types
Unsloth UD-Q2_K_XL: to be read from shards 2-3 headers. Other files observed:
- IQ1_M: experts IQ1_M/IQ2_XXS gate-up, IQ3_XXS/MXFP4 down, shexp down MXFP4, attn Q8_0, q_a Q5_K, embd/output Q4_K, gate_inp BF16, hc/norms F32
- Q2-0731: gate/up IQ2_XXS, down Q2_K, attn/shexp/output Q8_0, compressor/indexer/gate_inp/hc_fn/embd F16
Port must cover F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K, IQ1_M, IQ2_XXS, IQ3_XXS, MXFP4, I32.

## 10. Gotchas
- RoPE tail = last 64 of 512 (attn), last 64 of 128 (indexer); interleaved pairs
- de-rope attention output (V = K carries RoPE)
- weightless per-head q RMSNorm
- per-channel compressor softmax over tokens
- compressed K rotated at block start s0
- sinkhorn index dst + 4*src, softmax over dst, eps after softmax and in every denominator, 1 col-norm then 19 (row,col) pairs
- HC: pre +eps, post = 2*sigmoid without eps
- MoE weights from unbiased sqrt(softplus), normalized, x1.5; hash layers 0-2 use tid2eid[token]
