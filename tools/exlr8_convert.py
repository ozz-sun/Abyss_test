"""tools/exlr8_convert.py - an exlr8 v2 checkpoint (kindlingai's GLM-5.3-Flash K2/K3/K4) as an EXL3 model directory
that tools/exl3_pack.py packs.

    python tools/exlr8_convert.py --model ~/models/glm-5.3-flash-exlr8 --width 2 --out ~/models/glm-5.3-flash-exlr8-k2
    python tools/exl3_pack.py --model ~/models/glm-5.3-flash-exlr8-k2

exlr8 (https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4, its card's Loader section, and the reference
loader in https://github.com/kindlingai/exlr8) is exllamav3's trellis with three differences the engine cares about:

  * the codebook: mcg, every value then rounded to FP8 E4M3 (nearest even, saturating).  The rounding is part of the
    format; the engine's codebook 3 (include/strata/kernels/exl3.hpp) does it, and this tool marks every trellis
    matrix with a `mcg_e4m3` tensor so that exl3_pack.py writes cb 3.
  * the experts are stored in 16 halves of 128 intermediate channels, each half's tiles in its own order, with the
    hidden side's scales as the layer's signs times 32 per-expert block scales.  The halves are put back into one
    exllamav3 trellis [k/16][n/16][16 K] per matrix, and suh / svh expanded to their k / n FP16 values (exact: a sign
    times an FP16 value).
  * dense matrices come in seven formats.  The trellis ones are reassembled the same way; the FP8, MXFP8 and FP8
    block / row ones are decoded to BF16 `.weight` tensors (E4M3 times a power of two or an F32 scale - BF16 keeps
    E4M3's 3 mantissa bits; the F32 scales round), and BF16 / F32 are copied.  exl3_pack.py serves those as BF16 rows.

The NextN block's eh_proj is MXFP8, so exl3_pack.py packs the model without MTP.  The vision tower is left out.
Writes about the width's expert bytes (K2 78 GB, K3 117 GB) plus ~8 GB of dense BF16.

--check-expert L,E: decode expert E of layer L both ways - the converted trellis through exl3_pack.py's numpy
EXL3 decode (cb 3), and the reference loader's half-by-half reconstruction (--ref <its exlr8.py>, needs torch) - and
print their largest difference (a layout mistake shows as O(1), the right layout as float rounding).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PREFIX = "model.language_model."
PROJS = ("gate", "up", "down")
TRELLIS, INT4, MXFP8, FP8_BLOCK, FP8_ROW, BF16, FP32 = 1, 2, 3, 4, 5, 6, 7
SIDE_IN = 2
DATA = {TRELLIS: ("trellis", "suh", "svh"), INT4: ("q4", "s4"), MXFP8: ("fp8", "e8m0"), FP8_BLOCK: ("fp8", "scale"),
        FP8_ROW: ("fp8", "scale"), BF16: ("bf16",), FP32: ("fp32",)}
NP = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "I64": "<i8", "U64": "<u8", "I8": "i1", "U8": "u1", "I16": "<i2"}


def e4m3_table() -> np.ndarray:
    """The 256 OCP E4M3 values (0x7f / 0xff NaN) as float32."""
    b = np.arange(256)
    e, m = (b >> 3) & 15, (b & 7).astype(np.float64)
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
    v = np.where((b & 0x7F) == 0x7F, np.nan, v)
    return np.where(b & 0x80, -v, v).astype(np.float32)


E4M3 = e4m3_table()


def to_bf16(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class Safe:
    """A safetensors file read in place."""

    def __init__(self, p: pathlib.Path):
        self.path = p
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.meta = self.h.pop("__metadata__", None) or {}
        self.base = 8 + n
        self.mm = np.memmap(p, dtype=np.uint8, mode="r")

    def names(self):
        return [k for k in self.h if not k.startswith("__pad.")]

    def raw(self, name: str) -> np.ndarray:
        a, b = self.h[name]["data_offsets"]
        return self.mm[self.base + a:self.base + b]

    def get(self, name: str) -> np.ndarray:
        v = self.h[name]
        return self.raw(name).view(NP[v["dtype"]]).reshape(v["shape"])


class Writer:
    """A safetensors file written in two passes: tensors are declared (name, dtype, shape, producer), the header
    goes out, then each producer's bytes in order - a layer's 600 MB expert matrices never sit in memory at once."""

    def __init__(self):
        self.items: list[tuple] = []

    def add(self, name: str, dtype: str, shape, fn):
        self.items.append((name, dtype, [int(x) for x in shape], fn))

    def write(self, p: pathlib.Path) -> list[str]:
        size = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "U8": 1, "I64": 8}
        h, off = {}, 0
        for name, dt, shape, _ in self.items:
            n = size[dt] * int(np.prod(shape))
            h[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
            off += n
        raw = json.dumps(h, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        tmp = p.with_suffix(".tmp")
        with tmp.open("wb") as f:
            f.write(struct.pack("<Q", len(raw)) + raw)
            for name, dt, shape, fn in self.items:
                b = np.ascontiguousarray(fn())
                want = h[name]["data_offsets"][1] - h[name]["data_offsets"][0]
                if b.nbytes != want:
                    raise SystemExit("%s: %d bytes, declared %d" % (name, b.nbytes, want))
                f.write(memoryview(b.reshape(-1).view(np.uint8)))
        tmp.replace(p)
        return [n for n, *_ in self.items]


# ------------------------------------------------------------------ tile reassembly
def tiles_out_major(raw: np.ndarray, K: int, rows: int) -> np.ndarray:
    """A run of tiles stored [k16][c = 0..7] (a gate/up half, a dense entry on the output side) -> [k16][8][16 K]."""
    return raw.reshape(rows // 16, 8, 16 * K * 2)


def tiles_in_major(raw: np.ndarray, K: int, cols: int) -> np.ndarray:
    """A run of tiles stored [b][r = 0..7][c = 0..7] (a down half, a dense entry on the input side) -> the half's
    8 tile rows [r][cols / 16][16 K] (tile column 8 b + c)."""
    return raw.reshape(cols // 128, 8, 8, 16 * K * 2).transpose(1, 0, 2, 3).reshape(8, cols // 16, 16 * K * 2)


def trellis_from_halves(halves, K: int, k: int, n: int, side_in: bool) -> np.ndarray:
    """The exllamav3 trellis bytes [k/16][n/16][32 K] of a matrix stored as 128-channel entries: on the output side
    (entry g = columns 128 g..) or the input side (entry g = rows 128 g..)."""
    if side_in:
        return np.concatenate([tiles_in_major(h, K, n) for h in halves], axis=0)
    return np.concatenate([tiles_out_major(h, K, k) for h in halves], axis=1)


# ------------------------------------------------------------------ dense
def decode_dense(rec, t: dict) -> np.ndarray:
    """A non-trellis dense matrix as float32 (out, in), the HF weight."""
    fmt, out, inp = int(rec[0]), int(rec[1]), int(rec[2])
    if fmt == MXFP8:
        e = t["e8m0"].astype(np.int32) - 127
        return E4M3[t["fp8"]] * np.ldexp(np.float32(1), e).astype(np.float32).repeat(32, axis=1)
    if fmt == FP8_BLOCK:
        s = t["scale"].repeat(128, axis=0)[:out].repeat(128, axis=1)[:, :inp]
        return E4M3[t["fp8"]] * s
    if fmt == FP8_ROW:
        return E4M3[t["fp8"]] * t["scale"][:, None]
    if fmt == BF16:
        return (t["bf16"].astype(np.uint32) << 16).view(np.float32)
    if fmt == FP32:
        return t["fp32"].astype(np.float32)
    raise SystemExit("dense format %d (int4) is not handled yet" % fmt)


def add_dense_file(w: Writer, f: Safe, pre: str, skip=lambda n: False):
    names = f.names()
    mats = [n[:-len(".record")] for n in names if n.endswith(".record")]
    owned = set()
    for m in mats:
        rec = f.get(m + ".record")
        fmt = int(rec[0])
        owned.update("%s.%s" % (m, s) for s in ("record", "checksum") + DATA[fmt])
        if skip(m):
            continue
        if fmt == TRELLIS:
            out, inp, K, side = int(rec[1]), int(rec[2]), int(rec[3]), int(rec[4])
            ent = f.get(m + ".trellis")
            w.add(pre + m + ".trellis", "I16", [inp // 16, out // 16, 16 * K],
                  lambda ent=ent, K=K, inp=inp, out=out, side=side:
                  trellis_from_halves([ent[g] for g in range(ent.shape[0])], K, inp, out, side == SIDE_IN))
            w.add(pre + m + ".suh", "F16", [inp], lambda m=m: f.get(m + ".suh"))
            w.add(pre + m + ".svh", "F16", [out], lambda m=m: f.get(m + ".svh"))
            w.add(pre + m + ".mcg_e4m3", "U8", [1], lambda: np.ones(1, np.uint8))
        else:
            out, inp = int(rec[1]), int(rec[2])
            t = {s: f.get("%s.%s" % (m, s)) for s in DATA[fmt]}
            w.add(pre + m + ".weight", "BF16", [out, inp], lambda rec=rec, t=t: to_bf16(decode_dense(rec, t)))
    for n in names:
        if n in owned or skip(n):
            continue
        v = f.h[n]
        w.add(pre + n, v["dtype"], v["shape"], lambda n=n: f.raw(n))


# ------------------------------------------------------------------ experts
class Experts:
    def __init__(self, root: pathlib.Path, L: int, K: int):
        base = root / "experts" / ("exlr8-layer-%03d" % L)
        self.ix = Safe(base.with_name(base.name + ".index.safetensors"))
        self.tr = Safe(base.with_name(base.name + ".k%d.safetensors" % K))
        self.K = K
        self.H = int(self.ix.meta["hidden_size"])
        self.F = int(self.ix.meta["moe_intermediate_size"])
        self.n = int(self.ix.meta["routed_experts"])
        self.sin = self.ix.get("signs.in").astype(np.float32)
        self.sout = self.ix.get("signs.out").astype(np.float32)

    def _k(self, shared: bool) -> str:
        return ("shared.k%d." if shared else "k%d.") % self.K

    def trellis(self, e: int, p: int, shared: bool = False) -> np.ndarray:
        k = self._k(shared)
        idx = self.ix.get(k + "index")[0 if shared else e, p]
        halves = [np.asarray(self.tr.mm[int(o):int(o) + int(n)]) for o, n in idx]
        if p == 2:   # down: (F, H), halves of input rows
            return trellis_from_halves(halves, self.K, self.F, self.H, True)
        return trellis_from_halves(halves, self.K, self.H, self.F, False)

    def side(self, e: int, p: int, shared: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(suh, svh) as FP16 of expert e's projection p."""
        k, j = self._k(shared), 0 if shared else e
        if p == 2:
            suh = self.ix.get(k + "down.suh")[j]
            svh = self.sout * self.ix.get(k + "down.sv_block")[j].astype(np.float32).repeat(128)
            return suh.astype(np.float16), svh.astype(np.float16)
        nm = PROJS[p]
        suh = self.sin * self.ix.get(k + nm + ".su_block")[j].astype(np.float32).repeat(128)
        return suh.astype(np.float16), self.ix.get(k + nm + ".svh")[j].astype(np.float16)

    def add(self, w: Writer, pre: str):
        for e in list(range(self.n)) + ["shared"]:
            shared = e == "shared"
            for p, nm in enumerate(PROJS):
                q = pre + ("mlp.shared_experts." if shared else "mlp.experts.%d." % e) + nm + "_proj."
                k, n = (self.F, self.H) if p == 2 else (self.H, self.F)
                w.add(q + "trellis", "I16", [k // 16, n // 16, 16 * self.K],
                      lambda e=e, p=p, s=shared: self.trellis(0 if s else e, p, s))
                w.add(q + "suh", "F16", [k], lambda e=e, p=p, s=shared: self.side(0 if s else e, p, s)[0])
                w.add(q + "svh", "F16", [n], lambda e=e, p=p, s=shared: self.side(0 if s else e, p, s)[1])
                w.add(q + "mcg_e4m3", "U8", [1], lambda: np.ones(1, np.uint8))


# ------------------------------------------------------------------ --check-expert
def check_expert(root: pathlib.Path, K: int, L: int, e: int, ref: pathlib.Path) -> int:
    sys.path.insert(0, str(HERE))
    import exl3_pack
    sys.path.insert(0, str(ref.parent))
    import torch
    import exlr8 as R
    R.HIDDEN = None
    x = Experts(root, L, K)
    R.HIDDEN = x.H
    rl = R.ExpertLayer(str(root), L)
    worst = 0.0
    for p, nm in enumerate(PROJS):
        tr = x.trellis(e, p).view(np.int16)
        suh, svh = x.side(e, p)
        mine = exl3_pack.exl3_rows(tr, suh, svh, K, 3)               # (n, k): the HF weight
        want = rl.projection(e, nm, K).numpy()                       # (out, in)
        d = float(np.max(np.abs(mine - want)))
        scale = float(np.max(np.abs(want)))
        print("layer %d expert %d %s: max |diff| %.3g (max |w| %.3g)" % (L, e, nm, d, scale))
        worst = max(worst, d / max(scale, 1e-30))
    print("worst relative %.3g -> %s" % (worst, "OK" if worst < 1e-2 else "MISMATCH"))
    return 0 if worst < 1e-2 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the exlr8 release directory (experts/, dense/, config.json)")
    ap.add_argument("--width", type=int, required=True, choices=(2, 3, 4), help="the expert width K to convert")
    ap.add_argument("--out", help="the EXL3 directory to write (default: <model>-k<width>)")
    ap.add_argument("--layers", help="only these layers (a,b-c), for a test")
    ap.add_argument("--check-expert", help="L,E: compare one expert against the reference loader (--ref)")
    ap.add_argument("--ref", help="kindlingai/exlr8's loader/exlr8.py")
    a = ap.parse_args()
    root = pathlib.Path(a.model).expanduser().absolute()
    if a.check_expert:
        if not a.ref:
            print("--check-expert needs --ref <loader/exlr8.py>")
            return 1
        L, e = (int(v) for v in a.check_expert.split(","))
        return check_expert(root, a.width, L, e, pathlib.Path(a.ref).expanduser().absolute())
    out = pathlib.Path(a.out).expanduser().absolute() if a.out else root.with_name(root.name + "-k%d" % a.width)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "config.json").read_text())
    t = cfg["text_config"]
    n_layers = t["num_hidden_layers"] + int(t.get("num_nextn_predict_layers", 0) or 0)
    sel = set()
    for part in (a.layers.split(",") if a.layers else []):
        lo, _, hi = part.partition("-")
        sel.update(range(int(lo), int(hi or lo) + 1))
    wmap = {}
    # the vocabulary: embed_tokens / lm_head (FP8 a row) -> BF16, the final norm
    fn = "model-vocab.safetensors"
    if not (out / fn).exists():
        f = Safe(root / "dense" / "exlr8-vocab.safetensors")
        w = Writer()
        names = {"embed_tokens": PREFIX + "embed_tokens", "lm_head": "lm_head", "norm.weight": PREFIX + "norm.weight"}
        tmp = Writer()
        add_dense_file(tmp, f, "")
        for name, dt, shape, prod in tmp.items:
            base, dot, rest = name.partition(".weight")
            nm = names.get(name, names.get(base, base) + dot + rest)
            w.add(nm, dt, shape, prod)
        wmap.update({n: fn for n in w.write(out / fn)})
        print("%s: done" % fn)
    else:
        wmap.update({n: fn for n in Safe(out / fn).names()})
    for L in range(n_layers):
        if sel and L not in sel:
            continue
        fn = "model-%03d.safetensors" % L
        if (out / fn).exists():
            wmap.update({n: fn for n in Safe(out / fn).names()})
            print("%s: kept" % fn)
            continue
        pre = PREFIX + "layers.%d." % L
        w = Writer()
        add_dense_file(w, Safe(root / "dense" / ("exlr8-dense-%03d.safetensors" % L)), pre)
        if (root / "experts" / ("exlr8-layer-%03d.index.safetensors" % L)).exists():
            Experts(root, L, a.width).add(w, pre)
        wmap.update({n: fn for n in w.write(out / fn)})
        print("%s: %.2f GB" % (fn, (out / fn).stat().st_size / 1e9), flush=True)
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}, indent=1))
    cfg["quantization_config"] = {"quant_method": "exl3", "version": "exlr8-v2", "bits": a.width,
                                  "head_bits": "bf16", "codebook": "mcg_e4m3"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
              "special_tokens_map.json", "LICENSE", "README.md"):
        if (root / n).exists():
            shutil.copy2(root / n, out / n)
    print("%s: %d tensors" % (out, len(wmap)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
