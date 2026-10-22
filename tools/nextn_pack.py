#!/usr/bin/env python3
"""nextn_pack.py - the NextN (MTP) block of a split GGUF into one small GGUF the engine can draft with.

    .venv/bin/python tools/nextn_pack.py --gguf <first shard> --out <nextn.gguf> [--experts q4_k] [--ggml-lib <libggml-base.so>]

The block's tensors (blk.<block_count - nextn_predict_layers>.*) are copied as stored, except the routed experts, which
are requantized to a type the engine's expert kernels serve (the default Q4_K; q5_k, q6_k; the block's own are often
Q8_0, as in unsloth's Ornith-1.5-122B GGUFs, and Q8_0 experts have no decode kernel yet) through ggml's quantizer
(ggml_quantize_chunk of a libggml-base shared library: llama.cpp's build, or --ggml-lib). The model's general.* and
<arch>.* keys come along so the file reads as the same model. Load it with STRATA_GLM_MTP_GGUF=<nextn.gguf> (the
engine looks the block up there first; the trunk, the head and the embedding stay in the model's shards).
"""
import ctypes
import glob
import argparse
import os
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the model's first shard (or its only file)")
    ap.add_argument("--out", required=True, help="the NextN GGUF to write")
    ap.add_argument("--experts", default="q4_k", help="the routed experts' type: q4_k (default), q5_k, q6_k, or q8_0 (as stored)")
    ap.add_argument("--ggml-lib", default="", help="libggml-base.so (.dylib / ggml-base.dll) with ggml_quantize_chunk")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.environ.get("STRATA_GGUF_PY", os.path.join(here, "..", "third_party", "llama.cpp", "gguf-py")))
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
    from gguf.quants import dequantize, quantize

    want = {"q4_k": GGMLQuantizationType.Q4_K, "q5_k": GGMLQuantizationType.Q5_K, "q6_k": GGMLQuantizationType.Q6_K,
            "q8_0": GGMLQuantizationType.Q8_0}.get(a.experts.lower())
    if want is None:
        print("--experts: q4_k, q5_k, q6_k or q8_0")
        return 2
    # ggml's quantizer (gguf-py quantizes Q8_0 alone)
    lib = None
    if want != GGMLQuantizationType.Q8_0:
        cands = [a.ggml_lib] if a.ggml_lib else []
        for d in (os.path.join(here, "..", "build"), os.path.join(here, "..", "build-hip"),
                  os.path.join(here, "..", "third_party", "llama.cpp", "build", "bin"),
                  os.path.join(here, "..", "..", "Abyss-tools", "llama-build", "bin")):
            cands += glob.glob(os.path.join(d, "libggml-base.so")) + glob.glob(os.path.join(d, "libggml-base.dylib")) + \
                     glob.glob(os.path.join(d, "ggml-base.dll"))
        for c in cands:
            try:
                lib = ctypes.CDLL(c)
                lib.ggml_quantize_chunk
                break
            except (OSError, AttributeError):
                lib = None
        if lib is None:
            print("no libggml-base with ggml_quantize_chunk found (build llama.cpp, or pass --ggml-lib); --experts q8_0 copies")
            return 1
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_void_p]
        lib.ggml_row_size.restype = ctypes.c_size_t
        lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    first = GGUFReader(a.gguf)
    fields = first.fields
    arch = str(bytes(fields["general.architecture"].parts[fields["general.architecture"].data[0]]), "utf-8")
    def num(key, default):
        f = fields.get(key)
        return int(f.parts[f.data[0]][0]) if f is not None else default
    block_count = num(arch + ".block_count", 0)
    nextn = num(arch + ".nextn_predict_layers", 0)
    if block_count <= 0 or nextn <= 0:
        print("%s: block_count %d, nextn_predict_layers %d - no NextN block" % (arch, block_count, nextn))
        return 1
    L = block_count - nextn
    P = "blk.%d." % L
    # the shards: split.count files named ...-0000i-of-0000n.gguf
    paths = [a.gguf]
    n_split = num("split.count", 1)
    if n_split > 1:
        base = os.path.basename(a.gguf)
        i = base.rfind("-00001-of-")
        if i < 0:
            print("the shard name does not carry -00001-of-")
            return 1
        for k in range(2, n_split + 1):
            paths.append(os.path.join(os.path.dirname(a.gguf), base[:i] + "-%05d-of-%05d.gguf" % (k, n_split)))
    readers = [first] + [GGUFReader(p) for p in paths[1:]]
    tensors = []
    for r in readers:
        for t in r.tensors:
            if t.name.startswith(P):
                tensors.append(t)
    if not any(t.name == P + "nextn.eh_proj.weight" for t in tensors):
        print("%s: no %snextn.eh_proj.weight in the shards" % (arch, P))
        return 1
    w = GGUFWriter(a.out, arch)
    # the model's keys (not the tokenizer's, not the split's)
    for key, f in fields.items():
        if key == "general.architecture" or key.startswith("tokenizer.") or key.startswith("split.") or key.startswith("GGUF."):
            continue
        if not (key.startswith("general.") or key.startswith(arch + ".")):
            continue
        vt = f.types[0]
        if len(f.types) == 1:
            val = f.parts[f.data[0]]
            if vt == gguf.GGUFValueType.STRING:
                w.add_string(key, str(bytes(val), "utf-8"))
            elif vt in (gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32, gguf.GGUFValueType.UINT64, gguf.GGUFValueType.INT64,
                        gguf.GGUFValueType.UINT8, gguf.GGUFValueType.INT8, gguf.GGUFValueType.UINT16, gguf.GGUFValueType.INT16):
                w.add_uint32(key, int(val[0])) if vt == gguf.GGUFValueType.UINT32 else w.add_int32(key, int(val[0])) if vt == gguf.GGUFValueType.INT32 else w.add_uint64(key, int(val[0]))
            elif vt == gguf.GGUFValueType.FLOAT32:
                w.add_float32(key, float(val[0]))
            elif vt == gguf.GGUFValueType.BOOL:
                w.add_bool(key, bool(val[0]))
        elif vt == gguf.GGUFValueType.ARRAY:
            et = f.types[1]
            if et == gguf.GGUFValueType.STRING:
                w.add_array(key, [str(bytes(f.parts[i]), "utf-8") for i in f.data])
            elif et in (gguf.GGUFValueType.BOOL,):
                w.add_array(key, [bool(f.parts[i][0]) for i in f.data])
            elif et == gguf.GGUFValueType.FLOAT32:
                w.add_array(key, [float(f.parts[i][0]) for i in f.data])
            else:
                w.add_array(key, [int(f.parts[i][0]) for i in f.data])
    total = 0
    for t in tensors:
        data = np.asarray(t.data)
        qt = t.tensor_type
        if t.name.endswith("_exps.weight") and qt != want:
            t0 = time.time()
            f32 = np.ascontiguousarray(dequantize(data, qt), dtype=np.float32)
            n_per_row = int(f32.shape[-1])
            nrows = int(f32.size // n_per_row)
            row_bytes = int(lib.ggml_row_size(int(want), n_per_row))
            out = np.empty(f32.shape[:-1] + (row_bytes,), dtype=np.uint8)
            wrote = lib.ggml_quantize_chunk(int(want), f32.ctypes.data, out.ctypes.data, 0, nrows, n_per_row, None)
            if wrote != out.nbytes:
                print("%s: ggml_quantize_chunk wrote %d bytes, expected %d" % (t.name, wrote, out.nbytes))
                return 1
            data = out
            del f32
            print("%s: %s -> %s, %.1f MB, %.0f s" % (t.name, qt.name, want.name, data.nbytes / 1e6, time.time() - t0), flush=True)
            qt = want
        else:
            print("%s: %s as stored, %.1f MB" % (t.name, qt.name, data.nbytes / 1e6), flush=True)
        w.add_tensor(t.name, data, raw_dtype=qt)
        total += data.nbytes
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print("%s: %d tensors of blk.%d, %.2f GB -> %s" % (arch, len(tensors), L, total / 1e9, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
