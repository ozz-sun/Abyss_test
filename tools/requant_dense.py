"""Re-quantizes a few dense tensors of a (split) GGUF and copies everything else byte for byte.

usage: requant_dense.py <in shard 1> <out dir> <type> <regex> [<regex> ...] [--check] [--from=<type>[,<type>...]]
  e.g. requant_dense.py .../UD-Q2_K_XL/DeepSeek-...-00001-of-00003.gguf .../UD-Q2_K_XL-attn-q6k q6_k \
         'blk\\.\\d+\\.attn_q_b\\.weight' 'blk\\.\\d+\\.attn_output_[ab]\\.weight'
Each input shard becomes one output shard of the same name (the split metadata is copied as is).  The matched
tensors are dequantized with gguf-py and quantized by ggml_quantize_chunk (libggml-base.so: no imatrix for the
k-quants); llama-quantize cannot do this on an i-quant model without an imatrix, even for tensors it leaves alone.
GGUF_PY (default: a llama.cpp checkout's gguf-py) and LIBGGML (a llama.cpp build's libggml-base.so) name where those
are; the Abyss .venv's python has numpy.  Examples and the quality / speed of the packs made this way: docs/MANUAL.md,
"A lighter dense pack".
Only Q8_0 tensors are re-quantized unless --from names the source types (e.g. --from=q6_k for GLM-5.3-Flash AJ, whose
dense weights are all Q6_K): a k-quant source is dequantized first, so its rounding error and the new one add up.
"""
import ctypes
import os
import re
import sys
import time

sys.path.insert(0, os.environ.get("GGUF_PY", "/home/zephel01/llm/apps/llama.cpp/gguf-py"))   # llama.cpp/gguf-py
import numpy as np  # noqa: E402
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as Q  # noqa: E402

LIBGGML = os.environ.get("LIBGGML", "/home/zephel01/llm/apps/llama.cpp/build-cuda/bin/libggml-base.so")   # a llama.cpp build


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--check" and not a.startswith("--from=")]
    check = "--check" in sys.argv
    src_types = {Q.Q8_0}
    for a in sys.argv[1:]:
        if a.startswith("--from="):
            src_types = {Q[n.upper()] for n in a[len("--from="):].split(",")}
    if len(args) < 4:
        print(__doc__)
        return 2
    shard1, out_dir, type_name = args[0], args[1], args[2].upper()
    pats = [re.compile(p) for p in args[3:]]
    new_type = Q[type_name]
    lib = ctypes.CDLL(LIBGGML)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    bs, ts = gguf.GGML_QUANT_SIZES[new_type]
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(shard1))
    shards = [shard1] if m is None else [
        os.path.join(os.path.dirname(shard1), "%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(3)))
        for i in range(1, int(m.group(3)) + 1)]
    os.makedirs(out_dir, exist_ok=True)
    total_old = total_new = 0
    arch = None
    for sh in shards:
        r = gguf.GGUFReader(sh)
        if gguf.Keys.General.ARCHITECTURE in r.fields:   # (the later shards of a split carry only the split keys)
            arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
        out = os.path.join(out_dir, os.path.basename(sh))
        print("%s -> %s" % (sh, out), flush=True)
        plan = []
        for t in r.tensors:
            hit = any(p.search(t.name) for p in pats)
            if hit and t.tensor_type not in src_types:
                print("  %s is %s, not %s: left as is" % (t.name, t.tensor_type.name,
                                                         "/".join(sorted(q.name for q in src_types))))
                hit = False
            if hit:
                ne0, ne1 = int(t.shape[0]), int(t.shape[1]) if len(t.shape) > 1 else 1
                if ne0 % bs != 0:
                    print("  %s: ne0 %d not a multiple of %d: left as is" % (t.name, ne0, bs))
                    hit = False
            if hit:
                nbytes = ne1 * (ne0 // bs) * ts
                plan.append((t, ne0, ne1, nbytes))
                total_old += t.n_bytes
                total_new += nbytes
                print("  %-40s %s %9.2f MiB -> %s %9.2f MiB" % (t.name, t.tensor_type.name, t.n_bytes / 2**20,
                                                             type_name, nbytes / 2**20))
            else:
                plan.append((t, 0, 0, 0))
        if check:
            continue
        w = gguf.GGUFWriter(out, arch)
        for field in r.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            if field.name == gguf.Keys.General.ALIGNMENT:   # the writer's padding too, not just the key (nazeshinjite's
                w.add_custom_alignment(int(field.contents()))   # V4.1 GGUF has 16384: copying only the key shifted the data)
                continue
            val_type = field.types[0]
            sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
        for t, ne0, ne1, nbytes in plan:
            if nbytes:
                w.add_tensor_info(t.name, (ne1, nbytes // ne1), np.uint8, nbytes, raw_dtype=new_type)
            else:
                w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        t0 = time.time()
        done = 0
        for t, ne0, ne1, nbytes in plan:
            if nbytes:
                f32 = np.ascontiguousarray(gguf.quants.dequantize(t.data, t.tensor_type).reshape(ne1, ne0), dtype=np.float32)
                dst = np.empty(nbytes, dtype=np.uint8)
                got = lib.ggml_quantize_chunk(int(new_type), f32.ctypes.data, dst.ctypes.data, 0, ne1, ne0, None)
                if got != nbytes:
                    print("ggml_quantize_chunk wrote %d bytes, expected %d for %s" % (got, nbytes, t.name))
                    return 1
                w.write_tensor_data(dst.reshape(ne1, nbytes // ne1), tensor_endianess=r.endianess)
            else:
                w.write_tensor_data(t.data, tensor_endianess=r.endianess)
            done += t.n_bytes
            if done and int(done / 2**30) != int((done - t.n_bytes) / 2**30):
                print("  %.1f GiB written, %.0f s" % (done / 2**30, time.time() - t0), flush=True)
        w.close()
    print("re-quantized %.2f GiB -> %.2f GiB%s" % (total_old / 2**30, total_new / 2**30, " (check only)" if check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
