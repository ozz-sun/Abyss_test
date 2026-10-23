"""A refusal-direction edit (.npz) as the engine's STRATA_GLM_ABL file - the edit applied at run time, no overlay.

usage: abl_export.py <edit.npz> <out.abl>
  <edit.npz> as mrfakename/DeepSeek-V4.1-Flash-1017-abliterated ships it (abliteration-edit.npz): Q [n_embd, r]
  orthonormal, scales [layer, (attn_output_b, ffn_down_shexp)].  The engine subtracts s Q Q^T x from each layer's
  attention output and shared-expert output as they are written - the same as W' = W - s Q Q^T W on those weights
  (tools/dense_overlay.py --abl), without re-quantizing them.  Run with STRATA_GLM_ABL=<out.abl>.
File: "ABL1", int32 n_embd, rank, layers, then float32 Q [n_embd][rank] row-major and strengths [layers][2].
"""
import sys

import numpy as np


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    z = np.load(sys.argv[1])
    q = np.ascontiguousarray(z["Q"], dtype=np.float32)
    s = np.ascontiguousarray(z["scales"], dtype=np.float32)
    if q.ndim != 2 or not 1 <= q.shape[1] <= 4 or s.ndim != 2 or s.shape[1] != 2:
        print("unexpected shapes: Q %s, scales %s" % (q.shape, s.shape))
        return 1
    with open(sys.argv[2], "wb") as f:
        f.write(b"ABL1")
        f.write(np.array([q.shape[0], q.shape[1], s.shape[0]], dtype=np.int32).tobytes())
        f.write(q.tobytes())
        f.write(s.tobytes())
    print("%s: n_embd %d, rank %d, %d layers (attention %d, shared expert %d nonzero)"
          % (sys.argv[2], q.shape[0], q.shape[1], s.shape[0], int((s[:, 0] != 0).sum()), int((s[:, 1] != 0).sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
