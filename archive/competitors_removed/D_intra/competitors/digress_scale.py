"""Can one-shot DiGress generate a graph the size of Oldenburg? Measure it.

Runs on a BlueBEAR GPU, in the `digress` env, from D_intra/:

    sbatch competitors/run_digress_scale.sh

One-shot DiGress denoises the whole n x n adjacency at every step, so memory
grows with n^2 (edge features) and attention with n^2 x d. This builds the
denoiser with EXACTLY D_intra's architecture (configs/experiment/cluster_lux.yaml:
6 layers, dx 256, de 64, 8 heads) -- which is DiGress's own -- and runs single
no-grad forward passes, batch 1, at doubling n up to Oldenburg's 6105 nodes,
recording peak GPU memory and the time of one step. Weights are random: memory
and time do not depend on them, so no trained checkpoint is needed.

The extra features DiGress adds ('all': cycle counts from matrix powers, and the
Laplacian eigendecomposition) are left out, so the numbers are a LOWER bound on
what full DiGress needs.

Output: competitors/digress_scale.tsv  -- n, status, peak GiB, s/step, est. h for
500 steps. A row reading OOM at n=6105 is the competitor result.
"""
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from models.transformer_model import GraphTransformer  # noqa: E402

STEPS = 500                       # D_intra's diffusion_steps
SIZES = [256, 512, 1024, 2048, 3072, 4096, 6105]
DIMS = dict(n_layers=6,
            input_dims={'X': 2, 'E': 2, 'y': 1},
            hidden_mlp_dims={'X': 128, 'E': 64, 'y': 128},
            hidden_dims={'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8,
                         'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256},
            output_dims={'X': 2, 'E': 2, 'y': 0})


def main():
    assert torch.cuda.is_available(), "run this on a GPU node"
    dev = torch.device("cuda")
    model = GraphTransformer(act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU(), **DIMS).to(dev).eval()
    total = torch.cuda.get_device_properties(dev).total_memory / 2**30
    print(f"GPU: {torch.cuda.get_device_name(dev)}, {total:.1f} GiB")

    rows = [["n", "status", "peak_GiB", "s_per_step", f"est_h_{STEPS}_steps"]]
    for n in SIZES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        try:
            X = torch.zeros(1, n, 2, device=dev); X[..., 0] = 1
            E = torch.zeros(1, n, n, 2, device=dev); E[..., 0] = 1
            y = torch.zeros(1, 1, device=dev)
            mask = torch.ones(1, n, dtype=torch.bool, device=dev)
            with torch.no_grad():
                model(X, E, y, mask)                     # warm-up
                torch.cuda.synchronize()
                t0 = time.time()
                model(X, E, y, mask)
                torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated(dev) / 2**30
            rows.append([str(n), "ok", f"{peak:.2f}", f"{dt:.3f}", f"{dt * STEPS / 3600:.2f}"])
        except torch.cuda.OutOfMemoryError:
            rows.append([str(n), "OOM", f">{total:.0f}", "", ""])
        except RuntimeError as e:                        # older torch: plain RuntimeError
            if "out of memory" not in str(e):
                raise
            rows.append([str(n), "OOM", f">{total:.0f}", "", ""])
        finally:
            X = E = y = mask = None
        print("\t".join(rows[-1]), flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digress_scale.tsv")
    with open(out, "w") as f:
        f.write(f"# {torch.cuda.get_device_name(dev)}, {total:.1f} GiB; D_intra/DiGress "
                f"denoiser, batch 1, no extra features (a lower bound)\n")
        f.write("\n".join("\t".join(r) for r in rows) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
