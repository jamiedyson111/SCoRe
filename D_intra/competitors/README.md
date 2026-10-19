# DiGress [14]: one-shot scaling test

Can one-shot DiGress generate a graph the size of Oldenburg (6105 nodes)? This
builds DiGress's denoiser with exactly D_intra's architecture (6 layers, dx 256,
de 64, 8 heads) and runs single denoising steps at doubling n, recording peak GPU
memory. Weights are random -- memory does not depend on them -- so no checkpoint
is needed. DiGress's extra features are left out, so the numbers are a lower bound.

| file | what |
|---|---|
| `digress_scale.py` | the test |
| `run_digress_scale.sh` | the BlueBEAR job (30 min limit; finishes in minutes) |
| `results/competitors/digress_scale.tsv` (top level) | the result, copied back from BlueBEAR |

## Run (BlueBEAR)

```bash
cd /rds/projects/c/chenhp-graphsyn/D_intra && sbatch competitors/run_digress_scale.sh
```

## Result (A100, 40 GB)

| n | peak memory | status |
|---|---|---|
| 256 | 0.40 GB | ok |
| 512 | 1.53 GB | ok |
| 1024 | 6.04 GB | ok |
| 2048 | 24.07 GB | ok |
| 3072, 4096, 6105 | > 39.5 GB | OOM |

`syn_repeats_summary.py` reads `digress_scale.tsv` and adds DiGress to the
assembled-graph table as an `OOM` column.
