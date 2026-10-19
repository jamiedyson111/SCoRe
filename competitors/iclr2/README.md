# ICLR [2]: Iterative Local Expansion (Bergmeister et al., 2024)

Competitor for the assembled-graph table. Trained on 100 Luxembourg regions
(`../data/lux_regions.pkl`, public data only, as configuration B), then asked for
five graphs of Oldenburg's size (6105 nodes). It sees only the node count, never
the release.

Repo: <https://github.com/AndreasBergmeister/graph-generation>, cloned on
BlueBEAR to `/rds/projects/c/chenhp-graphsyn/competitors/graph-generation`.

| file | what |
|---|---|
| `setup_bluebear.sh` | clone the repo, build its conda env, compile orca, install data + config + patch |
| `convert_data.py` | `lux_regions.pkl` → the repo's networkx pickles (`lux_regions.pkl`, `lux_regions_olde5.pkl`) |
| `lux_regions.yaml` | experiment config (EDM diffusion, 2 spectral features, batch 8, 200k steps) |
| `apply_patch.py` | checkpoint every 5k steps; `skip_metrics` for sampling (see below) |
| `run_train.sh` | one 12 h training job; resumes from the newest checkpoint |
| `run_sample.sh` | 5 graphs of 6105 nodes from a checkpoint, exported to `competitors/out/ICLR2/` on BlueBEAR |
| `export.py` | the repo's test pickle → `graph_s<i>.npz` (`n`, `edges`) |

## Run (BlueBEAR)

```bash
cd /rds/projects/c/chenhp-graphsyn
bash competitors/iclr2/setup_bluebear.sh          # once, on a LOGIN node (needs internet)
sbatch competitors/iclr2/run_train.sh             # resubmit after each 12 h timeout
sbatch competitors/iclr2/run_sample.sh [STEP]     # after training; STEP defaults to the newest
```

Training stops by itself at step 200,000; a chained job that starts afterwards
exits immediately. About 0.8 s per step, so roughly 44 GPU hours in total.

## Changes to the repo (`apply_patch.py`)

The repo saves checkpoints only after a validation round, and at 2000-6000 nodes
one round ran for over six hours, so a 12 h job ended with nothing saved. The
patch saves a checkpoint every 5k steps and turns the repo's validation off
(`validation.interval: 0`); `skip_metrics=true` makes sampling return the graphs
without the repo's metric suite. Scoring is done by our own
`syn_repeats_summary.py` against Oldenburg. EMA beta 0.999 is used for sampling.

## Output

On BlueBEAR: `competitors/out/ICLR2/graph_s0.npz … graph_s4.npz`, each `{n, edges (m, 2)}`. Copy to
the local `results/competitors/ICLR2/` and run
`D_inter/private_inference/syn_repeats_summary.py`; it becomes a column.
