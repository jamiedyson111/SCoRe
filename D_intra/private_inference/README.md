# Private inference -- D_intra half

Stages 0 and 1 of the four-stage pipeline that turns a private graph into a
synthetic one. Stages 2 and 3 live in `D_inter/private_inference/`.

```
  stage 0  make_private_release.py      kmis env, local, CPU
             G^priv --> release_<graph>_k<k>.pt        the coarse release
  stage 1  sample_private_clusters.py   digress env, GPU
             release --> clusters_<graph>_k<k>.pt      one L_i per supernode
                     |
                     |  copy both files to D_inter/private_inference/out/
                     v
  stage 2  D_inter/private_inference/sample_private_blocks.py    one B_ij per pair
  stage 3  D_inter/private_inference/assemble_syn.py             L^syn + scoring
```

`../../make_graph.sh` runs all four stages from one command, submitting stages 1
and 2 to BlueBEAR over ssh and waiting for them. Nothing below needs typing by
hand for a standard run; it is here for when one stage has to be run on its own,
and `--dry-run` prints exactly these commands without submitting anything.

The two models can never share a process -- D_intra and D_inter each define a
top-level `src` package and importing both collides. That is why the pipeline is
file-mediated rather than one script.

## Stage 0 -- the release

```bash
python make_private_release.py --graph oldenburg --k 7
```

Coarsens with the same three k-MIS calls as `extraction/kmis_pairs.py`, so the
decomposition matches what the models were trained under. Emits per-supernode
`{n, rho_in, centre, h, y_raw_intra, nodes}` and per-connected-pair
`{sn_i, sn_j, w}`.

**The release carries the assembled condition, not its ingredients.**
`y_raw_intra` is built by the same `build_condition()` call the training records
use, so stage 1 never re-derives the layout. That matters: the current D_intra
condition is 10-dimensional -- `[n, rho_in, k, h1..h7]`, where `h` is the hop
histogram, the fraction of a supernode's non-centre members at shortest-path
distance 1..k from its centre. An earlier version of this file rebuilt a
3-vector from `n`/`rho_in`/`k` and would have fed the model a silently truncated
condition. Verified: all 8815 Luxembourg supernodes match
`pairs_A_lux2minn_k7.pt` bit for bit.

Two consequences worth stating in the write-up:

- The release is **7 numbers per supernode larger** than the `[n, rho_in]` the
  brief describes, because the model requires them. That is the first place to
  look if a DP budget is attached later.
- `hop_histogram` does its BFS over the **full** graph, not the induced
  subgraph, so a member two hops from the centre may be reached through a node in
  another supernode. `h` describes the supernode's embedding in `G`, not only its
  interior.

No noise is added -- `n`, `rho_in` and `w_ij` are exact. The hats in the brief mean
*released*, not *noised*; a DP mechanism, if added later, belongs here between the
exact statistic and the file, and nothing downstream would change.

Every supernode is kept, including those below the `MIN_N = 3` the training sets
dropped. Those nodes still exist in `G^priv`, and dropping them would silently
shrink `L^syn`.

It also writes `clusters_<graph>_k<k>_true.pt` beside the release: the private
graph's *real* internal blocks, in exactly the format stage 1 emits. Stage 2 uses
that to validate itself without stage 1 -- see the D_inter README. It exists as a
file because stage 2 runs in the GPU environment, which has neither `h5py` nor
`torch_sparse` and so cannot rebuild the blocks itself.

Self-checks on exit: supernode sizes must sum to the fine node count, and internal
plus cross edges must equal the fine edge count. A wrong coarsening fails here
rather than three stages later.

## Stage 1 -- internal blocks

```bash
sbatch private_inference/run_private_clusters.sh \
    /abs/path/checkpoints/cluster_lux/epoch=NNN.ckpt \
    /abs/path/release_oldenburg_k7.pt \
    cluster_lux
```

`sample_conditional.py` with the conditions taken from the release instead of the
dataset's test split. The dataset is still loaded -- the checkpoint needs its
`dataset_infos`, and its blob carries the standardisation.

**The condition is standardised with the training graph's statistics, not the
private graph's.** Those numbers are part of the model. `standardise()` asserts
that `release['intra_cond_names']` equals the checkpoint blob's `cond_names`, so a
schema change fails loudly on the first batch rather than producing plausible
garbage.

A constant feature like `k` is left on its raw scale rather than divided by a tiny
epsilon -- `cond_constant` is True and `cond_std` is 1.0, see the comment in
`kmis_pairs.standardise`. A mismatched `k` therefore shifts the feature by a few
units instead of ~10^6, but it is still a train/test mismatch, hence the hard
assert.

Supernodes with `n < 3` are filled arithmetically: `n = 1` has no internal slot,
`n = 2` has exactly one whose occupancy `rho_in` determines outright. The model
never saw sizes that small.

## Which checkpoint to use

D_intra was retrained on 2026-08-12, so the caveat that stage 1 output was
provisional against a pre-k-MIS-fix checkpoint no longer applies -- **provided you
pass a checkpoint from one of the completed A/B/C runs and name its dataset.**

Use **configuration B** (`cluster_B_lux2olde_k7`): trained on Luxembourg, applied
to Oldenburg, which is the public → private direction this stage demonstrates.
(Oldenburg replaced Minnesota as the private graph on 2026-10-05. B's training
data is unchanged by that, so the existing `transfer_B` checkpoint is the one to
use.)
That is now the script's default for the DATASET argument. It used to be
`cluster_lux`, which is wrong twice over -- legacy 3-dim condition, and a pairs
file predating the coarsening fix.

Find the checkpoint on BlueBEAR, taking the **newest** run directory:

```bash
ls -td /rds/projects/c/chenhp-graphsyn/D_intra/outputs/*/*-transfer_B/ | head -1
```

Then, with `CKPT` set to an `epoch=N.ckpt` under `<that dir>/checkpoints/transfer_B/`:

```bash
sbatch private_inference/run_private_clusters.sh \
    "$CKPT" out/release_oldenburg_k7.pt cluster_B_lux2olde_k7
```

> **Take the newest run directory deliberately.** Several configurations have more
> than one dated run, because the 11 August jobs crashed at epoch 29 and were
> resubmitted. `run_transfer.sh` originally globbed across all of them and picked
> the oldest checkpoint, which sampled A and B from the *crashed* runs; that is
> fixed, but the stale directories are still on disk and a hand-written path can
> still land in one.

The validation curve wanders between roughly 88 and 106 after epoch 10 (best
84.59 for A, 86.65 for B), so differences between individual checkpoints from the
same run are within noise. Do not read significance into which epoch was picked.

Stage 2 can be validated without stage 1 at all, by feeding it
`clusters_<graph>_k<k>_true.pt`; see the D_inter README.
