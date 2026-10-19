# Changes to stock DiGress

Baseline: `cvignac/DiGress` at commit `780242b`. A pristine clone is at
`../DiGress/` — `diff -r ../DiGress/src src` shows exactly what follows and
nothing else.

Everything here is **generic conditional-generation machinery**. Nothing in this
folder is specific to D_intra or D_inter; those live in their own folders and are
copied into each experiment folder (formerly by `sync.sh`, now in archive/), which is what makes
D_intra/ and D_inter/ self-contained, uploadable trees.

## `from src.*` and PYTHONPATH — resolved, but know the history

Stock DiGress mixes two import styles: `from src.diffusion import
diffusion_utils` and bare `from diffusion.noise_schedule import ...`. Where the
first one lands depends entirely on how `src` is found.

**On BlueBEAR `src` is not pip-installed at all**, so `from src import utils` in
`main.py` died with `ModuleNotFoundError: No module named 'src'`.

**Locally it *was* installed** — an editable install pointing at `../DiGress/`,
the pristine clone — so `src.*` imports silently resolved to unmodified upstream
files while bare imports resolved to this tree. That cost a debugging cycle: a
change to `mask_distributions` in `src/diffusion/diffusion_utils.py` was loaded
from the pristine copy and simply ignored, with no error.

Both are fixed the same way. Every run script now sets

```bash
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
```

so `src` resolves to **the experiment's own tree**, identically on the cluster
and locally. Verified: `src.diffusion.diffusion_utils.__file__` now points inside
`D_inter/`, not `DiGress/`.

Do NOT `pip install -e .` to fix this instead. It would bind `src` to whichever
folder you installed from, and D_intra's imports would then silently load
D_inter's model code. PYTHONPATH is per-job, so each tree keeps its own.

If you run `main.py` by hand rather than through a script, set it yourself:

```bash
cd D_inter && PYTHONPATH=$PWD python src/main.py dataset=pair_lux +experiment=pair_lux
```

`_restrict_to_cross` still lives in `diffusion_model_discrete.py` rather than in
`diffusion_utils.mask_distributions` where it belongs. That was a workaround for
the trap above and is no longer necessary — it is merely not worth moving now,
since `diffusion_utils.py` is byte-identical to stock and that is worth keeping.

## `src/diffusion_model_discrete.py` — 9 hunks + partially-observed mode

DiGress already carries a per-graph vector `y` that is never noised and never
predicted, which makes it a natural conditioning channel. But every stock dataset
leaves it width 0, so several code paths silently discard it. The edits:

1. `training_step` no longer passes `y` to the loss as a prediction target.
   It is an input; `output_dims['y']` is 0, so there is nothing to compare against.
2. `reconstruction_logp` accepts and uses the condition instead of building a
   width-0 one from the feature sampler.
3. `compute_val_loss` passes the condition down to it.
4. `sample_batch` accepts `cond_y`.
5. `sample_batch` applies it *after* drawing the initial noise.
6. The reverse loop no longer overwrites `y`. `sample_p_zs_given_zt` returns a
   width-0 `y` every step, which would discard the condition after one step.
7. `sample_batch` falls back to a bank of real training conditions when called
   with none — which is what DiGress's own periodic validation sampling does.
   Without this a conditionally-trained model crashes at the first validation.
8. `sample_batch` accepts an explicit `num_nodes` tensor, so generated graphs
   can be forced to the true size of the thing being conditioned on.
9. `apply_noise` puts `y` into `noisy_data['y_t']` instead of `z_t.y`. **Bug
   fix, not a feature.** The line above it is
   `PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t)`, and `X_t` is what `F.one_hot`
   returns — `int64`. So `type_as` was casting the float condition to an
   integer and truncating it: a standardised `0.72` reached the model as `0`,
   `-0.31` as `0`. D_intra's three features were being quantised to whole
   numbers on every training and validation step. `mask()` never touches `y`,
   so `z_t.y` and `y` differ *only* by that cast.

Hunks 1–8 are guarded by `if cond_y is not None` / `if y is not None`, so the
stock unconditional datasets (planar, sbm, qm9, moses, guacamol) behave exactly
as before. Hunk 9 needs no guard: with a width-0 `y` the two expressions are
identical, so it is a no-op for every unconditional dataset.

Verified in the `digress` env (torch 2.0.1):

```python
>>> torch.tensor([[0.317, -0.72, 8.93]]).type_as(F.one_hot(torch.tensor([0]), 2))
tensor([[0, 0, 8]])
```

### Partially-observed mode (`pair_mode`)

For D_inter, part of the graph is GIVEN rather than generated. A dataset opts in
by setting `dataset_infos.pair_mode = True`; every other dataset leaves the
attribute absent, `getattr` returns `False`, and none of the code below runs.

The contract: `X` is a one-hot **side** indicator, and an entry of `E` is a
cross entry exactly when its two endpoints have different node classes. That one
rule is what the whole mode is built on — it makes the mask a function of the
data rather than of node ordering, and because `X` is never noised, the mask is
exact at every training step and every reverse step rather than decaying with t.

    cross_mask()          the mask, from X and node_mask
    _restrict_to_cross()  neutralises the excluded distributions for the val NLL
    apply_noise           noises the cross block only; X_t = X
    training_step         zeroes non-cross rows of the CE target, so they reach
                          neither the numerator nor the denominator of the mean
    compute_val_loss      builds the mask, threads it through kl_prior /
                          compute_Lt / reconstruction_logp, restricts the L0
                          term, and logs val/cross_CE
    sample_batch          `given=` holds X and the non-cross part of E fixed,
                          re-imposed after every reverse step
    sample_batch          calls `dataset_infos.sample_conditioning(bs, device,
                          test)` when it needs condition + sizes + context

`_restrict_to_cross` sets excluded entries to the same **uniform** row in both
true and pred, so their KL is exactly zero. Uniform and not one-hot: these feed
`log(pred)`, and a one-hot row would make that `-inf` and turn the excluded
terms into `nan` rather than zero.

It lives here rather than in `diffusion_utils.mask_distributions`, which is
where it naturally belongs, purely because of the `from src.*` trap above.

## `src/main.py` — 4 hunks

New dataset branches, purely additive. Each branch imports its dataset module
*inside* the branch, so a missing module only fails if you actually select that
dataset.

## New files

    src/datasets/generic_dataset.py    generic graph-list dataset
    configs/dataset/generic.yaml
    configs/experiment/generic.yaml
    configs/dataset/coarse_sbm.yaml
    configs/experiment/coarse_sbm.yaml

## Known issue

`main.py` has a `coarse_sbm` branch importing `datasets/coarse_dataset.py`, which
is not here — it was lost in a local re-sync. The import is inside the branch, so
it only fails if you run `dataset=coarse_sbm`.

## What is deliberately NOT here

- `src/datasets/cluster_dataset.py`, `src/sample_conditional.py`, the `cluster_*`
  configs and the `pairs_*.pt` data — all D_intra, in `../D_intra/`
- `src/datasets/pair_dataset.py` and the `pair_*` configs — D_inter, in
  `../D_inter/`. `pair_mode` above is generic; the side encoding, the
  cross-block-only edge marginal and the W_out-degree metric are not, and stay
  in that folder
- `outputs/`, `results/`, `generated_samples/` — DiGress rebuilds these
- `logs/` is empty but must exist; SLURM fails immediately if `--output`'s
  directory is missing

## Before uploading

`setup.sh` activates `/rds/projects/c/chenhp-graphsyn/envs/digress`, which lives
outside every project folder.

```bash
ssh USER@bluebear.bham.ac.uk "ls -d /rds/projects/c/chenhp-graphsyn/envs/digress && echo ENV OK || echo ENV MISSING"
```

Rebuilding it (torch + graph_tool + rdkit + lightning) is a multi-hour job.

`src/analysis/orca/orca` must keep its executable bit. If it loses `+x` the run
crashes at the very END, after training has finished. Both projects carry the fix.
