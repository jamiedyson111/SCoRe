# ER and SC-RG Baselines — Tested on Configuration B where splits[2] = test is used


The input file is:

```text
pairs_B_lux2olde_k7.pt
```

The Oldenburg test condition contains:

```text
[n, rho_in, k, h1, ..., h7]
```

---

## 1. Load the BEAR environment

On the BEAR server:

```bash
module load bear-apps/2021a/live
module load PyTorch/1.10.0-foss-2021a
```

---

## 2. Erdős–Rényi \(G(n,m)\) baseline

For each Oldenburg test supernode \(i\), the ER baseline samples:

\[
\hat{G}_i \sim G(n_i,m_i),
\]

by uniformly selecting exactly \(m_i\) edges from the \(\binom{n_i}{2}\) possible node pairs.

The baseline preserves:

```text
n_i, m_i
```

and does not use the hop condition \(h_i\).

### Run 5 stochastic samples

```bash
python3 ER_intra.py \
  --input pairs_B_lux2olde_k7.pt \
  --output-dir er_results \
  --num-samples 5 \
  --seed 42 \
  --check
```

The generated results are stored in:

```text
  er_results/ER_samples_s0.txt
  ......
```


---

## 3. Synopsis-Constrained Random Graph (SC-RG)

SC-RG uses the same count information as ER but additionally uses the hop synopsis \(h_i\).

For each Oldenburg test supernode, SC-RG:

1. converts \(h_i\) into centroid-based hop-layer sizes;
2. creates a random radial backbone;
3. adds random edges within the same layer or between adjacent layers;
4. continues until exactly \(m_i\) edges are present.

The baseline therefore preserves:

```text
n_i, m_i, h_i
```

while the detailed topology remains random.

### Run 5 stochastic samples

```bash
python3 SC_RG_intra.py \
  --input pairs_B_lux2olde_k7.pt \
  --output-dir scrg_results \
  --num-samples 5 \
  --seed 42 \
  --check
``` 
The generated results are stored in:

```text
  scrg_results/scrg_samples_s0.txt
  ......
```


## 4. Output format

Both baselines use the same block format as the \(D_{\rm intra}\) diffusion output:

```text
N=52
supernode=82 n_true=52 m_true=58
cond: 52.0000 0.0437 7.0000
X:
1 0 0 ...
E:
0 1 0 ...
1 0 0 ...
...
```

