# Costs: runtime, memory, coarsening (Sections IV-B, V-D, VI-E)

| file | what | made by |
|---|---|---|
| `coarsening_stats.tsv` | M, M/n against n/(⌊K/2⌋+1); supernode size median/mean/max b; m_c, m_c/m; Σn_i² against nb; Σ(n_i+n_j)² against 4m_c b²; n²; T. Oldenburg and Luxembourg, K = 7 (`--ks` for others) | `D_intra/extraction/coarsening_stats.py` (local, seconds) |
| `runtime_memory.tsv` | training time and peak GPU memory per model; test-set sampling; whole-graph inference split into D_intra, D_inter and assembly | `D_inter/private_inference/cost_summary.py` |
| `raw/*.json` | each job's own record (`src/costlog.py`): wall clock, torch peak allocated/reserved memory, GPU, batch, T, SLURM job id | written on BlueBEAR, fetched by `fetch_s80.sh` |
| `gpumem/*.tsv` | every GPU job's nvidia-smi peak device memory and wall clock (`gpu_monitor.sh`) | written on BlueBEAR to `/rds/projects/c/chenhp-graphsyn/gpumem/`, fetched by `fetch_s80.sh` |
