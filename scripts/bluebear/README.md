# BlueBEAR job submitters

Each script writes its job scripts into the project's `logs/jobs/` and submits them with
`sbatch`; run it once on a BlueBEAR login node after uploading `D_intra/` and `D_inter/`.
Paths point at the project directory `/rds/projects/c/chenhp-graphsyn`.

| Script | Experiments |
|---|---|
| `submit_road_remb.sh` | Oldenburg / Vermont D_inter with the r embedding |
| `submit_vermont.sh` | Vermont: training, repeated sampling, whole graph |
| `submit_dblp_collab.sh` | DBLP -> Collab: D_intra and D_inter training |
| `submit_lj_youtube.sh` | LiveJournal -> YouTube: D_intra training |
| `submit_priority_nov.sh`, `submit_sampling_nov.sh` | Target-only and DiGress models; test-set sampling |
| `submit_collab_eval.sh` | Collab and YouTube sampling on the fixed test groups |
| `submit_collab_lanes.sh`, `submit_collab_whole.sh` | Collab whole graph (partial reconstruction) |
| `submit_youtube_inter.sh` | YouTube D_inter: reduced datasets and training |
| `submit_youtube_sampling.sh`, `submit_youtube_halves.sh` | YouTube D_inter sampling on the test groups |
| `fetch_remb.sh` | Copies the r-embedded samples back (run locally) |
