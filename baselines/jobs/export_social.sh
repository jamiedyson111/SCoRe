#!/bin/bash
# CPU job: our split of DBLP, Collab, LiveJournal and YouTube as competitor files.
#   sbatch --account=chenhp-graphsyn --qos=bbdefault --cpus-per-task=2 --mem=96G --time=03:00:00 baselines/jobs/export_social.sh
set -euo pipefail
P=/rds/projects/c/chenhp-graphsyn
. $P/D_intra/scripts/setup.sh
O=$P/coarsen_compare_expr/output
X=$P/baselines/export_partition_competitor_data.py
S=$P/D_intra/data/splits
cd $P/baselines
python3 -I $X --graph dblp   --npz $O/dblp_collab/com_dblp_csr_kmis_sized_k3.npz          --split $S/dblp_k3_s80.json   --out $P/competitor_data/dblp
python3 -I $X --graph collab --npz $O/dblp_collab/ogbl_collab_full_csr_kmis_sized_k3.npz  --split $S/collab_k3_s80.json --out $P/competitor_data/collab
python3 -I $X --graph youtube     --npz $O/livejournal_youtube/com_youtube_csr_kmis_cb_v4_k4_b500_l10.npz     --split $S/youtube_k4_s80.json     --out $P/competitor_data/youtube
python3 -I $X --graph livejournal --npz $O/livejournal_youtube/com_livejournal_csr_kmis_cb_v4_k4_b500_l10.npz --split $S/livejournal_k4_s80.json --out $P/competitor_data/livejournal
du -sh $P/competitor_data/*
