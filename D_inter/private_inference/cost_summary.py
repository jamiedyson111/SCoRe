"""Runtime and peak GPU memory of every s80 run (Section VI-E), as one TSV.

Runs LOCALLY in the `kmis` env, after fetch_s80.sh has copied the cost files
into results/costs/raw/ (and the full-graph samples into results/samples/assembled/):

    python cost_summary.py

Each BlueBEAR job wrote its own measurements (src/costlog.py): wall clock and
torch's peak allocated / reserved GPU memory, plus the GPU it ran on. This
script only collects them; the one thing it measures itself is the ASSEMBLY of
the full graph (placing every L_i and B_ij into one adjacency, CPU), timed here
on the fetched samples with assemble_syn.assemble.

Output, in results/costs/:
    runtime_memory.tsv    training per model; test-set sampling and whole-graph
                          inference (D_intra, D_inter, assembly) per configuration;
                          mean ± std over seeds where there are seeds
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from assemble_syn import _load, assemble
from syn_repeats_summary import load_run, seed_of

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', '..', 'results')
RAW = os.path.join(RESULTS, 'costs', 'raw')
GPUMEM = os.path.join(RESULTS, 'costs', 'gpumem')   # gpu_monitor.sh, one file per job

# (label, raw file): one trained model each. A and B share the Luxembourg models.
TRAIN = [('D_intra v3, Luxembourg (A, B)',             'train_D_intra_A_v3.json'),
         ('D_intra no-hop + centroid, Luxembourg (A, B)', 'train_D_intra_A_nohop_centroid.json'),
         ('D_intra v3, Lux + Olde jointly (C)',                'train_D_intra_C_v3.json'),
         ('D_intra no-hop + centroid, Lux + Olde jointly (C)', 'train_D_intra_C_nohop_centroid.json'),
         ('D_intra plain DiGress (no condition), Luxembourg (A, B)', 'train_D_intra_A_uncond.json'),
         ('D_intra plain DiGress (no condition), Lux + Olde jointly (C)', 'train_D_intra_C_uncond.json'),
         ('D_inter, Luxembourg (A, B)',                'train_D_inter_pair_lux_s80.json'),
         ('D_inter, Lux + Olde jointly (C)',                   'train_D_inter_pair_C_s80.json'),
         ('D_inter internal-degree, Luxembourg (A, B)',  'train_D_inter_pair_lux_s80_deg.json'),
         ('D_inter internal-degree, A fine-tuned on Olde (C)', 'train_D_inter_pair_C_s80_deg.json')]
SAMPLE = [(f'D_intra {c}', f'sample_D_intra_{c}.json') for c in
          ('B_v3', 'B_nohop_centroid', 'C_v3', 'C_nohop_centroid', 'A_v3', 'A_nohop_centroid',
           'B_uncond', 'C_uncond', 'A_uncond')] + \
         [(f'D_inter {c}', f'sample_D_inter_{c}.json') for c in
          ('B', 'C', 'A', 'B_deg', 'C_deg', 'A_deg', 'B_deg_realX', 'C_deg_realX')]
FULL = {'B': ('full_intra_B_v3_s80.json', 'full_inter_B_v3_s80-B.json', 'B_v3_s80-B'),
        'C': ('full_intra_C_v3_s80.json', 'full_inter_C_v3_s80-C.json', 'C_v3_s80-C'),
        # the internal-degree D_inter, around the same D_intra samples
        'B_deg': ('full_intra_B_v3_s80.json', 'full_inter_B_v3_s80-B_deg.json', 'B_v3_s80-B_deg'),
        'C_deg': ('full_intra_C_v3_s80.json', 'full_inter_C_v3_s80-C_deg.json', 'C_v3_s80-C_deg')}

# --final: exactly the models of the final tables (make_final_tables.sh). C is a
# fine-tune of A in both halves; its training rows are the fine-tuning alone.
FINAL_TRAIN = [('D_intra v3, Luxembourg (A, B)',                'train_D_intra_A_v3.json'),
               ('D_intra no h, Luxembourg (B no h)',             'train_D_intra_A_nohop_centroid.json'),
               ('D_intra v3, A fine-tuned on Olde (C)',          'train_D_intra_Cft_v3.json'),
               ('D_intra no h, fine-tuned on Olde (C no h)',     'train_D_intra_Cft_nohop_centroid.json'),
               ('D_intra plain DiGress, Olde only (80/20)',      'train_D_intra_O_uncond.json'),
               ('D_inter, Luxembourg (A, B)',                    'train_D_inter_pair_lux_s80_deg.json'),
               ('D_inter, A fine-tuned on Olde (C)',             'train_D_inter_pair_C_s80_deg.json'),
               ('D_inter no r, Luxembourg (B no r)',             'train_D_inter_pair_lux_s80_deg_nor.json'),
               ('D_inter no r, fine-tuned on Olde (C no r)',     'train_D_inter_pair_C_s80_deg_nor.json')]
FINAL_SAMPLE = [(f'D_intra {lab}', f'sample_D_intra_{c}.json') for lab, c in
                [('A v3', 'A_v3'), ('B (no h)', 'B_nohop_centroid'), ('B v3', 'B_v3'),
                 ('C (no h)', 'Cft_nohop_centroid'), ('C v3', 'Cft_v3'),
                 ('DiGress (Olde 80/20)', 'O_uncond')]] + \
               [(f'D_inter {lab}', f'sample_D_inter_{c}.json') for lab, c in
                [('A', 'A_deg'), ('B (no r)', 'B_deg_nor'), ('B', 'B_deg'),
                 ('C (no r)', 'Cft_deg_nor'), ('C', 'Cft_deg')]]
FINAL_FULL = {'B': ('full_intra_B_v3_s80.json', 'full_inter_B_v3_s80-B_deg.json', 'B_v3_s80-B_deg'),
              'C': ('full_intra_Cft_v3_s80.json', 'full_inter_Cft_v3_s80-C_deg.json',
                    'Cft_v3_s80-C_deg')}

# --final --graph vermont: the same models with Vermont in place of Oldenburg. The
# Luxembourg models (and A's sampling) are shared; everything else carries _verm.
VERM_TRAIN = [r for r in FINAL_TRAIN if 'Luxembourg' in r[0]] + [
    ('D_intra v3, A fine-tuned on Vermont (C)',      'train_D_intra_Cft_v3_verm.json'),
    ('D_intra no h, fine-tuned on Vermont (C no h)', 'train_D_intra_Cft_nohop_centroid_verm.json'),
    ('D_intra plain DiGress, Vermont only (80/20)',  'train_D_intra_O_uncond_verm.json'),
    ('D_inter, A fine-tuned on Vermont (C)',         'train_D_inter_pair_C_verm_s80_deg.json'),
    ('D_inter no r, fine-tuned on Vermont (C no r)', 'train_D_inter_pair_C_verm_s80_deg_nor.json')]
VERM_SAMPLE = [('D_intra A v3', 'sample_D_intra_A_v3.json')] + \
              [(f'D_intra {lab}', f'sample_D_intra_{c}_verm.json') for lab, c in
               [('B (no h)', 'B_nohop_centroid'), ('B v3', 'B_v3'), ('C (no h)', 'Cft_nohop_centroid'),
                ('C v3', 'Cft_v3'), ('DiGress (VT 80/20)', 'O_uncond')]] + \
              [('D_inter A', 'sample_D_inter_A_deg.json')] + \
              [(f'D_inter {lab}', f'sample_D_inter_{c}_verm.json') for lab, c in
               [('B (no r)', 'B_deg_nor'), ('B', 'B_deg'), ('C (no r)', 'Cft_deg_nor'), ('C', 'Cft_deg')]]
VERM_FULL = {'B': ('full_intra_B_v3_s80_verm.json', 'full_inter_B_v3_s80-B_deg_verm.json', 'B_v3_s80-B_deg'),
             'C': ('full_intra_Cft_v3_s80_verm.json', 'full_inter_Cft_v3_s80-C_deg_verm.json',
                   'Cft_v3_s80-C_deg')}


def device_peaks():
    """{jobid: (jobname, gpu, peak GB, wall s)} from gpu_monitor.sh's files."""
    out = {}
    for f in glob.glob(os.path.join(GPUMEM, '*.tsv')):
        for line in open(f):
            c = line.rstrip('\n').split('\t')
            if len(c) >= 5 and c[3]:
                out[c[0]] = (c[1], c[2], int(c[3]) / 1024.0, int(c[4]))
    return out


def dev(peaks, r):
    """Peak device memory (GB) of the job that wrote cost record r."""
    j = r.get('slurm_job_id') if r else None
    return f"{peaks[j][2]:.2f}" if j in peaks else ''


def read(name):
    p = os.path.join(RAW, name)
    return json.load(open(p)) if os.path.isfile(p) else None


def ms(v, dp):
    v = [x for x in v if x is not None]
    if not v:
        return ''
    if len(v) == 1:
        return f"{v[0]:.{dp}f}"
    return f"{np.mean(v):.{dp}f} ± {np.std(v, ddof=1):.{dp}f}"


def gb(mb):
    return None if mb is None else mb / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', default=os.path.join(RESULTS, 'private_inference',
                                                      'release_oldenburg_k7.pt'))
    ap.add_argument('--samples-dir', default=os.path.join(RESULTS, 'samples', 'assembled'))
    ap.add_argument('--out', default=os.path.join(RESULTS, 'costs', 'runtime_memory.tsv'))
    ap.add_argument('--final', action='store_true',
                    help='only the final tables\' models (default --out: runtime_memory_final.tsv)')
    ap.add_argument('--graph', default='oldenburg', choices=('oldenburg', 'vermont'),
                    help='with --final: the private graph (vermont: runtime_memory_vermont.tsv)')
    args = ap.parse_args()
    train, sample, full = (FINAL_TRAIN, FINAL_SAMPLE, FINAL_FULL) if args.final else (TRAIN, SAMPLE, FULL)
    if args.final and args.out == ap.get_default('out'):
        args.out = os.path.join(RESULTS, 'costs', 'runtime_memory_final.tsv')
    if args.graph == 'vermont':
        assert args.final, '--graph vermont goes with --final'
        train, sample, full = VERM_TRAIN, VERM_SAMPLE, VERM_FULL
        if args.out == os.path.join(RESULTS, 'costs', 'runtime_memory_final.tsv'):
            args.out = os.path.join(RESULTS, 'costs', 'runtime_memory_vermont.tsv')
        if args.release == ap.get_default('release'):
            args.release = os.path.join(RESULTS, 'private_inference', 'release_vermont_k7.pt')
    peaks = device_peaks()

    rows = [['TRAINING', 'GPU', 'train examples', 'batch', 'T', 'epochs',
             'training (h)', 's / epoch', 'validation (h)', 'whole fit (h)',
             'peak GPU alloc (GB)', 'peak GPU reserved (GB)', 'peak device (GB, nvidia-smi)']]
    for lab, f in train:
        r = read(f)
        if r is None:
            rows.append([lab, 'not fetched'])
            continue
        rows.append([lab, r['gpu'], str(r.get('n_train')), str(r['batch_size']),
                     str(r['diffusion_steps']), str(r['epochs']),
                     f"{r['train_s'] / 3600:.2f}", f"{r['s_per_epoch']:.1f}",
                     f"{r['val_s'] / 3600:.2f}", f"{r['fit_s'] / 3600:.2f}",
                     ms([gb(r['peak_alloc_mb'])], 2), ms([gb(r['peak_reserved_mb'])], 2),
                     dev(peaks, r)])

    rows += [[], ['TEST-SET SAMPLING (per seed)', 'GPU', 'items', 'batch', 'T',
                  'seeds', 'time (s)', 's / item', 'peak GPU alloc (GB)', 'peak GPU reserved (GB)',
                  'peak device (GB, nvidia-smi)']]
    for lab, f in sample:
        r = read(f)
        if r is None:
            rows.append([lab, 'not fetched'])
            continue
        sec = [x['seconds'] for x in r['runs']]
        rows.append([lab, r['gpu'], str(r['n_items']), str(r['sample_batch']),
                     str(r['diffusion_steps']), str(len(sec)), ms(sec, 1),
                     ms([s / r['n_items'] for s in sec], 3),
                     ms([gb(x['peak_alloc_mb']) for x in r['runs']], 2),
                     ms([gb(x['peak_reserved_mb']) for x in r['runs']], 2), dev(peaks, r)])

    release = _load(args.release) if os.path.isfile(args.release) else None
    rows += [[], ['WHOLE-GRAPH INFERENCE (per seed)', 'GPU', 'seeds',
                  'D_intra (s)', 'D_inter (s)', 'assembly (s, CPU)', 'total (s)',
                  'D_intra peak GPU alloc (GB)', 'D_inter peak GPU alloc (GB)',
                  'D_intra peak device (GB)', 'D_inter peak device (GB)']]
    for cfg, (fi, fx, tag) in full.items():
        ri, rx = read(fi), read(fx)
        if ri is None or rx is None:
            rows.append([f'{cfg} (D_intra all supernodes + D_inter all pairs)', 'not fetched'])
            continue
        ti = {x['seed']: x['seconds'] for x in ri['runs']}
        tx = {x['seed']: x['seconds'] for x in rx['runs']}
        ta = {}
        if release is not None:
            for f in glob.glob(os.path.join(args.samples_dir,
                                            f"blocks_{release['graph']}_k{release['k']}_{tag}_s*.pt")):
                s = seed_of(f)
                run = load_run(release, args.samples_dir, tag, s)
                t0 = time.perf_counter()
                assemble(release, *run)
                ta[s] = time.perf_counter() - t0
        seeds = sorted(set(ti) & set(tx))
        rows.append([f'{cfg} ({ri["n_items"]} supernodes + {rx["n_items"]} pairs)',
                     ri['gpu'], str(len(seeds)),
                     ms([ti[s] for s in seeds], 1), ms([tx[s] for s in seeds], 1),
                     ms([ta[s] for s in seeds if s in ta], 3),
                     ms([ti[s] + tx[s] + ta.get(s, 0.0) for s in seeds], 1),
                     ms([gb(x['peak_alloc_mb']) for x in ri['runs']], 2),
                     ms([gb(x['peak_alloc_mb']) for x in rx['runs']], 2),
                     dev(peaks, ri), dev(peaks, rx)])

    rows += [[], ["Times are wall clock on the GPU named; 'training' is the training epochs "
                  "alone, 'validation' DiGress's periodic validation (incl. its sampling "
                  "check), 'whole fit' both plus checkpointing. Peak memory is torch's "
                  "max_memory_allocated / max_memory_reserved (excludes the CUDA context); 'peak "
                  "device' is nvidia-smi's memory in use over the whole job, CUDA context "
                  "included. " + ("C fine-tunes A's model in both D_intra and D_inter; its "
                  "training rows are the fine-tuning alone. " if args.final else "Config C is "
                  "one joint training run (no pre-training or fine-tuning stage), except the "
                  "internal-degree D_inter's C, which fine-tunes A's model; its training row "
                  "is the fine-tuning alone. ") + "Mean ± std over seeds."]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    text = '\n'.join('\t'.join(r) for r in rows) + '\n'
    open(args.out, 'w').write(text)
    print(text + f"\nwrote {os.path.abspath(args.out)}")


if __name__ == '__main__':
    main()
