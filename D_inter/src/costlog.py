"""Runtime and peak GPU memory, written as JSON beside a run's output (Section VI-E).

Training: CostCallback goes in the Trainer's callbacks and writes cost_train.json
into the run directory when fit ends:

    fit_s           wall clock of trainer.fit, everything included
    train_s         the training epochs alone (no validation, no checkpointing)
    val_s           the validation epochs, including DiGress's sampling validation
    epochs, s_per_epoch (train_s / epochs)
    peak_alloc_mb   torch.cuda.max_memory_allocated over the whole fit
    peak_reserved_mb  the same for the caching allocator's reservation

Sampling: wrap each repeat in `with Measure() as m:` and pass m.record to
write(); each record has seconds and the peak memory of that repeat alone.

Every file also carries the GPU name, batch size, diffusion steps T, the SLURM
job id and the host, so results from different hardware are never mixed up.
"""
import json
import os
import socket
import time

import torch

try:
    from pytorch_lightning import Callback
except ImportError:                     # sampling-only environments
    Callback = object

MB = 1024.0 ** 2


def _cuda():
    return torch.cuda.is_available()


def reset_peak():
    if _cuda():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak():
    if not _cuda():
        return {'peak_alloc_mb': None, 'peak_reserved_mb': None}
    torch.cuda.synchronize()
    return {'peak_alloc_mb': round(torch.cuda.max_memory_allocated() / MB, 1),
            'peak_reserved_mb': round(torch.cuda.max_memory_reserved() / MB, 1)}


def context(cfg=None, **extra):
    """What the numbers were measured on and with."""
    c = {'gpu': torch.cuda.get_device_name(0) if _cuda() else 'cpu',
         'host': socket.gethostname(),
         'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
         'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    if cfg is not None:
        c.update(name=str(cfg.general.name),
                 dataset=str(cfg.dataset.get('pairs_file', cfg.dataset.name)),
                 batch_size=int(cfg.train.batch_size),
                 diffusion_steps=int(cfg.model.diffusion_steps),
                 finetune_from=cfg.general.get('finetune_from'))
    c.update(extra)
    return c


def write(path, record):
    """Never raises: a failed measurement must not fail the job (and its dependants)."""
    try:
        with open(path, 'w') as f:
            json.dump(record, f, indent=2, default=str)
        print(f"cost: {json.dumps(record, default=str)}\n  -> {os.path.abspath(path)}", flush=True)
    except Exception as e:                                  # noqa: BLE001
        print(f"WARNING: cost record not written to {path}: {e!r}", flush=True)


def safe_context(cfg=None, **extra):
    try:
        return context(cfg, **extra)
    except Exception as e:                                  # noqa: BLE001
        return {'context_error': repr(e), **extra}


class Measure:
    """`with Measure() as m:` -> m.record = {'seconds', 'peak_alloc_mb', 'peak_reserved_mb'}."""

    def __enter__(self):
        try:
            reset_peak()
        except Exception:                                   # noqa: BLE001
            pass
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        secs = round(time.perf_counter() - self.t0, 2)
        try:
            self.record = {'seconds': secs, **peak()}
        except Exception:                                   # noqa: BLE001
            self.record = {'seconds': secs, 'peak_alloc_mb': None, 'peak_reserved_mb': None}
        return False


class CostCallback(Callback):
    """Training cost of one fit, written to <run dir>/cost_train.json."""

    def __init__(self, cfg, path='cost_train.json'):
        super().__init__()
        self.cfg, self.path = cfg, path
        self.train_s = self.val_s = 0.0
        self.epochs = 0

    def on_fit_start(self, trainer, pl_module):
        self.t_fit = time.perf_counter()
        try:
            reset_peak()
        except Exception:                                   # noqa: BLE001
            pass

    # Lightning 2 runs the end-of-epoch validation INSIDE the training epoch
    # (before on_train_epoch_end), so validation inside the window is subtracted.
    def on_train_epoch_start(self, trainer, pl_module):
        self.t_ep, self.val_in_ep = time.perf_counter(), 0.0

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_s += time.perf_counter() - self.t_ep - self.val_in_ep
        self.epochs += 1

    def on_validation_epoch_start(self, trainer, pl_module):
        self.t_val = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            dt = time.perf_counter() - self.t_val
            self.val_s += dt
            self.val_in_ep = getattr(self, 'val_in_ep', 0.0) + dt

    def on_fit_end(self, trainer, pl_module):
        try:
            self._write(trainer)
        except Exception as e:                              # noqa: BLE001
            print(f"WARNING: training cost not recorded: {e!r}", flush=True)

    def _write(self, trainer):
        n_train = None
        try:
            n_train = len(trainer.datamodule.train_dataloader().dataset)
        except Exception:                                   # noqa: BLE001
            pass
        write(self.path, {
            **safe_context(self.cfg, phase='train', n_train=n_train),
            'fit_s': round(time.perf_counter() - self.t_fit, 1),
            'train_s': round(self.train_s, 1),
            'val_s': round(self.val_s, 1),
            'epochs': self.epochs,
            's_per_epoch': round(self.train_s / max(self.epochs, 1), 2),
            **peak()})
