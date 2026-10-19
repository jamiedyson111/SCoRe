"""Two small, idempotent patches to the ICLR [2] checkout (graph_generation/training.py).

    python apply_patch.py --repo $REPO

The repo saves a checkpoint only after a validation round, and a validation
round samples every val graph once per EMA beta and then runs its full metric
suite (orbit counts, spectra, wavelets, isomorphism-based novelty/uniqueness,
spring-layout plots). At 2000-6000 nodes the first one was still running six
hours after it started (2026-10-13), so a 12 h job would end with no checkpoint
and every resubmission would start again from step 0.

1. Trainer.train: with training.checkpoint_interval = N, save a checkpoint every
   N steps on its own, so validation can be switched off (validation.interval: 0).
2. Trainer.evaluate: with skip_metrics = true, return right after generating --
   no metrics, no plots. Sampling only needs the graphs; they are scored by
   syn_repeats_summary.py against Oldenburg.
"""
import argparse
import os

MARK = "# competitors/iclr2/apply_patch.py"

CKPT = f'''
            _ck = self.cfg.training.get("checkpoint_interval", 0)   {MARK}
            if _ck and self.step % _ck == 0 and self.cfg.training.save_checkpoint:
                self.save_checkpoint()
                print(f"checkpoint saved at step {{self.step}}", flush=True)
'''

SKIP = f'''        if self.cfg.get("skip_metrics", False):   {MARK}
            return results

'''


def patch(path, anchor, insert, before):
    s = open(path).read()
    first = insert.strip().splitlines()[0]
    if first in s:
        print(f"  already patched: {first[:50]}")
        return s
    assert s.count(anchor) == 1, f"anchor not found once: {anchor!r}"
    s = s.replace(anchor, insert + anchor if before else anchor + insert)
    open(path, "w").write(s)
    print(f"  patched: {first[:50]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    p = os.path.join(ap.parse_args().repo, "graph_generation", "training.py")
    patch(p, "                self.log({\"training\": loss_terms})\n", CKPT, before=False)
    patch(p, "        # Validate graphs\n", SKIP, before=True)


if __name__ == "__main__":
    main()
