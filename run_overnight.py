"""Overnight automation - run once and walk away.

    python run_overnight.py

Resumes the Sleep-EDF download until all 153 recordings are present (or it
clearly stops advancing), then runs the full pipeline, the Sleep-EDF permutation
test, the cross-cohort comparison, and the manuscript. Every step is appended to
outputs/logs/overnight.log with timestamps so you can see exactly what happened
while you were away. No prompts, no manual input, safe to leave unattended.
"""
import time

import matplotlib
matplotlib.use("Agg")          # headless: never tries to open a window

from config import PATHS, ensure_dirs
import download
import run_pipeline
import compare
import manuscript
import robustness

ensure_dirs()
LOG = PATHS["logs"] / "overnight.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_psg():
    return len(list(PATHS["sleep_edf_raw"].glob("*-PSG.edf")))


def main():
    log(f"=== overnight start; {n_psg()}/153 Sleep-EDF PSG already on disk ===")

    # 1. Download to completion. Each pass resumes (skips finished files); we stop
    # when all 153 arrive or the count stalls for 3 passes (some files unreachable).
    prev, stall = -1, 0
    for attempt in range(300):
        try:
            download.download_sleep_edf(workers=10)
        except Exception as e:                       # never let one failure stop the night
            log(f"download pass errored (continuing): {e}")
        done = n_psg()
        log(f"download pass {attempt}: {done}/153 PSG")
        if done >= 153:
            break
        stall = stall + 1 if done == prev else 0
        if stall >= 3:
            log(f"download no longer advancing; proceeding with {done}/153")
            break
        prev = done
        time.sleep(60)

    # 2. Full pipeline on whatever downloaded.
    log(f"running full pipeline on {n_psg()} Sleep-EDF subjects")
    run_pipeline.run("sleep_edf")

    # 3. Validity + interpretation + write-up across all cohorts.
    log("running Sleep-EDF permutation test")
    robustness.permutation_test("sleep_edf")
    robustness.write_report("sleep_edf")

    cohorts = ["dreams", "dreams_subjects", "sleep_edf"]
    log("regenerating cohort comparison + manuscript")
    compare.main(cohorts)
    manuscript.build(cohorts)

    log(f"=== ALL DONE; final Sleep-EDF N = {n_psg()} subjects ===")
    log("see outputs/MANUSCRIPT.md and outputs/figures/cohort_comparison.png")


if __name__ == "__main__":
    main()
