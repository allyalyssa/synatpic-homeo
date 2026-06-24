"""Stage 3 — detect slow waves and record each wave's negative slope and the
time in the night at which it occurred.

The negative-going slope of a slow wave is our proxy for synaptic strength
(Riedner et al. 2007). We detect waves per subject with yasa.sw_detect, then map
each detection's time back onto the real recording clock using the epoch_idx that
preprocessing preserved. Time is what makes the later dissipation curve possible.

We concatenate the kept NREM epochs into one continuous signal and detect once
per subject (fast, 1 call). The 30 s seams can in principle spawn edge artifacts,
but yasa's duration/frequency/amplitude criteria reject almost all of them.

Output per subject: data/<dataset>/slow_waves/<subject>.npz with parallel arrays
  ch_idx    (n_sw,) int    index into ch_names
  slope     (n_sw,) float  negative-to-midcrossing slope, uV/s (the SHY proxy)
  night_sec (n_sw,) float  time of the wave from recording start, seconds
  ch_names  list[str]
"""
import sys

import numpy as np
import yasa

from config import PATHS, PREP, SW, ensure_dirs


def detect_subject(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    epochs = d["epochs"]                         # (n_nrem, n_ch, n_samp), volts
    epoch_idx = d["epoch_idx"]
    ch_names = [str(c) for c in d["ch_names"]]
    sf = float(d["sfreq"])
    n_nrem, n_ch, n_samp = epochs.shape

    # continuous (n_ch, n_nrem*n_samp), in microvolts for yasa
    cont = epochs.transpose(1, 0, 2).reshape(n_ch, -1) * 1e6
    sw = yasa.sw_detect(cont, sf, ch_names=ch_names, freq_sw=SW["freq_sw"],
                        dur_neg=SW["dur_neg"], dur_pos=SW["dur_pos"],
                        amp_ptp=SW["amp_ptp"], verbose=False)
    if sw is None:
        return None
    df = sw.summary()

    # concatenated Start -> real night time via the kept-epoch mapping
    j = np.clip((df["Start"].values // PREP["epoch_sec"]).astype(int), 0, n_nrem - 1)
    within = df["Start"].values - j * PREP["epoch_sec"]
    night_sec = epoch_idx[j] * PREP["epoch_sec"] + within

    ch_to_idx = {c: i for i, c in enumerate(ch_names)}
    ch_idx = df["Channel"].map(ch_to_idx).values.astype(int)
    return {
        "ch_idx": ch_idx,
        "slope": df["Slope"].values.astype(float),
        "night_sec": night_sec.astype(float),
        "ch_names": ch_names,
    }


def run(dataset):
    ensure_dirs()
    prep_dir, out_dir = PATHS[f"{dataset}_prep"], PATHS[f"{dataset}_sw"]
    files = sorted(prep_dir.glob("*.npz"))
    print(f"=== {dataset}: detecting slow waves in {len(files)} subjects ===")
    for f in files:
        res = detect_subject(f)
        if res is None or len(res["slope"]) == 0:
            print(f"  {f.stem}: no slow waves detected")
            continue
        np.savez(out_dir / f"{f.stem}.npz", **res)
        s = res["slope"]
        print(f"  {f.stem}: {len(s):6d} slow waves | mean slope {s.mean():6.1f} uV/s "
              f"across {len(res['ch_names'])} ch")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["dreams", "sleep_edf"]):
        run(ds)
