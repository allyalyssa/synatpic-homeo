"""Adversarial controls — try to BREAK the preliminary findings.

Every control prints a one-line verdict (SURVIVES / WEAKENED / FAILS) with the
number that justifies it. Fixed seed, subject-level throughout, N stated.

    python controls.py A          # amplitude-matched slope (Tier 0)
    python controls.py B          # within-stage density
    ...
"""
import sys

import numpy as np
from scipy import stats

from config import PATHS, SEED

SEED_USED = SEED
COHORTS = ["dreams_subjects", "sleep_edf", "dreams"]   # healthy first; dreams=patients
PRETTY = {"dreams_subjects": "DREAMS Subjects (healthy)",
          "sleep_edf": "Sleep-EDF", "dreams": "DREAMS Patients (pathology)"}


def _load_sw(dataset):
    """Yield (subject, dict) of enriched slow-wave tables for a cohort."""
    for f in sorted(PATHS[f"{dataset}_sw"].glob("*.npz")):
        yield f.stem, np.load(f, allow_pickle=True)


# ---------------------------------------------------------------------------
# Control A — amplitude-matched slope
# ---------------------------------------------------------------------------
def control_a(n_deciles=10, min_waves=30):
    print(f"=== CONTROL A: amplitude-matched slope (seed={SEED_USED}) ===")
    print("Original finding: per-wave SLOPE is flat overnight (the 'dissociation').")
    print("If slope DECLINES once amplitude is held fixed, that finding is an artifact.\n")

    for ds in COHORTS:
        raw_e, raw_l, amp_e, amp_l, mat_e, mat_l = [], [], [], [], [], []
        for subj, d in _load_sw(ds):
            slope, ptp, night = d["slope"], d["ptp"], d["night_sec"]
            t0, t1 = night.min(), night.max()
            third = (t1 - t0) / 3
            e = night < t0 + third
            l = night > t1 - third
            if e.sum() < min_waves or l.sum() < min_waves:
                continue
            raw_e.append(slope[e].mean()); raw_l.append(slope[l].mean())
            amp_e.append(ptp[e].mean()); amp_l.append(ptp[l].mean())

            # match within amplitude deciles defined on the pooled early+late PTP
            edges = np.percentile(np.concatenate([ptp[e], ptp[l]]),
                                  np.linspace(0, 100, n_deciles + 1))
            me, ml = [], []
            for b in range(n_deciles):
                lo, hi = edges[b], edges[b + 1]
                ein = e & (ptp >= lo) & (ptp <= hi)
                lin = l & (ptp >= lo) & (ptp <= hi)
                if ein.sum() and lin.sum():
                    me.append(slope[ein].mean()); ml.append(slope[lin].mean())
            if me:
                mat_e.append(np.mean(me)); mat_l.append(np.mean(ml))

        n = len(mat_e)
        raw_e, raw_l = np.array(raw_e), np.array(raw_l)
        amp_e, amp_l = np.array(amp_e), np.array(amp_l)
        mat_e, mat_l = np.array(mat_e), np.array(mat_l)
        p_raw = stats.ttest_rel(raw_e, raw_l)[1]
        p_amp = stats.ttest_rel(amp_e, amp_l)[1]
        p_mat = stats.ttest_rel(mat_e, mat_l)[1]
        # Cohen's dz for the matched comparison
        diff = mat_e - mat_l
        dz = diff.mean() / (diff.std(ddof=1) + 1e-12)

        print(f"--- {PRETTY[ds]}  (N={n}) ---")
        print(f"  amplitude (PTP)  early {amp_e.mean():6.1f} vs late {amp_l.mean():6.1f} uV   p={p_amp:.1e}  "
              f"({'declines' if amp_e.mean() > amp_l.mean() and p_amp < 0.05 else 'flat'})")
        print(f"  RAW slope        early {raw_e.mean():6.1f} vs late {raw_l.mean():6.1f} uV/s p={p_raw:.3f}")
        print(f"  MATCHED slope    early {mat_e.mean():6.1f} vs late {mat_l.mean():6.1f} uV/s p={p_mat:.3f}  dz={dz:+.2f}")
        decline = mat_e.mean() > mat_l.mean() and p_mat < 0.05
        verdict = ("FAILS (slope declines within matched amplitude -> dissociation is an artifact)"
                   if decline else
                   "SURVIVES (slope still flat after amplitude matching)")
        print(f"  VERDICT: {verdict}\n")


# ---------------------------------------------------------------------------
# Control B — within-stage density (is the decline just the N3->N2 shift?)
# ---------------------------------------------------------------------------
def control_b(min_stage_min=3.0):
    print(f"=== CONTROL B: within-stage density (seed={SEED_USED}) ===")
    print("Original finding: SW density (waves/min total NREM) declines overnight.")
    print("If it only declines because N3 (wave-rich) gives way to N2, that is sleep")
    print("architecture, not homeostasis. Test: density per min of N3 ONLY and N2 ONLY.\n")

    for ds in COHORTS:
        rows = {3: [[], []], 2: [[], []], "tot": [[], []]}   # stage -> [early[], late[]]
        n3frac_e, n3frac_l = [], []
        for subj, d in _load_sw(ds):
            night, sw_stage = d["night_sec"], d["sw_stage"]
            nrem_sec, nrem_stage = d["nrem_sec"], d["nrem_stage"]
            t0, t1 = nrem_sec.min(), nrem_sec.max()
            third = (t1 - t0) / 3
            e_ep, l_ep = nrem_sec < t0 + third, nrem_sec > t1 - third
            e_sw, l_sw = night < t0 + third, night > t1 - third

            tot_e_min, tot_l_min = e_ep.sum() * 0.5, l_ep.sum() * 0.5
            if tot_e_min > 2 and tot_l_min > 2:
                rows["tot"][0].append(e_sw.sum() / tot_e_min)
                rows["tot"][1].append(l_sw.sum() / tot_l_min)
                n3frac_e.append((e_ep & (nrem_stage == 3)).sum() / e_ep.sum())
                n3frac_l.append((l_ep & (nrem_stage == 3)).sum() / l_ep.sum())

            for st in (3, 2):
                em = ((nrem_sec < t0 + third) & (nrem_stage == st)).sum() * 0.5
                lm = ((nrem_sec > t1 - third) & (nrem_stage == st)).sum() * 0.5
                if em >= min_stage_min and lm >= min_stage_min:
                    rows[st][0].append(((night < t0 + third) & (sw_stage == st)).sum() / em)
                    rows[st][1].append(((night > t1 - third) & (sw_stage == st)).sum() / lm)

        print(f"--- {PRETTY[ds]} ---")
        # show the architectural confound is real
        if n3frac_e:
            print(f"  N3 fraction of NREM:   early {np.mean(n3frac_e):.0%} vs late {np.mean(n3frac_l):.0%} "
                  f"(the architectural shift, p={stats.ttest_rel(n3frac_e, n3frac_l)[1]:.1e})")
        for key, lab in [("tot", "TOTAL-NREM density"), (3, "N3-only density"), (2, "N2-only density")]:
            e, l = np.array(rows[key][0]), np.array(rows[key][1])
            if len(e) < 5:
                print(f"  {lab:20} N={len(e)} (too few)")
                continue
            p = stats.ttest_rel(e, l)[1]
            pct = (e.mean() - l.mean()) / e.mean() * 100
            print(f"  {lab:20} N={len(e):3d}  early {e.mean():5.1f} vs late {l.mean():5.1f} waves/min "
                  f"({pct:+.0f}%, p={p:.1e})")
        # verdict from within-stage results
        n3 = [np.array(rows[3][0]), np.array(rows[3][1])]
        n2 = [np.array(rows[2][0]), np.array(rows[2][1])]
        n3_ok = len(n3[0]) >= 5 and stats.ttest_rel(*n3)[1] < 0.05 and n3[0].mean() > n3[1].mean()
        n2_ok = len(n2[0]) >= 5 and stats.ttest_rel(*n2)[1] < 0.05 and n2[0].mean() > n2[1].mean()
        if n3_ok and n2_ok:
            v = "SURVIVES (density declines within BOTH N3 and N2 -> homeostatic, not just architecture)"
        elif n3_ok or n2_ok:
            v = f"WEAKENED (declines within {'N3' if n3_ok else 'N2'} only)"
        else:
            v = "FAILS (no within-stage decline -> the drop is the N3->N2 shift, not homeostasis)"
        print(f"  VERDICT: {v}\n")


# ---------------------------------------------------------------------------
# Control C — does density track canonical SWA (0.5-4 Hz power)?
# ---------------------------------------------------------------------------
def control_c(n_bins=12):
    from scipy.signal import welch
    print(f"=== CONTROL C: density vs SWA power (seed={SEED_USED}) ===")
    print("If my density metric does not correlate with canonical SWA (0.5-4 Hz power),")
    print("it is not a valid homeostatic marker. Within-subject r across 12 night bins.\n")

    for ds in COHORTS:
        within_r, swa_decl, dens_decl = [], [], []
        for f in sorted(PATHS[f"{ds}_prep"].glob("*.npz")):
            p = np.load(f, allow_pickle=True)
            epochs, eidx, sf = p["epochs"], p["epoch_idx"], float(p["sfreq"])
            swf = PATHS[f"{ds}_sw"] / f.name
            if not swf.exists():
                continue
            night = np.load(swf, allow_pickle=True)["night_sec"]

            # SWA per epoch: 0.5-4 Hz band power, averaged over channels
            fr, psd = welch(epochs, fs=sf, nperseg=int(2 * sf), axis=2)
            band = (fr >= 0.5) & (fr <= 4)
            swa_ep = psd[:, :, band].mean(axis=(1, 2))            # (n_nrem,)
            et = eidx * 30.0

            edges = np.linspace(et.min(), et.max(), n_bins + 1)
            ebin = np.clip(np.digitize(et, edges) - 1, 0, n_bins - 1)
            sbin = np.clip(np.digitize(night, edges) - 1, 0, n_bins - 1)
            swa_b = np.array([swa_ep[ebin == b].mean() if (ebin == b).any() else np.nan
                              for b in range(n_bins)])
            mins = np.array([(ebin == b).sum() * 0.5 for b in range(n_bins)])
            dens_b = np.array([(sbin == b).sum() / mins[b] if mins[b] > 0 else np.nan
                               for b in range(n_bins)])
            ok = ~(np.isnan(swa_b) | np.isnan(dens_b))
            if ok.sum() >= 6:
                within_r.append(stats.pearsonr(swa_b[ok], dens_b[ok])[0])
                swa_decl.append((swa_b[ok][0] - swa_b[ok][-1]) / swa_b[ok][0])
                dens_decl.append((dens_b[ok][0] - dens_b[ok][-1]) / (dens_b[ok][0] + 1e-9))

        r = np.array(within_r)
        print(f"--- {PRETTY[ds]}  (N={len(r)}) ---")
        print(f"  within-subject r(SWA, density) across bins: mean {r.mean():.2f} "
              f"(median {np.median(r):.2f}, {(r > 0.5).mean():.0%} of subjects > 0.5)")
        print(f"  overnight decline: SWA {np.mean(swa_decl):.0%} vs density {np.mean(dens_decl):.0%}")
        verdict = ("SURVIVES (density tracks SWA)" if r.mean() > 0.5
                   else "WEAKENED (density only loosely tracks SWA)" if r.mean() > 0.3
                   else "FAILS (density does not track SWA)")
        print(f"  VERDICT: {verdict}\n")


# ---------------------------------------------------------------------------
# Control D — subject-level split audit for the age model
# ---------------------------------------------------------------------------
def control_d():
    from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
    from train import set_seed, fit, predict, _t
    from model import make_model
    from age_prediction import get_ages

    print(f"=== CONTROL D: subject-level split audit (seed={SEED_USED}) ===")
    print("Sleep-EDF has ~2 nights/subject. If a subject's two nights span train/test the")
    print("model can memorize the subject, inflating age r. Compare recording-level vs")
    print("subject-grouped CV.\n")

    d = np.load(PATHS["harmonized"] / "sleep_edf.npz", allow_pickle=True)
    X = d["X"].astype("float32")
    ids = [str(i) for i in d["ids"]]
    age = get_ages(ids)
    groups = np.array([i[3:5] for i in ids])           # subject number (SC4ssN -> ss)

    uniq = np.unique(groups)
    per = np.array([(groups == g).sum() for g in uniq])
    print(f"  {len(ids)} recordings from {len(uniq)} unique subjects "
          f"({(per == 2).sum()} with 2 nights, {(per == 1).sum()} with 1).")

    mu, sd = age.mean(), age.std()
    y_reg = ((age - mu) / sd).astype("float32")
    y_cls = (age >= np.median(age)).astype("float32")

    def run(splitter, grouped):
        pred = np.zeros(len(age))
        leak = 0
        it = splitter.split(X, y_cls, groups) if grouped else splitter.split(X, y_cls)
        for fold, (tr, te) in enumerate(it):
            if grouped:
                leak += len(set(groups[tr]) & set(groups[te]))
            tr2, va = train_test_split(tr, test_size=0.2, random_state=SEED)
            set_seed(SEED + fold)
            m = make_model(X.shape[2])
            m, _ = fit(m, _t(X[tr2]), _t(y_cls[tr2]), _t(y_reg[tr2]),
                       val=(_t(X[va]), _t(y_cls[va]), _t(y_reg[va])), epochs=120, reg_weight=1.0)
            _, rate, _ = predict(m, _t(X[te]))
            pred[te] = rate * sd + mu
        return stats.pearsonr(age, pred)[0], leak

    r_rec, _ = run(StratifiedKFold(5, shuffle=True, random_state=SEED), grouped=False)
    r_grp, leak = run(GroupKFold(5), grouped=True)

    print(f"  recording-level CV (original, potentially leaky): r = {r_rec:.3f}")
    print(f"  subject-grouped CV (no subject in train+test):     r = {r_grp:.3f}")
    print(f"  subject overlap across grouped folds: {leak} (must be 0)")
    drop = r_rec - r_grp
    if leak == 0 and drop < 0.07:
        v = f"SURVIVES (r barely changes: {r_rec:.2f}->{r_grp:.2f}, no leakage)"
    elif r_grp > 0.3:
        v = f"WEAKENED (r drops {r_rec:.2f}->{r_grp:.2f} but still real)"
    else:
        v = f"FAILS (r collapses {r_rec:.2f}->{r_grp:.2f} -> the age finding was leakage)"
    print(f"  VERDICT: {v}\n")


CONTROLS = {"A": control_a, "B": control_b, "C": control_c, "D": control_d}

if __name__ == "__main__":
    for key in (sys.argv[1:] or ["A"]):
        CONTROLS[key]()
