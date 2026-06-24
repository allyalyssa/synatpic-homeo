"""Stage 1 — download and extract the raw EEG datasets.

DREAMS (Zenodo) ships as a single .rar; we extract it with bsdtar (libarchive),
which reads RAR without any extra install. Sleep-EDF (PhysioNet) is a flat
directory of per-recording EDFs, each PSG paired with an expert hypnogram that
later stages actually use (unlike the original auto-staging pipeline).

Run:  python download.py            # both datasets
      python download.py dreams     # just one
      python download.py sleep_edf
"""
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import requests

from config import PATHS, DREAMS, SLEEP_EDF, RAR_TAR, ensure_dirs


def _stream(url, dest, chunk=1 << 20, quiet=False):
    """Download `url` to `dest`, skipping if already present. Atomic via .part."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk):
                f.write(block)
    tmp.rename(dest)
    if not quiet:
        print(f"  done: {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")


def download_dreams():
    """Fetch DatabasePatients.rar from Zenodo and extract the EDFs."""
    ensure_dirs()
    out = PATHS["dreams_raw"]
    rar = out / DREAMS["rar_key"]
    url = f"https://zenodo.org/api/records/{DREAMS['zenodo_record']}/files/{DREAMS['rar_key']}/content"
    print(f"DREAMS: downloading {DREAMS['rar_key']} ...")
    _stream(url, rar)

    print("DREAMS: extracting (bsdtar) ...")
    # bsdtar autodetects RAR; -C extracts into the raw dir.
    subprocess.run([RAR_TAR, "-xf", str(rar), "-C", str(out)], check=True)
    edfs = sorted(out.rglob("*.edf"))
    print(f"DREAMS: {len(edfs)} EDF files present (expected ~{DREAMS['n_expected']})")
    return edfs


def _sleep_edf_listing():
    """Return all filenames in the Sleep-EDF sleep-cassette directory."""
    html = urllib.request.urlopen(SLEEP_EDF["base_url"], timeout=60).read().decode()
    return sorted(set(re.findall(r"SC\d+[A-Z0-9]*-(?:PSG|Hypnogram)\.edf", html)))


def download_sleep_edf(workers=8):
    """Download every SC recording as a (PSG, hypnogram) pair, in parallel.

    Pairing is by the first 6 chars (SC4ssN = subject+night), which is robust to
    the varying 7th 'device' letter in the PhysioNet naming scheme. PhysioNet
    throttles each connection to ~0.15 MB/s, so we fan out across `workers`
    connections; the job is idempotent (skips finished files) and can be re-run
    to resume.
    """
    ensure_dirs()
    out = PATHS["sleep_edf_raw"]
    files = _sleep_edf_listing()
    psgs = [f for f in files if f.endswith("-PSG.edf")]
    hyps = {f[:6]: f for f in files if f.endswith("-Hypnogram.edf")}

    pairs = [(p, hyps[p[:6]]) for p in psgs if p[:6] in hyps]
    if SLEEP_EDF["max_subjects"]:
        pairs = pairs[: SLEEP_EDF["max_subjects"]]

    jobs = []
    for psg, hyp in pairs:
        jobs += [(SLEEP_EDF["base_url"] + psg, out / psg),
                 (SLEEP_EDF["base_url"] + hyp, out / hyp)]
    print(f"Sleep-EDF: {len(pairs)} pairs ({len(jobs)} files) via {workers} workers")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda j: _stream(*j), jobs))
    print(f"Sleep-EDF: {len(list(out.glob('*-PSG.edf')))} PSG files on disk")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "dreams"):
        download_dreams()
    if which in ("all", "sleep_edf"):
        download_sleep_edf()
