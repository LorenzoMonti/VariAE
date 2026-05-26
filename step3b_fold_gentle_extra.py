"""
STEP 3b — Phase-fold LCs present in lc_gentle but not in lc_strict.
Processes only the objects crossmatched with Gaia (labels for step4).

Output:
  folded_gentle_extra/folded_<id>.npy   (6, 100) float32
  periods/periods_gentle_extra.csv      diaObjectId, period_days, ls_power
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from step3_lomb_scargle import BANDS, BAND_IDX, N_PHASE, phase_fold_band, normalize_band

BASE              = "/home/monti/LSST/VariAE"
LC_STRICT_DIR     = os.path.join(BASE, "lc_strict")
LC_GENTLE_DIR     = os.path.join(BASE, "lc_gentle")
FOLDED_EXTRA_DIR  = os.path.join(BASE, "folded_gentle_extra")
PERIODS_EXTRA_CSV = os.path.join(BASE, "periods", "periods_gentle_extra.csv")
XMATCH_CSV        = os.path.join(BASE, "crossmatch", "catalogue_lsst_classifier.csv")

FREQ_MIN = 1.0 / 100.0
FREQ_MAX = 24.0
N_FREQ   = 100_000


def get_extra_ids():
    strict_ids = {os.path.basename(f).replace("lc_", "").replace(".csv", "")
                  for f in glob.glob(os.path.join(LC_STRICT_DIR, "lc_*.csv"))}
    gentle_ids = {os.path.basename(f).replace("lc_", "").replace(".csv", "")
                  for f in glob.glob(os.path.join(LC_GENTLE_DIR, "lc_*.csv"))}
    return sorted(gentle_ids - strict_ids)


def process_one(obj_id):
    out_npy     = os.path.join(FOLDED_EXTRA_DIR, f"folded_{obj_id}.npy")
    fold_exists = os.path.exists(out_npy)

    fpath = os.path.join(LC_GENTLE_DIR, f"lc_{obj_id}.csv")
    try:
        df = pd.read_csv(fpath, usecols=["band", "psfFlux", "psfFluxErr", "expMidptMJD"])
        df = df.dropna(subset=["psfFlux", "psfFluxErr", "expMidptMJD"])
        df = df[df["psfFluxErr"] > 0]
    except Exception:
        return None

    if len(df) < 5:
        return None

    times_all, flux_all, err_all, bands_used = [], [], [], []
    for b in BANDS:
        sub = df[df["band"] == b]
        if len(sub) < 3:
            continue
        bands_used.append(b)
        t = sub["expMidptMJD"].values
        f = sub["psfFlux"].values - sub["psfFlux"].mean()
        e = sub["psfFluxErr"].values
        times_all.append(t); flux_all.append(f); err_all.append(e)

    if not bands_used:
        return None

    t_cat = np.concatenate(times_all)
    f_cat = np.concatenate(flux_all)
    e_cat = np.concatenate(err_all)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls    = LombScargle(t_cat, f_cat, e_cat, normalization="standard")
        freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)
        power = ls.power(freqs)

    best_idx = np.argmax(power)
    period   = 1.0 / freqs[best_idx]
    ls_power = float(power[best_idx])

    if not fold_exists:
        folded = np.full((6, N_PHASE), np.nan)
        for b in bands_used:
            sub = df[df["band"] == b]
            raw = phase_fold_band(sub["expMidptMJD"].values,
                                  sub["psfFlux"].values,
                                  sub["psfFluxErr"].values, period)
            folded[BAND_IDX[b]] = normalize_band(raw)
        np.save(out_npy, folded.astype(np.float32))

    return {"diaObjectId": obj_id, "period_days": period,
            "ls_power": ls_power, "fold_new": not fold_exists}


def main():
    os.makedirs(FOLDED_EXTRA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PERIODS_EXTRA_CSV), exist_ok=True)

    extra_ids = get_extra_ids()
    print(f"Extra LCs (lc_gentle \\ lc_strict): {len(extra_ids)}")

    clf = pd.read_csv(XMATCH_CSV, dtype={"lsst_id": str})
    xmatch_extra = [i for i in extra_ids if i in set(clf["lsst_id"])]
    print(f"Of which crossmatched with Gaia: {len(xmatch_extra)}")
    print(f"Processing {len(xmatch_extra)} crossmatched objects (LS always, fold only if missing).\n")

    n_workers = max(1, cpu_count() - 1)
    results = []
    with Pool(n_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_one, xmatch_extra),
                        total=len(xmatch_extra)):
            if res:
                results.append(res)

    fold_new = sum(1 for r in results if r.get("fold_new"))
    print(f"\nNew folds: {fold_new}  |  Already existing: {len(results) - fold_new}")

    periods_df = pd.DataFrame([
        {"diaObjectId": r["diaObjectId"],
         "period_days": r["period_days"],
         "ls_power":    r["ls_power"]}
        for r in results
    ]).sort_values("diaObjectId")
    periods_df.to_csv(PERIODS_EXTRA_CSV, index=False)
    print(f"Periods saved: {PERIODS_EXTRA_CSV}  ({len(periods_df)} objects)")


if __name__ == "__main__":
    main()
