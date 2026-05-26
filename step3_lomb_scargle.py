"""
STEP 3 — Lomb-Scargle multiband + phase-fold
Input:  lc_strict/lc_<id>.csv
Output: folded_strict/folded_<id>.npy  shape (6, 100)  bands=[u,g,r,i,z,y]
        periods/periods_strict.csv     diaObjectId, period_days, ls_power, bands_used, n_points
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

BASE        = "/home/monti/LSST/VariAE"
LC_DIR      = os.path.join(BASE, "lc_strict")
FOLDED_DIR  = os.path.join(BASE, "folded_strict")
PERIODS_DIR = os.path.join(BASE, "periods")
PERIODS_CSV = os.path.join(PERIODS_DIR, "periods_strict.csv")

BANDS        = ["u", "g", "r", "i", "z", "y"]
BAND_IDX     = {b: i for i, b in enumerate(BANDS)}
N_PHASE      = 100
FREQ_MIN     = 1.0 / 100.0   # day⁻¹
FREQ_MAX     = 24.0           # day⁻¹
N_FREQ       = 100_000
PHASE_GRID   = np.linspace(0, 1, N_PHASE, endpoint=False)


def phase_fold_band(times, flux, flux_err, period):
    """Bin flux onto uniform phase grid, weighted mean. Returns (N_PHASE,) or None."""
    phase = (times % period) / period
    folded = np.full(N_PHASE, np.nan)
    for j in range(N_PHASE):
        lo = PHASE_GRID[j]
        hi = lo + 1.0 / N_PHASE
        mask = (phase >= lo) & (phase < hi)
        if mask.sum() == 0:
            continue
        w = 1.0 / flux_err[mask] ** 2
        folded[j] = np.sum(w * flux[mask]) / np.sum(w)
    return folded


def normalize_band(arr):
    """Zero mean, unit std; returns NaN array if not enough valid points."""
    valid = arr[~np.isnan(arr)]
    if len(valid) < 3:
        return np.full(N_PHASE, np.nan)
    mu, sigma = np.nanmean(arr), np.nanstd(arr)
    if sigma < 1e-10:
        return np.full(N_PHASE, np.nan)
    return (arr - mu) / sigma


def process_one(fpath):
    obj_id = os.path.basename(fpath).replace("lc_", "").replace(".csv", "")
    out_npy = os.path.join(FOLDED_DIR, f"folded_{obj_id}.npy")

    if os.path.exists(out_npy):
        # resume: period cannot be recovered from the existing .npy file;
        # caller skips writing to CSV for already-processed objects
        return None

    try:
        df = pd.read_csv(fpath, usecols=["band", "psfFlux", "psfFluxErr", "expMidptMJD"])
        df = df.dropna(subset=["psfFlux", "psfFluxErr", "expMidptMJD"])
        df = df[df["psfFluxErr"] > 0]
    except Exception:
        return None

    if len(df) < 10:
        return None

    # Combined multiband LS: sum normalized powers
    times_all, flux_all, err_all = [], [], []
    bands_used = []
    for b in BANDS:
        sub = df[df["band"] == b]
        if len(sub) < 3:
            continue
        bands_used.append(b)
        t = sub["expMidptMJD"].values
        f = sub["psfFlux"].values
        e = sub["psfFluxErr"].values
        # zero-mean per band before combining
        f = f - np.mean(f)
        times_all.append(t)
        flux_all.append(f)
        err_all.append(e)

    if not bands_used:
        return None

    t_cat = np.concatenate(times_all)
    f_cat = np.concatenate(flux_all)
    e_cat = np.concatenate(err_all)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls = LombScargle(t_cat, f_cat, e_cat, normalization="standard")
        freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)
        power = ls.power(freqs)

    best_idx  = np.argmax(power)
    best_freq = freqs[best_idx]
    best_pow  = float(power[best_idx])
    period    = 1.0 / best_freq

    # Phase-fold per band
    folded = np.full((6, N_PHASE), np.nan)
    for b in bands_used:
        sub = df[df["band"] == b]
        t = sub["expMidptMJD"].values
        f = sub["psfFlux"].values
        e = sub["psfFluxErr"].values
        raw = phase_fold_band(t, f, e, period)
        folded[BAND_IDX[b]] = normalize_band(raw)

    np.save(out_npy, folded.astype(np.float32))

    return {
        "diaObjectId": obj_id,
        "period_days": period,
        "ls_power": best_pow,
        "bands_used": ",".join(bands_used),
        "n_points": len(df),
    }


def main():
    os.makedirs(FOLDED_DIR, exist_ok=True)
    os.makedirs(PERIODS_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(LC_DIR, "lc_*.csv")))
    print(f"LC files found: {len(files)}")

    already_done = len(glob.glob(os.path.join(FOLDED_DIR, "folded_*.npy")))
    if already_done:
        print(f"Resume: {already_done} already processed, skipping.")

    n_workers = max(1, cpu_count() - 1)
    print(f"CPU workers: {n_workers}")

    results = []
    with Pool(n_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_one, files), total=len(files)):
            if res is not None:
                results.append(res)

    if results:
        df_out = pd.DataFrame(results)
        # merge with existing CSV if resuming
        if os.path.exists(PERIODS_CSV):
            existing = pd.read_csv(PERIODS_CSV, dtype={"diaObjectId": str})
            df_out = pd.concat([existing, df_out], ignore_index=True)
            df_out = df_out.drop_duplicates(subset="diaObjectId", keep="last")
        df_out.to_csv(PERIODS_CSV, index=False)

    total_folded = len(glob.glob(os.path.join(FOLDED_DIR, "folded_*.npy")))
    print(f"\nFolded saved: {total_folded}")
    print(f"Periods CSV:  {PERIODS_CSV}")

    if os.path.exists(PERIODS_CSV):
        df_p = pd.read_csv(PERIODS_CSV)
        print(f"\nls_power distribution:")
        for thr in [0.1, 0.2, 0.3, 0.5, 0.7]:
            n = (df_p["ls_power"] >= thr).sum()
            print(f"  >= {thr}: {n} ({n/len(df_p)*100:.1f}%)")


if __name__ == "__main__":
    main()
