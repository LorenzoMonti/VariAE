"""
STEP 2 — Crossmatch analysis
Analyses Gaia DR3 classes and per-band LC coverage.
Output: crossmatch/step2_analysis_log.txt
"""

import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

BASE = "/home/monti/LSST/VariAE"
LC_DIR = os.path.join(BASE, "lc_strict")
LC_GENTLE_DIR = os.path.join(BASE, "lc_gentle")
XMATCH_DIR = os.path.join(BASE, "crossmatch")
OUT_LOG = os.path.join(XMATCH_DIR, "step2_analysis_log.txt")
CATALOG_DIR = os.path.join(BASE, "catalogs")

CLASSIFIER_CSV = os.path.join(XMATCH_DIR, "catalogue_lsst_classifier.csv")
SOS_CSV = os.path.join(XMATCH_DIR, "catalogue_lsst_sos.csv")
VARIABLE_CATALOG = os.path.join(CATALOG_DIR, "variable_objects_catalog.csv")

SCORE_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
BANDS = ["u", "g", "r", "i", "z", "y"]


def load_data():
    clf = pd.read_csv(CLASSIFIER_CSV)
    sos = pd.read_csv(SOS_CSV)
    var = pd.read_csv(VARIABLE_CATALOG)
    return clf, sos, var


def analyze_classifier(clf, lines):
    lines.append("=" * 60)
    lines.append("1. GAIA CLASSIFIER (catalogue_lsst_classifier.csv)")
    lines.append("=" * 60)
    lines.append(f"  Total crossmatched objects: {len(clf)}")
    lines.append(f"  Unique lsst_id: {clf['lsst_id'].nunique()}")
    lines.append("")

    lines.append("  Class distribution (best_class_name):")
    class_counts = clf['best_class_name'].value_counts()
    for cls, cnt in class_counts.items():
        lines.append(f"    {cls:20s}: {cnt:4d}  ({cnt/len(clf)*100:.1f}%)")
    lines.append("")

    lines.append("  Mean score per class:")
    score_stats = clf.groupby('best_class_name')['best_class_score'].agg(['mean', 'median', 'min', 'max'])
    for cls, row in score_stats.iterrows():
        lines.append(f"    {cls:20s}: mean={row['mean']:.3f}  median={row['median']:.3f}  "
                     f"min={row['min']:.3f}  max={row['max']:.3f}")
    lines.append("")

    lines.append("  Objects per score threshold (best_class_score >= threshold):")
    for thr in SCORE_THRESHOLDS:
        filtered = clf[clf['best_class_score'] >= thr]
        lines.append(f"    >= {thr:.1f}: {len(filtered):4d} objects")
        per_class = filtered['best_class_name'].value_counts()
        for cls, cnt in per_class.items():
            lines.append(f"           {cls:20s}: {cnt}")
        lines.append("")

    return clf


def analyze_sos(sos, lines):
    lines.append("=" * 60)
    lines.append("2. GAIA SOS (catalogue_lsst_sos.csv)")
    lines.append("=" * 60)
    lines.append(f"  Total objects: {len(sos)}")
    lines.append(f"  Unique lsst_id: {sos['lsst_id'].nunique()}")
    lines.append("")

    sos_type_cols = [c for c in sos.columns if c.startswith("in_vari_") and c != "in_vari_classification_result"]
    lines.append("  Count by SOS type:")
    for col in sos_type_cols:
        count = sos[col].astype(str).str.lower().eq("true").sum()
        label = col.replace("in_vari_", "")
        lines.append(f"    {label:30s}: {count:4d}  ({count/len(sos)*100:.1f}%)")
    lines.append("")

    # Multi-type objects
    sos_bool = sos[sos_type_cols].apply(lambda x: x.astype(str).str.lower() == "true")
    multi = sos_bool.sum(axis=1)
    lines.append("  Objects with N SOS types assigned:")
    for n, cnt in multi.value_counts().sort_index().items():
        lines.append(f"    N={n}: {cnt}")
    lines.append("")

    return sos


def analyze_overlap(clf, sos, var, lines):
    lines.append("=" * 60)
    lines.append("3. CATALOGUE OVERLAP")
    lines.append("=" * 60)

    clf_ids = set(clf['lsst_id'].astype(str))
    sos_ids = set(sos['lsst_id'].astype(str))
    var_ids = set(var['diaObjectId'].astype(str))

    in_both = clf_ids & sos_ids
    clf_only = clf_ids - sos_ids
    sos_only = sos_ids - clf_ids
    in_var_and_clf = clf_ids & var_ids
    in_var_and_sos = sos_ids & var_ids

    lines.append(f"  variable_objects_catalog.csv: {len(var_ids)} objects")
    lines.append(f"  catalogue_lsst_classifier:    {len(clf_ids)} objects")
    lines.append(f"  catalogue_lsst_sos:           {len(sos_ids)} objects")
    lines.append(f"  In both classifier+SOS:       {len(in_both)}")
    lines.append(f"  Only in classifier:           {len(clf_only)}")
    lines.append(f"  Only in SOS:                  {len(sos_only)}")
    lines.append(f"  var_catalog ∩ classifier:     {len(in_var_and_clf)}")
    lines.append(f"  var_catalog ∩ SOS:            {len(in_var_and_sos)}")
    lines.append("")

    # Labels available for seed selection
    merged = clf.merge(
        sos[['lsst_id'] + [c for c in sos.columns if c.startswith("in_vari_")]],
        on='lsst_id', how='left'
    )
    seed_candidates = clf[clf['best_class_score'] >= 0.5]
    lines.append(f"  Seed candidates (score>=0.5): {len(seed_candidates)}")
    per_class = seed_candidates['best_class_name'].value_counts()
    for cls, cnt in per_class.items():
        lines.append(f"    {cls:20s}: {cnt}")
    lines.append("")


def analyze_gentle(lines):
    lines.append("=" * 60)
    lines.append("4b. DATASET lc_gentle (51k)")
    lines.append("=" * 60)
    import sys
    gentle_files = glob.glob(os.path.join(LC_GENTLE_DIR, "lc_*.csv"))
    lines.append(f"  LC files found (lc_gentle): {len(gentle_files)}")
    strict_ids = {os.path.basename(f) for f in glob.glob(os.path.join(LC_DIR, "lc_*.csv"))}
    gentle_ids = {os.path.basename(f) for f in gentle_files}
    only_gentle = gentle_ids - strict_ids
    in_both = gentle_ids & strict_ids
    lines.append(f"  In lc_strict:          {len(strict_ids)}")
    lines.append(f"  In both:               {len(in_both)}")
    lines.append(f"  Only in lc_gentle:     {len(only_gentle)}")
    lines.append("")


def analyze_bands(var_ids, lines):
    lines.append("=" * 60)
    lines.append("4. BAND COVERAGE — lc_strict (13505 objects)")
    lines.append("=" * 60)
    lines.append("  Analysing all LC files (may take ~1-2 min)...")
    lines.append("")

    files = glob.glob(os.path.join(LC_DIR, "lc_*.csv"))
    lines.append(f"  LC files found: {len(files)}")

    band_presence = {b: 0 for b in BANDS}
    obs_per_band = {b: [] for b in BANDS}
    total_obs_per_obj = []

    for fpath in files:
        try:
            df = pd.read_csv(fpath, usecols=['band'])
        except Exception:
            continue
        n_total = len(df)
        total_obs_per_obj.append(n_total)
        bc = df['band'].value_counts()
        for b in BANDS:
            n = bc.get(b, 0)
            if n > 0:
                band_presence[b] += 1
                obs_per_band[b].append(n)

    lines.append("")
    lines.append("  Band coverage (% objects with at least 1 observation):")
    for b in BANDS:
        pct = band_presence[b] / len(files) * 100
        counts = obs_per_band[b]
        if counts:
            med = np.median(counts)
            p25 = np.percentile(counts, 25)
            p75 = np.percentile(counts, 75)
            lines.append(f"    {b}: {band_presence[b]:5d}/{len(files)} ({pct:5.1f}%)  "
                         f"obs per obj: median={med:.0f}  [p25={p25:.0f}, p75={p75:.0f}]")
        else:
            lines.append(f"    {b}: 0 objects")

    lines.append("")
    total_arr = np.array(total_obs_per_obj)
    lines.append(f"  Total observations per object:")
    lines.append(f"    min={total_arr.min()}  p25={np.percentile(total_arr,25):.0f}  "
                 f"median={np.median(total_arr):.0f}  p75={np.percentile(total_arr,75):.0f}  "
                 f"max={total_arr.max()}")
    lines.append("")

    # Band recommendation for VAE/LS
    lines.append("  Band recommendation for VAE/LS:")
    ranked = sorted(BANDS, key=lambda b: band_presence[b], reverse=True)
    for b in ranked:
        pct = band_presence[b] / len(files) * 100
        lines.append(f"    {b}: {pct:.1f}% coverage")
    primary = [b for b in ranked if band_presence[b] / len(files) >= 0.5]
    lines.append(f"  Primary bands (>=50% coverage): {primary}")
    lines.append("")


def analyze_xmatch_band_coverage(clf, lines):
    """Check band coverage specifically for the 725 crossmatched objects."""
    lines.append("=" * 60)
    lines.append("5. BAND COVERAGE — 725 CROSSMATCHED OBJECTS")
    lines.append("=" * 60)

    xmatch_ids = clf['lsst_id'].astype(str).tolist()
    band_presence = {b: 0 for b in BANDS}
    obs_per_band_per_class = {}

    found = 0
    missing = 0
    for lid in xmatch_ids:
        fpath = os.path.join(LC_DIR, f"lc_{lid}.csv")
        if not os.path.exists(fpath):
            missing += 1
            continue
        found += 1
        try:
            df = pd.read_csv(fpath, usecols=['band'])
        except Exception:
            continue
        bc = df['band'].value_counts()
        for b in BANDS:
            if bc.get(b, 0) > 0:
                band_presence[b] += 1

    lines.append(f"  LC files found for crossmatch: {found}/{len(xmatch_ids)}  (missing: {missing})")
    lines.append("")
    lines.append("  Band coverage for the 725 crossmatched objects:")
    for b in BANDS:
        pct = band_presence[b] / max(found, 1) * 100
        lines.append(f"    {b}: {band_presence[b]:3d}/{found} ({pct:5.1f}%)")
    lines.append("")


def main():
    lines = []
    lines.append("VariAE — STEP 2: CROSSMATCH ANALYSIS LOG")
    lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    clf, sos, var = load_data()
    analyze_classifier(clf, lines)
    analyze_sos(sos, lines)
    analyze_overlap(clf, sos, var, lines)
    analyze_gentle(lines)
    analyze_bands(set(var['diaObjectId'].astype(str)), lines)
    analyze_xmatch_band_coverage(clf, lines)

    lines.append("=" * 60)
    lines.append("END LOG")
    lines.append("=" * 60)

    log_text = "\n".join(lines)
    print(log_text)
    with open(OUT_LOG, "w") as f:
        f.write(log_text)
    print(f"\n[Log saved to {OUT_LOG}]")


if __name__ == "__main__":
    main()
