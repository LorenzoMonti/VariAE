"""
STEP 5b — Multiclass classifier (10 classes) with k-NN and linear SVM.
With only ~14 samples/class in training, non-parametric models generalise
better than an MLP, which tends to memorise the training set.

Train:      labelled in lc_strict       (score>=0.5, ~137 objects)
Validation: labelled in lc_gentle_extra (score>=0.5, ~185 objects)
Inference:  all 13505 lc_strict objects

Models:
  k-NN  — Euclidean distance in latent space, no training
  SVM   — Linear max-margin, class_weight='balanced'

RR calibration: threshold on prob_RR (SVM) or k-NN vote fraction that
maximises F1(RR) on the validation set.

Output: results/clf_predictions.csv    diaObjectId, pred_knn, pred_svm,
                                        prob_rr_knn, prob_rr_svm
        results/clf_eval.txt           validation metrics for both models
        results/clf_umap.png           UMAP coloured by class (k-NN and SVM)
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_curve, roc_auc_score,
                              average_precision_score)

BASE        = "/home/monti/LSST/VariAE"
LATENT_DIR  = os.path.join(BASE, "latent")
RESULTS_DIR = os.path.join(BASE, "results")
XMATCH_CSV  = os.path.join(BASE, "crossmatch", "catalogue_lsst_classifier.csv")

LATENT_IDX_STRICT = "ssvae_latent_index.csv"
LATENT_IDX_EXTRA  = "ssvae_gentle_extra_index.csv"

SCORE_THR       = 0.5
EXCLUDE_CLASSES = {"SN"}
K_NEIGHBORS     = 7
RANDOM_SEED     = 42


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    Z_strict = np.load(os.path.join(LATENT_DIR, "ssvae_latent_vectors.npy"))
    Z_extra  = np.load(os.path.join(LATENT_DIR, "ssvae_gentle_extra.npy"))
    idx_strict = pd.read_csv(os.path.join(LATENT_DIR, LATENT_IDX_STRICT), dtype={"diaObjectId": str})
    idx_extra  = pd.read_csv(os.path.join(LATENT_DIR, LATENT_IDX_EXTRA),  dtype={"diaObjectId": str})

    clf       = pd.read_csv(XMATCH_CSV, dtype={"lsst_id": str})
    label_map = dict(zip(clf["lsst_id"], clf["best_class_name"]))
    score_map = dict(zip(clf["lsst_id"], clf["best_class_score"]))

    all_ids = list(idx_strict["diaObjectId"]) + list(idx_extra["diaObjectId"])
    Z_all   = np.vstack([Z_strict, Z_extra])
    source  = np.array(["strict"] * len(idx_strict) + ["extra"] * len(idx_extra))
    labels  = np.array([label_map.get(i) for i in all_ids], dtype=object)
    scores  = np.array([score_map.get(i, 0.0) for i in all_ids], dtype=float)

    return Z_all, all_ids, labels, scores, source


def build_split(labels, scores, source):
    def idx(src):
        return np.where([
            l is not None and l not in EXCLUDE_CLASSES and s >= SCORE_THR and so == src
            for l, s, so in zip(labels, scores, source)
        ])[0]

    train_idx   = idx("strict")
    holdout_idx = idx("extra")
    classes     = sorted(set(labels[train_idx]) | set(labels[holdout_idx]))

    print(f"Train  (lc_strict,       score>={SCORE_THR}): {len(train_idx)}")
    print(f"Val    (lc_gentle_extra, score>={SCORE_THR}): {len(holdout_idx)}")
    print(f"Classes: {classes}\n")
    for c in classes:
        n_tr = (labels[train_idx] == c).sum()
        n_ho = (labels[holdout_idx] == c).sum()
        print(f"  {c:22s}: train={n_tr:3d}  val={n_ho:3d}")
    return train_idx, holdout_idx, classes


# ── RR threshold calibration ──────────────────────────────────────────────────

def calibrate_rr(prob_rr_val, y_val_labels, model_name):
    y_bin = (y_val_labels == "RR").astype(int)
    if y_bin.sum() == 0:
        print(f"  [{model_name}] No RR in val set, default threshold=0.5")
        return 0.5

    auc_roc = roc_auc_score(y_bin, prob_rr_val)
    auc_pr  = average_precision_score(y_bin, prob_rr_val)

    precision, recall, thresholds = precision_recall_curve(y_bin, prob_rr_val)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best_i   = np.argmax(f1[:-1])
    best_thr = float(thresholds[best_i])

    print(f"\n[{model_name}] RR calibration on validation set:")
    print(f"  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}")
    print(f"  Threshold={best_thr:.3f}  P={precision[best_i]:.3f}  "
          f"R={recall[best_i]:.3f}  F1={f1[best_i]:.3f}")
    return best_thr, auc_roc, auc_pr, precision[best_i], recall[best_i], f1[best_i]


def apply_rr_threshold(pred, prob_rr, threshold):
    cal = pred.copy()
    cal[(prob_rr >= threshold) & (pred != "RR")] = "RR"
    return cal


# ── UMAP ──────────────────────────────────────────────────────────────────────

def plot_umap(Z, pred_knn, pred_svm, source, out_path):
    import umap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strict_mask = source == "strict"
    Z_plot = Z[strict_mask]

    print("UMAP fit...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="euclidean", random_state=RANDOM_SEED, verbose=False)
    emb = reducer.fit_transform(Z_plot)

    CLASS_COLORS = {
        "RR": "#e63946", "CEP": "#f4a261", "ECL": "#2a9d8f", "LPV": "#e9c46a",
        "AGN": "#8338ec", "YSO": "#06d6a0", "SOLAR_LIKE": "#118ab2",
        "RS": "#ffd166", "S": "#ff9f1c", "DSCT|GDOR|SXPHE": "#cbf3f0", "CV": "#ff006e",
    }

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, preds, title in [
        (axes[0], pred_knn[strict_mask], f"k-NN (k={K_NEIGHBORS})"),
        (axes[1], pred_svm[strict_mask], "Linear SVM (Platt calibrated)"),
    ]:
        for cls in sorted(set(preds)):
            idx  = preds == cls
            col  = CLASS_COLORS.get(cls, "#888888")
            ax.scatter(emb[idx, 0], emb[idx, 1], s=2, c=col, alpha=0.6,
                       rasterized=True, label=f"{cls} ({idx.sum()})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(loc="upper right", fontsize=6, markerscale=3, framealpha=0.85)
        ax.set_aspect("equal", "datalim")

    fig.suptitle("STEP 5b — Multiclass classifier (k-NN vs SVM)\n"
                 "(lc_strict, 13505 objects, calibrated RR threshold)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== VariAE STEP 5b — Multiclass k-NN + SVM ===\n")

    Z, all_ids, labels, scores, source = load_data()
    print(f"Latent: {Z.shape}\n")

    train_idx, holdout_idx, classes = build_split(labels, scores, source)

    X_tr  = Z[train_idx]
    y_tr  = labels[train_idx]
    X_val = Z[holdout_idx]
    y_val = labels[holdout_idx]

    # ── k-NN ─────────────────────────────────────────────────────────────────
    print(f"\n--- k-NN (k={K_NEIGHBORS}) ---")
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric="euclidean", n_jobs=-1)
    knn.fit(X_tr, y_tr)

    pred_val_knn   = knn.predict(X_val)
    proba_val_knn  = knn.predict_proba(X_val)
    classes_knn    = list(knn.classes_)
    rr_idx_knn     = classes_knn.index("RR") if "RR" in classes_knn else None
    prob_rr_knn    = proba_val_knn[:, rr_idx_knn] if rr_idx_knn is not None else np.zeros(len(X_val))

    thr_knn, auc_roc_knn, auc_pr_knn, p_knn, r_knn, f1_knn = calibrate_rr(prob_rr_knn, y_val, "k-NN")
    pred_val_knn_cal = apply_rr_threshold(pred_val_knn, prob_rr_knn, thr_knn)

    print(f"\nClassification report k-NN (val, calibrated RR):")
    print(classification_report(y_val, pred_val_knn_cal, zero_division=0))

    # ── SVM ──────────────────────────────────────────────────────────────────
    print("\n--- Linear SVM (Platt calibrated) ---")
    # Drop classes with a single sample: CV cannot be stratified with cv>=2
    tr_counts   = Counter(y_tr)
    valid_cls   = {c for c, n in tr_counts.items() if n >= 2}
    dropped     = {c for c, n in tr_counts.items() if n < 2}
    if dropped:
        print(f"  Classes removed from SVM training (n<2): {dropped}")
    mask_svm    = np.array([c in valid_cls for c in y_tr])
    X_tr_svm, y_tr_svm = X_tr[mask_svm], y_tr[mask_svm]
    cv_folds    = max(2, min(5, min(Counter(y_tr_svm).values())))
    svm_base    = LinearSVC(class_weight="balanced", max_iter=5000, random_state=RANDOM_SEED)
    svm         = CalibratedClassifierCV(svm_base, cv=cv_folds)
    svm.fit(X_tr_svm, y_tr_svm)

    pred_val_svm   = svm.predict(X_val)
    proba_val_svm  = svm.predict_proba(X_val)
    classes_svm    = list(svm.classes_)
    rr_idx_svm     = classes_svm.index("RR") if "RR" in classes_svm else None
    prob_rr_svm    = proba_val_svm[:, rr_idx_svm] if rr_idx_svm is not None else np.zeros(len(X_val))

    thr_svm, auc_roc_svm, auc_pr_svm, p_svm, r_svm, f1_svm = calibrate_rr(prob_rr_svm, y_val, "SVM")
    pred_val_svm_cal = apply_rr_threshold(pred_val_svm, prob_rr_svm, thr_svm)

    print(f"\nClassification report SVM (val, calibrated RR):")
    print(classification_report(y_val, pred_val_svm_cal, zero_division=0))

    # ── Save eval ─────────────────────────────────────────────────────────────
    lines = [
        "VariAE STEP 5b — Multiclass k-NN + SVM", "=" * 50, "",
        f"Train: {len(train_idx)} (strict, score>={SCORE_THR})  |  "
        f"Val: {len(holdout_idx)} (gentle_extra)",
        f"Classes: {classes}", "",
        f"k-NN (k={K_NEIGHBORS}):",
        f"  RR AUC-ROC={auc_roc_knn:.4f}  AUC-PR={auc_pr_knn:.4f}  "
        f"threshold={thr_knn:.3f}  P={p_knn:.3f}  R={r_knn:.3f}  F1={f1_knn:.3f}",
        "",
        "Classification report k-NN (val, calibrated):",
        classification_report(y_val, pred_val_knn_cal, zero_division=0), "",
        f"Linear SVM:",
        f"  RR AUC-ROC={auc_roc_svm:.4f}  AUC-PR={auc_pr_svm:.4f}  "
        f"threshold={thr_svm:.3f}  P={p_svm:.3f}  R={r_svm:.3f}  F1={f1_svm:.3f}",
        "",
        "Classification report SVM (val, calibrated):",
        classification_report(y_val, pred_val_svm_cal, zero_division=0),
    ]
    eval_text = "\n".join(lines)
    print("\n" + eval_text)
    with open(os.path.join(RESULTS_DIR, "clf_eval.txt"), "w") as f:
        f.write(eval_text)

    # ── Inference on all 13505 lc_strict ──────────────────────────────────────
    strict_mask = source == "strict"
    ids_strict  = [i for i, s in zip(all_ids, source) if s == "strict"]
    Z_strict    = Z[strict_mask]

    proba_knn_all   = knn.predict_proba(Z_strict)
    pred_knn_all    = np.array(classes_knn)[proba_knn_all.argmax(axis=1)]
    prob_rr_knn_all = proba_knn_all[:, rr_idx_knn] if rr_idx_knn is not None else np.zeros(len(Z_strict))
    pred_knn_cal    = apply_rr_threshold(pred_knn_all, prob_rr_knn_all, thr_knn)

    proba_svm_all   = svm.predict_proba(Z_strict)
    pred_svm_all    = np.array(classes_svm)[proba_svm_all.argmax(axis=1)]
    prob_rr_svm_all = proba_svm_all[:, rr_idx_svm] if rr_idx_svm is not None else np.zeros(len(Z_strict))
    pred_svm_cal    = apply_rr_threshold(pred_svm_all, prob_rr_svm_all, thr_svm)

    print("\nPrediction distribution lc_strict — k-NN (calibrated):")
    for cls, cnt in sorted(Counter(pred_knn_cal).items(), key=lambda x: -x[1]):
        print(f"  {cls:22s}: {cnt:5d}")

    print("\nPrediction distribution lc_strict — SVM (calibrated):")
    for cls, cnt in sorted(Counter(pred_svm_cal).items(), key=lambda x: -x[1]):
        print(f"  {cls:22s}: {cnt:5d}")

    # Agreement between k-NN and SVM predictions
    agree   = (pred_knn_cal == pred_svm_cal)
    print(f"\nk-NN / SVM agreement: {agree.sum()}/{len(agree)} ({agree.mean()*100:.1f}%)")
    rr_agree = ((pred_knn_cal == "RR") & (pred_svm_cal == "RR"))
    print(f"RR consensus (both agree): {rr_agree.sum()}")

    df_out = pd.DataFrame({
        "diaObjectId": ids_strict,
        "pred_knn":    pred_knn_cal,
        "pred_svm":    pred_svm_cal,
        "prob_rr_knn": prob_rr_knn_all,
        "prob_rr_svm": prob_rr_svm_all,
        "rr_consensus": rr_agree.astype(int),
    })
    df_out.to_csv(os.path.join(RESULTS_DIR, "clf_predictions.csv"), index=False)
    print(f"\nPredictions → {RESULTS_DIR}/clf_predictions.csv")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    pred_all_knn = np.empty(len(Z), dtype=object)
    pred_all_svm = np.empty(len(Z), dtype=object)
    pred_all_knn[strict_mask] = pred_knn_cal
    pred_all_svm[strict_mask] = pred_svm_cal
    pred_all_knn[~strict_mask] = "extra"
    pred_all_svm[~strict_mask] = "extra"

    plot_umap(Z, pred_all_knn, pred_all_svm, source,
              os.path.join(RESULTS_DIR, "clf_umap.png"))


if __name__ == "__main__":
    main()
