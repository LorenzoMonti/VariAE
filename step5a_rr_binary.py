"""
STEP 5a — Binary classifier RR Lyrae vs rest on the SS-VAE latent space.
Uses ALL 725 labelled objects (no score threshold) to maximise RR training samples.

Train:      labelled in lc_strict       (~295 objects, ~22 RR)
Validation: labelled in lc_gentle_extra (~430 objects, ~31 RR)
Inference:  all 13505 lc_strict objects

Output: results/rr_predictions.csv    diaObjectId, prob_rr, pred_rr (0/1)
        results/rr_eval.txt           AUC-ROC, AUC-PR, F1, threshold
        results/rr_umap.png           UMAP with prob_rr as heatmap
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, classification_report)

BASE        = "/home/monti/LSST/VariAE"
LATENT_DIR  = os.path.join(BASE, "latent")
RESULTS_DIR = os.path.join(BASE, "results")
XMATCH_CSV  = os.path.join(BASE, "crossmatch", "catalogue_lsst_classifier.csv")

LATENT_IDX_STRICT = "ssvae_latent_index.csv"
LATENT_IDX_EXTRA  = "ssvae_gentle_extra_index.csv"

# Use all labelled objects (no score threshold) for more RR training samples
SCORE_THR   = 0.0
RANDOM_SEED = 42

LATENT_DIM   = 32
HIDDEN       = [128, 64]
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 400
PATIENCE     = 50
BATCH_SIZE   = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    Z_strict = np.load(os.path.join(LATENT_DIR, "ssvae_latent_vectors.npy"))
    Z_extra  = np.load(os.path.join(LATENT_DIR, "ssvae_gentle_extra.npy"))
    idx_strict = pd.read_csv(os.path.join(LATENT_DIR, LATENT_IDX_STRICT), dtype={"diaObjectId": str})
    idx_extra  = pd.read_csv(os.path.join(LATENT_DIR, LATENT_IDX_EXTRA),  dtype={"diaObjectId": str})

    clf = pd.read_csv(XMATCH_CSV, dtype={"lsst_id": str})
    label_map = dict(zip(clf["lsst_id"], clf["best_class_name"]))
    score_map = dict(zip(clf["lsst_id"], clf["best_class_score"]))

    all_ids = list(idx_strict["diaObjectId"]) + list(idx_extra["diaObjectId"])
    Z_all   = np.vstack([Z_strict, Z_extra])
    source  = np.array(["strict"] * len(idx_strict) + ["extra"] * len(idx_extra))
    labels  = np.array([label_map.get(i) for i in all_ids], dtype=object)
    scores  = np.array([score_map.get(i, 0.0) for i in all_ids], dtype=float)

    return Z_all, all_ids, labels, scores, source


def build_split(labels, scores, source):
    def rr_idx(src):
        # All labelled objects (score >= SCORE_THR = 0.0)
        return np.where([
            l is not None and s >= SCORE_THR and so == src
            for l, s, so in zip(labels, scores, source)
        ])[0]

    train_idx   = rr_idx("strict")
    holdout_idx = rr_idx("extra")

    y_train = (labels[train_idx] == "RR").astype(np.float32)
    y_val   = (labels[holdout_idx] == "RR").astype(np.float32)

    print(f"Train  (lc_strict):       {len(train_idx):4d}  →  RR={int(y_train.sum()):3d}  non-RR={int((1-y_train).sum())}")
    print(f"Val    (lc_gentle_extra): {len(holdout_idx):4d}  →  RR={int(y_val.sum()):3d}  non-RR={int((1-y_val).sum())}")
    return train_idx, holdout_idx, y_train, y_val


# ── Training ──────────────────────────────────────────────────────────────────

def train(Z, train_idx, y_train):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    print(f"\npos_weight RR: {pos_weight.item():.2f}")

    X_tr = torch.tensor(Z[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = MLP(LATENT_DIM, HIDDEN, DROPOUT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_loss, patience_cnt, best_state = float("inf"), 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        avg = total / len(X_tr)
        if avg < best_loss - 1e-4:
            best_loss, patience_cnt = avg, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}  patience={patience_cnt}/{PATIENCE}")
        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


# ── Threshold calibration ─────────────────────────────────────────────────────

def calibrate_threshold(probs_val, y_val):
    """Find the threshold that maximises F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_val, probs_val)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best_idx = np.argmax(f1[:-1])   # last precision/recall element has no corresponding threshold
    best_thr = float(thresholds[best_idx])
    print(f"\nCalibrated threshold (max F1 val): {best_thr:.3f}"
          f"  →  P={precision[best_idx]:.3f}  R={recall[best_idx]:.3f}  F1={f1[best_idx]:.3f}")
    return best_thr, precision, recall, thresholds, f1


# ── Prediction ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, Z):
    model.eval()
    X = torch.tensor(Z, dtype=torch.float32).to(DEVICE)
    logits = model(X)
    return torch.sigmoid(logits).cpu().numpy()


# ── UMAP ──────────────────────────────────────────────────────────────────────

def plot_umap(Z, prob_rr, true_labels, source, threshold, out_path):
    import umap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strict_mask = source == "strict"
    Z_plot = Z[strict_mask]
    prob_plot = prob_rr[strict_mask]

    print("UMAP fit...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="euclidean", random_state=RANDOM_SEED, verbose=False)
    emb = reducer.fit_transform(Z_plot)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Panel 1: prob_rr heatmap
    ax = axes[0]
    sc = ax.scatter(emb[:, 0], emb[:, 1], s=2, c=prob_plot, cmap="Reds",
                    alpha=0.7, rasterized=True, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="prob_RR")
    ax.set_title("RR Lyrae probability", fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_aspect("equal", "datalim")

    # Panel 2: binary prediction with known labelled RR highlighted
    ax = axes[1]
    pred_plot = (prob_plot >= threshold).astype(int)
    ax.scatter(emb[pred_plot == 0, 0], emb[pred_plot == 0, 1],
               s=1, c="#cccccc", alpha=0.3, rasterized=True, label=f"non-RR ({(pred_plot==0).sum()})")
    ax.scatter(emb[pred_plot == 1, 0], emb[pred_plot == 1, 1],
               s=4, c="#e63946", alpha=0.8, rasterized=True, label=f"RR pred ({(pred_plot==1).sum()})")

    # Highlight known Gaia RR labels
    all_ids_strict = [i for i, s in zip(range(len(source)), source) if s == "strict"]
    known_rr = np.array([i for i, idx in enumerate(all_ids_strict)
                         if true_labels[idx] == "RR"])
    if len(known_rr):
        ax.scatter(emb[known_rr, 0], emb[known_rr, 1],
                   s=40, c="gold", edgecolors="k", linewidths=0.5, zorder=6,
                   label=f"RR Gaia ({len(known_rr)})")

    ax.set_title(f"Binary predictions (threshold={threshold:.2f})", fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    ax.set_aspect("equal", "datalim")

    fig.suptitle("STEP 5a — RR Lyrae binary classifier\n(lc_strict, 13505 objects)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"=== VariAE STEP 5a — RR Binary Classifier ===\n")
    print(f"Device: {DEVICE}\n")

    Z, all_ids, labels, scores, source = load_data()
    print(f"Latent: {Z.shape}  (strict={(source=='strict').sum()}, extra={(source=='extra').sum()})")
    print(f"Total labelled: {(labels != None).sum()}  (no score threshold)\n")

    train_idx, holdout_idx, y_train, y_val = build_split(labels, scores, source)

    print("\nTraining binary MLP...")
    model = train(Z, train_idx, y_train)

    # Validation evaluation
    probs_val = predict(model, Z[holdout_idx])
    auc_roc = roc_auc_score(y_val, probs_val)
    auc_pr  = average_precision_score(y_val, probs_val)
    print(f"\nAUC-ROC: {auc_roc:.4f}  |  AUC-PR: {auc_pr:.4f}")

    threshold, precision, recall, thresholds, f1 = calibrate_threshold(probs_val, y_val)

    y_pred_val = (probs_val >= threshold).astype(int)
    lines = [
        "VariAE STEP 5a — RR Binary Classifier", "=" * 50, "",
        f"Train: {len(train_idx)} objects (strict labelled, score>=0.0)",
        f"Val:   {len(holdout_idx)} objects (gentle_extra labelled)",
        f"RR train: {int(y_train.sum())}  |  RR val: {int(y_val.sum())}",
        "",
        f"AUC-ROC:  {auc_roc:.4f}",
        f"AUC-PR:   {auc_pr:.4f}",
        f"Calibrated threshold (max F1 val): {threshold:.3f}",
        "",
        "Classification report (val, calibrated threshold):",
        classification_report(y_val.astype(int), y_pred_val,
                              target_names=["non-RR", "RR"], zero_division=0),
    ]

    # PR curve at selected thresholds
    lines.append("Precision-Recall at selected thresholds:")
    lines.append(f"  {'Threshold':>9}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = precision[np.searchsorted(thresholds, thr, side="right") - 1]
        r = recall[np.searchsorted(thresholds, thr, side="right") - 1]
        f = 2 * p * r / max(p + r, 1e-9)
        lines.append(f"  {thr:>9.2f}  {p:>9.3f}  {r:>6.3f}  {f:>6.3f}")

    eval_text = "\n".join(lines)
    print("\n" + eval_text)
    with open(os.path.join(RESULTS_DIR, "rr_eval.txt"), "w") as f:
        f.write(eval_text)

    # Inference on all 13505 lc_strict objects
    strict_mask = source == "strict"
    Z_strict    = Z[strict_mask]
    ids_strict  = [i for i, s in zip(all_ids, source) if s == "strict"]

    prob_rr_all    = predict(model, Z)          # all (strict+extra) for UMAP
    prob_rr_strict = prob_rr_all[strict_mask]
    pred_rr_strict = (prob_rr_strict >= threshold).astype(int)

    print(f"\nRR predicted in lc_strict (threshold={threshold:.2f}): {pred_rr_strict.sum()}")
    print(f"  prob_rr >= 0.9: {(prob_rr_strict >= 0.9).sum()}")
    print(f"  prob_rr >= 0.7: {(prob_rr_strict >= 0.7).sum()}")
    print(f"  prob_rr >= 0.5: {(prob_rr_strict >= 0.5).sum()}")

    df_out = pd.DataFrame({
        "diaObjectId": ids_strict,
        "prob_rr":     prob_rr_strict,
        "pred_rr":     pred_rr_strict,
    })
    df_out.to_csv(os.path.join(RESULTS_DIR, "rr_predictions.csv"), index=False)
    print(f"Predictions → {RESULTS_DIR}/rr_predictions.csv")

    torch.save(model.state_dict(), os.path.join(LATENT_DIR, "rr_binary.pt"))
    np.save(os.path.join(LATENT_DIR, "rr_threshold.npy"), np.array([threshold]))

    plot_umap(Z, prob_rr_all, labels, source, threshold,
              os.path.join(RESULTS_DIR, "rr_umap.png"))


if __name__ == "__main__":
    main()
