"""
STEP 4 — Semi-Supervised VAE with SupCon loss + tabular features (period, LS power)
Combines:
  - Unsupervised VAE on 13935 light curves (lc_strict + crossmatch gentle extra)
  - Supervised Contrastive loss (SupCon) on labelled samples (score>=0.5)
  - Late fusion: log10(period) + ls_power concatenated to CNN output before mu/logvar

Encoder:
  CNN(6x100) → 256D
  tab[log10_period, ls_power] → FC(2→32) → 32D
  concat(256+32) → mu(32D), logvar(32D)

Loss:
  L = L_vae  +  λ_supcon · L_supcon

Output: latent/ssvae_latent_vectors.npy    (13505, 32)  — lc_strict
        latent/ssvae_latent_index.csv
        latent/ssvae_gentle_extra.npy       (430, 32)   — crossmatch extra
        latent/ssvae_gentle_extra_index.csv
        latent/ssvae_best.pt
        latent/ssvae_tab_stats.npz          mean/std of tabular features (used by step5)
        latent/ssvae_umap_preview.png
"""

import os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE              = "/home/monti/LSST/VariAE"
FOLDED_STRICT     = os.path.join(BASE, "folded_strict")
FOLDED_EXTRA      = os.path.join(BASE, "folded_gentle_extra")
LATENT_DIR        = os.path.join(BASE, "latent")
PERIODS_CSV       = os.path.join(BASE, "periods", "periods_strict.csv")
PERIODS_EXTRA_CSV = os.path.join(BASE, "periods", "periods_gentle_extra.csv")
XMATCH_CSV        = os.path.join(BASE, "crossmatch", "catalogue_lsst_classifier.csv")

LATENT_DIM    = 32
TAB_DIM       = 2       # [log10_period, ls_power]
BATCH_SIZE    = 256
EPOCHS        = 200
LR            = 1e-3
BETA_MAX      = 0.1
BETA_WARMUP   = 50
FREE_BITS     = 0.5
LAMBDA_MAX    = 1.0
LAMBDA_WARMUP = 80
TEMP          = 0.07
SCORE_THR     = 0.5

CLASSES      = ['AGN', 'CV', 'DSCT|GDOR|SXPHE', 'ECL', 'LPV',
                'RR',  'RS', 'S',  'SOLAR_LIKE',  'YSO']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES    = len(CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tabular features ──────────────────────────────────────────────────────────

def load_tab_features():
    """
    Load period_days and ls_power from periods_strict.csv and (if present)
    periods_gentle_extra.csv. Normalisation is computed on strict only (training
    set) to avoid contamination; extra objects use the same parameters.
    Returns:
      tab_map:      {diaObjectId: np.array([log10_period, ls_power])} — already normalised
      tab_fallback: np.zeros(2) — mean imputation for objects without a period
      tab_mean, tab_std: saved for step5 inference on new objects
    """
    periods = pd.read_csv(PERIODS_CSV, dtype={"diaObjectId": str})

    log_p  = np.log10(np.clip(periods["period_days"].values, 1e-3, None))
    ls_pow = periods["ls_power"].values.astype(float)

    # Normalisation statistics computed on strict (training set)
    tab_mean = np.array([log_p.mean(), ls_pow.mean()], dtype=np.float32)
    tab_std  = np.array([log_p.std(),  ls_pow.std()],  dtype=np.float32)
    tab_std  = np.where(tab_std < 1e-6, 1.0, tab_std)

    tab_map = {}
    for oid, lp, lsp in zip(periods["diaObjectId"], log_p, ls_pow):
        raw = np.array([lp, lsp], dtype=np.float32)
        tab_map[str(oid)] = (raw - tab_mean) / tab_std

    # Add gentle_extra if available (same normalisation as strict)
    if os.path.exists(PERIODS_EXTRA_CSV):
        extra = pd.read_csv(PERIODS_EXTRA_CSV, dtype={"diaObjectId": str})
        lp_e  = np.log10(np.clip(extra["period_days"].values, 1e-3, None))
        lsp_e = extra["ls_power"].values.astype(float)
        for oid, lp, lsp in zip(extra["diaObjectId"], lp_e, lsp_e):
            raw = np.array([lp, lsp], dtype=np.float32)
            tab_map[str(oid)] = (raw - tab_mean) / tab_std

    # Zero == mean in normalised space (mean imputation for objects without a period)
    tab_fallback = np.zeros(2, dtype=np.float32)

    return tab_map, tab_fallback, tab_mean, tab_std


# ── Dataset ───────────────────────────────────────────────────────────────────

class FoldedDataset(Dataset):
    def __init__(self, files, label_map, tab_map, tab_fallback):
        self.files  = files
        self.labels = []
        self.tabs   = []
        for f in files:
            oid = os.path.basename(f).replace("folded_", "").replace(".npy", "")
            self.labels.append(label_map.get(oid, -1))
            self.tabs.append(tab_map.get(oid, tab_fallback))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr  = np.load(self.files[idx])
        mask = ~np.isnan(arr)
        x    = torch.from_numpy(np.where(mask, arr, 0.0)).float()
        m    = torch.from_numpy(mask).float()
        tab  = torch.from_numpy(self.tabs[idx])
        return x, m, self.labels[idx], tab


# ── Model ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, ld, tab_dim=TAB_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 32, 5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2, stride=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2, stride=2), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.fc_cnn = nn.Sequential(nn.Linear(128 * 25, 256), nn.GELU())
        self.fc_tab = nn.Sequential(nn.Linear(tab_dim, 32), nn.GELU())
        self.fc_mu  = nn.Linear(256 + 32, ld)
        self.fc_lv  = nn.Linear(256 + 32, ld)

    def forward(self, x, tab):
        h_cnn = self.fc_cnn(self.conv(x).flatten(1))
        h_tab = self.fc_tab(tab)
        h     = torch.cat([h_cnn, h_tab], dim=1)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, ld):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ld, 256), nn.GELU(),
            nn.Linear(256, 128 * 25), nn.GELU(),
        )
        self.dconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, 6, 5, padding=2),
        )

    def forward(self, z):
        return self.dconv(self.fc(z).view(-1, 128, 25))


class SSVAE(nn.Module):
    def __init__(self, ld=LATENT_DIM, tab_dim=TAB_DIM):
        super().__init__()
        self.encoder = Encoder(ld, tab_dim)
        self.decoder = Decoder(ld)

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def forward(self, x, tab):
        mu, lv = self.encoder(x, tab)
        return self.decoder(self.reparameterize(mu, lv)), mu, lv


# ── Losses ────────────────────────────────────────────────────────────────────

def vae_loss(recon, x, mask, mu, lv, beta):
    n_valid    = mask.sum().clamp(min=1)
    recon_loss = ((recon - x) ** 2 * mask).sum() / n_valid
    kl         = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).clamp(min=FREE_BITS).sum(1).mean()
    return recon_loss + beta * kl, recon_loss, kl


def supcon_loss(features, labels, temp=TEMP):
    N = features.size(0)
    if N < 2:
        return torch.tensor(0.0, device=features.device)
    sim  = torch.mm(features, features.T).clamp(-1.0, 1.0) / temp
    eye  = torch.eye(N, device=features.device).bool()
    sim.masked_fill_(eye, float("-inf"))
    labels_row = labels.unsqueeze(0).expand(N, N)
    labels_col = labels.unsqueeze(1).expand(N, N)
    pos_mask   = (labels_row == labels_col) & ~eye
    has_pos    = pos_mask.any(1)
    if not has_pos.any():
        return torch.tensor(0.0, device=features.device)
    n_pos    = pos_mask.float().sum(1).clamp(min=1)
    log_prob = F.log_softmax(sim, dim=1)
    loss     = -(log_prob.masked_fill(~pos_mask, 0.0)).sum(1) / n_pos
    return loss[has_pos].mean()


def beta_schedule(epoch):
    return BETA_MAX * min(1.0, epoch / BETA_WARMUP)


def lambda_schedule(epoch):
    if epoch <= BETA_WARMUP:
        return 0.0
    return LAMBDA_MAX * min(1.0, (epoch - BETA_WARMUP) / LAMBDA_WARMUP)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, labeled_x, labeled_tab, labeled_y, optimizer, epoch):
    model.train()
    beta   = beta_schedule(epoch)
    lam    = lambda_schedule(epoch)
    totals = dict(loss=0, recon=0, kl=0, sc=0, mu_abs=0)

    for x, mask, _, tab in loader:
        x, mask, tab = x.to(DEVICE), mask.to(DEVICE), tab.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, lv = model(x, tab)
        loss, rl, kl  = vae_loss(recon, x, mask, mu, lv, beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["loss"]   += loss.item()
        totals["recon"]  += rl.item()
        totals["kl"]     += kl.item()
        totals["mu_abs"] += mu.abs().mean().item()

    if lam > 0 and labeled_x is not None:
        optimizer.zero_grad()
        mu_lab, _ = model.encoder(labeled_x, labeled_tab)
        z_norm    = F.normalize(mu_lab, dim=1)
        sc        = supcon_loss(z_norm, labeled_y)
        (lam * sc).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["sc"] = sc.item()

    nb = len(loader)
    return {k: v / nb for k, v in totals.items()}, beta, lam


# ── Latent extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(model, files, tab_map, tab_fallback):
    model.eval()
    tabs = [tab_map.get(os.path.basename(f).replace("folded_", "").replace(".npy", ""),
                        tab_fallback)
            for f in files]
    ds     = _InferenceDataset(files, tabs)
    loader = DataLoader(ds, batch_size=512, num_workers=4, pin_memory=True)
    vecs   = []
    for x, tab in tqdm(loader, desc="Encoding"):
        mu, _ = model.encoder(x.to(DEVICE), tab.to(DEVICE))
        vecs.append(mu.cpu().numpy())
    return np.vstack(vecs)


class _InferenceDataset(Dataset):
    def __init__(self, files, tabs):
        self.files = files
        self.tabs  = tabs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        x   = torch.from_numpy(np.where(~np.isnan(arr), arr, 0.0)).float()
        tab = torch.from_numpy(self.tabs[idx])
        return x, tab


# ── UMAP ──────────────────────────────────────────────────────────────────────

def umap_preview(latents, obj_ids, out_path):
    import umap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("UMAP fit...")
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    random_state=42).fit_transform(latents)

    clf    = pd.read_csv(XMATCH_CSV, dtype={"lsst_id": str})
    lmap   = dict(zip(clf.lsst_id, clf.best_class_name))
    smap   = dict(zip(clf.lsst_id, clf.best_class_score))
    labels = [lmap.get(o, "unknown") for o in obj_ids]
    scores = [smap.get(o, 0.0) for o in obj_ids]
    high   = np.array([l != "unknown" and s >= SCORE_THR for l, s in zip(labels, scores)])

    COLORS = {"RR": "#e63946", "ECL": "#2a9d8f", "LPV": "#e9c46a", "AGN": "#8338ec",
              "YSO": "#06d6a0", "SOLAR_LIKE": "#118ab2", "RS": "#ffd166",
              "S": "#ff9f1c", "DSCT|GDOR|SXPHE": "#cbf3f0", "CV": "#ff006e", "CEP": "#f4a261"}

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(emb[~high, 0], emb[~high, 1], s=1, c="#cccccc", alpha=0.3,
               rasterized=True, label="unlabeled")
    for cls, color in COLORS.items():
        idx = np.array([i for i, (l, h) in enumerate(zip(labels, high)) if l == cls and h])
        if len(idx) == 0:
            continue
        ax.scatter(emb[idx, 0], emb[idx, 1], s=22, c=color, zorder=5,
                   edgecolors="k", linewidths=0.3, label=f"{cls} ({len(idx)})")
    ax.set_title("UMAP — SS-VAE + SupCon + period features\n(score≥0.5 per class)", fontsize=13)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", fontsize=7, markerscale=2, framealpha=0.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LATENT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Tabular features
    tab_map, tab_fallback, tab_mean, tab_std = load_tab_features()
    n_extra_periods = 0 if not os.path.exists(PERIODS_EXTRA_CSV) else len(
        pd.read_csv(PERIODS_EXTRA_CSV))
    print(f"\nPeriods loaded: {len(tab_map)} objects "
          f"(strict + {n_extra_periods} gentle_extra)")
    print(f"Tab mean: log10_period={tab_mean[0]:.3f}  ls_power={tab_mean[1]:.3f}")
    print(f"Tab std:  log10_period={tab_std[0]:.3f}  ls_power={tab_std[1]:.3f}")
    np.savez(os.path.join(LATENT_DIR, "ssvae_tab_stats.npz"), mean=tab_mean, std=tab_std)

    # Labels
    clf       = pd.read_csv(XMATCH_CSV, dtype={"lsst_id": str})
    label_map = {oid: CLASS_TO_IDX[cls]
                 for oid, cls, sc in zip(clf.lsst_id, clf.best_class_name, clf.best_class_score)
                 if sc >= SCORE_THR and cls in CLASS_TO_IDX}

    # Training files
    files_strict = sorted(glob.glob(os.path.join(FOLDED_STRICT, "folded_*.npy")))
    files_extra  = sorted(glob.glob(os.path.join(FOLDED_EXTRA,  "folded_*.npy")))
    files_all    = files_strict + files_extra

    def oid(f): return os.path.basename(f).replace("folded_", "").replace(".npy", "")

    n_labeled   = sum(1 for f in files_all if oid(f) in label_map)
    n_no_period = sum(1 for f in files_all if oid(f) not in tab_map)
    print(f"\nTraining: {len(files_all)} curves  "
          f"(strict={len(files_strict)}, extra={len(files_extra)})")
    print(f"Labeled (score>={SCORE_THR}): {n_labeled}")
    print(f"No period (mean imputation): {n_no_period}")
    print(f"Classes: {CLASSES}\n")

    dataset = FoldedDataset(files_all, label_map, tab_map, tab_fallback)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=False)

    # Pre-load labeled samples on GPU for the SupCon pass
    lab_files = [f for f in files_all if oid(f) in label_map]
    lab_y     = torch.tensor([label_map[oid(f)] for f in lab_files], device=DEVICE)
    lab_arrs, lab_tabs = [], []
    for f in lab_files:
        arr = np.load(f)
        lab_arrs.append(torch.from_numpy(np.where(~np.isnan(arr), arr, 0.0)).float())
        lab_tabs.append(torch.from_numpy(tab_map.get(oid(f), tab_fallback)))
    labeled_x   = torch.stack(lab_arrs).to(DEVICE)
    labeled_tab = torch.stack(lab_tabs).to(DEVICE)
    print(f"Labeled batch: {labeled_x.shape}  "
          f"classes: {dict(zip(*np.unique(lab_y.cpu(), return_counts=True)))}\n")

    model     = SSVAE().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}  "
          f"Epochs: {EPOCHS}  β_max={BETA_MAX}  λ_max={LAMBDA_MAX}  τ={TEMP}\n")

    best_recon = float("inf")
    ckpt = os.path.join(LATENT_DIR, "ssvae_best.pt")

    for epoch in range(1, EPOCHS + 1):
        m, beta, lam = train_epoch(model, loader, labeled_x, labeled_tab, lab_y, optimizer, epoch)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"recon={m['recon']:.4f}  KL={m['kl']:.4f}  "
                  f"SC={m['sc']:.4f}  |μ|={m['mu_abs']:.3f}  "
                  f"β={beta:.3f}  λ={lam:.3f}")
        if m["recon"] < best_recon:
            best_recon = m["recon"]
            torch.save(model.state_dict(), ckpt)

    print(f"\nBest recon: {best_recon:.4f}  → {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # Extract latents
    latents_strict = extract_latents(model, files_strict, tab_map, tab_fallback)
    obj_ids_strict = [oid(f) for f in files_strict]
    np.save(os.path.join(LATENT_DIR, "ssvae_latent_vectors.npy"), latents_strict)
    pd.DataFrame({"diaObjectId": obj_ids_strict, "idx": range(len(obj_ids_strict))}) \
      .to_csv(os.path.join(LATENT_DIR, "ssvae_latent_index.csv"), index=False)

    latents_extra = extract_latents(model, files_extra, tab_map, tab_fallback)
    np.save(os.path.join(LATENT_DIR, "ssvae_gentle_extra.npy"), latents_extra)
    pd.DataFrame({"diaObjectId": [oid(f) for f in files_extra], "idx": range(len(files_extra))}) \
      .to_csv(os.path.join(LATENT_DIR, "ssvae_gentle_extra_index.csv"), index=False)

    print(f"Latent strict: {latents_strict.shape}")
    print(f"Latent extra:  {latents_extra.shape}")

    umap_preview(latents_strict, obj_ids_strict,
                 os.path.join(LATENT_DIR, "ssvae_umap_preview.png"))

    # Inter-centroid separation diagnostic
    from scipy.spatial.distance import cdist as scipy_cdist
    all_lat = np.vstack([latents_strict, latents_extra])
    inv_label = {v: k for k, v in CLASS_TO_IDX.items()}
    print("\nCentroid separation post-training:")
    centroids = {}
    for f, lat in zip(files_all, all_lat):
        o = oid(f)
        if o in label_map:
            centroids.setdefault(label_map[o], []).append(lat)
    cents    = {cls: np.mean(vecs, axis=0) for cls, vecs in centroids.items()}
    cls_list = sorted(cents.keys())
    c_mat    = np.array([cents[c] for c in cls_list])
    dists    = scipy_cdist(c_mat, c_mat)
    print(f"  Min inter-centroid: {dists[dists > 0].min():.3f}")
    for c in cls_list:
        vecs  = np.array(centroids[c])
        intra = np.linalg.norm(vecs - cents[c], axis=1).mean()
        print(f"  {inv_label[c]:22s}: n={len(vecs):3d}  intra-spread={intra:.3f}")


if __name__ == "__main__":
    main()
