# VariAE — Semi-supervised latent learning for LSST variable star classification

Semi-supervised classification of variable stars from LSST DP1.
Primary goal: discriminate **RR Lyrae** from other variable types.

---

## Pipeline

```
STEP 2  Gaia DR3 crossmatch analysis
  ↓
STEP 3  Multiband Lomb-Scargle + phase-fold
  ↓
STEP 4  SS-VAE — Semi-Supervised VAE with SupCon loss + period features
  ↓
STEP 5a  Binary classifier: RR Lyrae vs rest
STEP 5b  Multiclass classifier (10 classes)
```

---

## Data structure

```
VariAE/
├── lc_strict/                  # 13,505 LCs (nDiaSources≥30, StetsonJ>0.5)
├── lc_gentle/                  # 51,473 LCs (nDiaSources≥10)
├── folded_strict/              # 13,505 × folded_<id>.npy  (6, 100) float32
├── folded_gentle_extra/        # 430 × folded_<id>.npy     (crossmatch only, not in strict)
├── crossmatch/
│   ├── catalogue_lsst_classifier.csv   # 725 objects × Gaia DR3 best_class_name/score
│   └── catalogue_lsst_sos.csv
├── catalogs/
│   └── variable_objects_catalog.csv
├── periods/
│   ├── periods_strict.csv              # diaObjectId, period_days, ls_power
│   └── periods_gentle_extra.csv        # diaObjectId, period_days, ls_power (430 objects)
├── latent/
│   ├── ssvae_best.pt                   # SS-VAE checkpoint (with period features)
│   ├── ssvae_latent_vectors.npy        # (13,505, 32) — lc_strict
│   ├── ssvae_latent_index.csv
│   ├── ssvae_gentle_extra.npy          # (430, 32)   — crossmatch extra
│   ├── ssvae_gentle_extra_index.csv
│   ├── ssvae_tab_stats.npz             # mean/std of tabular features (for inference)
│   ├── rr_binary.pt                    # binary RR classifier checkpoint
│   └── rr_threshold.npy               # calibrated RR threshold
└── results/
    ├── rr_predictions.csv              # diaObjectId, prob_rr, pred_rr
    ├── rr_eval.txt                     # AUC-ROC, AUC-PR, calibrated threshold
    ├── rr_umap.png
    ├── clf_predictions.csv             # diaObjectId, pred_knn, pred_svm, prob_rr_*
    ├── clf_eval.txt
    └── clf_umap.png
```

### Light curve format

Each `lc_<diaObjectId>.csv`:
```
diaObjectId, coord_ra, coord_dec, visit, band, psfFlux, psfFluxErr,
psfDiffFlux, psfDiffFluxErr, expMidptMJD
```

### Folded array format

Each `folded_<diaObjectId>.npy`: shape `(6, 100)` float32.
- Rows: bands `[u, g, r, i, z, y]`
- Columns: 100 uniform phase bins over `[0, 1)`
- `NaN` where a band is absent
- Values normalised: zero mean, unit std per band

---

## Gaia DR3 labels

Crossmatch with Gaia DR3 `vari_classifier_result`: **725 objects**.

| Class | Total | Score≥0.5 | In lc_strict | In gentle_extra |
|-------|-------|-----------|--------------|-----------------|
| SOLAR_LIKE | 159 | 70 | ~46 | ~24 |
| LPV | 148 | 32 | ~42 | ~106 |
| RS | 113 | 33 | ~33 | ~80 |
| AGN | 101 | 63 | ~29 | ~72 |
| ECL | 61 | 36 | ~18 | ~43 |
| **RR** | **53** | **40** | **~16** | **~37** |
| YSO | 44 | 23 | ~13 | ~31 |
| S | 22 | 13 | ~6 | ~16 |
| DSCT\|GDOR\|SXPHE | 18 | 10 | ~5 | ~13 |
| CEP | 3 | 0 | ~1 | ~2 |
| CV | 2 | 2 | ~1 | ~1 |
| SN | 1 | — | — | — (excluded) |

Natural split: **lc_strict = training**, **lc_gentle_extra = validation**.

---

## Usage

```bash
# STEP 2 — Crossmatch analysis
python3 step2_crossmatch_analysis.py

# STEP 3 — Lomb-Scargle + phase-fold (lc_strict, ~5 min on 19 cores)
python3 step3_lomb_scargle.py

# STEP 3b — Fold and period extraction for the 430 extra LCs
python3 step3b_fold_gentle_extra.py

# STEP 4 — SS-VAE with SupCon + period features (~200 epochs, GPU recommended)
python3 step4_ssvae.py

# STEP 5a — Binary RR Lyrae vs rest classifier
python3 step5a_rr_binary.py

# STEP 5b — 10-class classifier (k-NN + SVM)
python3 step5b_multiclass.py
```

---

## Technical details

### STEP 3 — Lomb-Scargle + phase-fold

- Multiband LS: sum of normalised powers, frequency grid `[1/100, 24]` day⁻¹, 100k points
- Phase-fold onto a uniform 100-bin grid, normalised per band (zero mean, unit std)
- Automatic resume (skips already-processed files)

**Results**: ls_power ≥ 0.3 for **60%** of objects (8,084/13,505).

---

### STEP 4 — SS-VAE (Semi-Supervised VAE)

**Late fusion** architecture: the phase-folded light curve and tabular features
(period, ls_power) are processed by two separate branches and concatenated
before the latent bottleneck.

```
CNN(6×100) → 256D  ┐
                    ├─ concat(288D) → μ(32D), logvar(32D)
FC(log10P, lsp→32) ┘
```

**Loss per epoch**:
```
L = L_vae  +  λ · L_SupCon
L_vae    = MSE(recon, x, NaN-mask) + β · KL(free_bits=0.5)
L_SupCon = Supervised Contrastive Loss (Khosla et al. 2020, τ=0.07)
```

- β warmup: 0 → 0.1 over 50 epochs
- λ warmup: 0 → 1.0 between epochs 50 and 130
- SupCon applied to all 322 labelled objects (score≥0.5) in a dedicated per-epoch pass

**Results**:

| Metric | Value |
|--------|-------|
| Final SC loss | 0.068 |
| Min inter-centroid distance | 2.225 |
| Intra-spread (all classes) | 1.4 – 2.0 |

---

### STEP 5a — RR Lyrae binary classifier

MLP `32→128→64→1` with `BCEWithLogitsLoss(pos_weight)`.
Uses all 725 labelled objects (no score threshold) to maximise RR training samples.

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.845 |
| AUC-PR | 0.731 |
| Calibrated threshold (max F1 val) | 0.946 |
| Precision @ threshold | 1.000 |
| Recall @ threshold | 0.649 |
| RR predicted in lc_strict | 176 |

Limitation: only ~16 RR in training (the majority are in lc_gentle_extra).

---

### STEP 5b — Multiclass classifier

**k-NN / SVM** classifier operating directly on the SS-VAE latent space.
Non-parametric models are preferred over an MLP given the small number of
training samples (~14/class): they generalise better in this regime.

- **k-NN**: no training, Euclidean distance in latent space
- **Linear SVM**: max-margin, robust with few samples, Platt-calibrated probabilities
- RR threshold calibrated on the validation set (max F1)

---

## Methodological notes

### Data leakage and performance estimates

The SS-VAE is trained with SupCon on all 322 labelled objects (score≥0.5),
including the validation objects (lc_gentle_extra). This structures the latent
space around those same examples, making validation metrics optimistic.

True performance on the ~12k unlabelled objects is unknown by construction
(no ground truth available). The classifiers produce **candidate catalogues**
to be validated via follow-up observations or crossmatch with external catalogues
(VSX, LINEAR, SDSS Stripe 82).

### Calibrated RR threshold

For both classifiers the RR score threshold is calibrated by maximising F1 on
the validation set. Lower thresholds increase recall (more candidates) at the
cost of precision (more false positives).

---

## Hardware

- GPU: **NVIDIA GeForce RTX 4070 Laptop** (CUDA 12.6)
- CPU: 19 cores (used for LS multiprocessing)
- OS: Linux WSL2

---

## Citation

```bibtex
@article{monti2026VariAE,
  title   = {VariAE: Semi-supervised variable star classification on
             LSST with a Variational Autoencoder},
  author  = {Monti, Lorenzo},
  year    = {2026}
}
```
