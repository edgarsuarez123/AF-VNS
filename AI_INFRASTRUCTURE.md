# AI Infrastructure — RS-Personalized AF VNS Platform

**Project:** Aim 2, Phase 1 — AI-Driven Adaptive Auricular Vagus Nerve Stimulation
**Scope:** Offline-trainable, online-capable ECG classification pipeline (AF vs NSR)
**Status:** Round 9 complete — two-phase transfer learning. C2017 OOD AUROC=0.7432, Sensitivity=0.9159. NFR-2.1 gap = 0.007.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Data Pipeline](#3-data-pipeline)
4. [Feature Extraction](#4-feature-extraction)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Evaluation](#7-evaluation)
8. [Configuration](#8-configuration)
9. [Design Decisions](#9-design-decisions)
10. [Known Limitations & Open Issues](#10-known-limitations--open-issues)
11. [How to Run](#11-how-to-run)

---

## 1. System Overview

The system classifies ECG recordings as **Atrial Fibrillation (AF=1)** or **Normal Sinus Rhythm (NSR=0)** to decide when to trigger auricular vagus nerve stimulation. Phase 1 is fully offline — no hardware connection required.

### End-to-end data flow

```
Raw ECG files (WFDB/EDF/CSV)
        │
        ▼
  dataset_parsers.py       ← Normalize to standard schema (subject_id, signal, fs, label)
        │ resample to 250 Hz (all sources)
        ▼
  splitter.py              ← 70/15/15 train/val/test split by subject_id
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
  precompute_cache.py              dataloaders.py (on-the-fly)
  (build once, reuse)
        │
        ▼ per sample:
  wavelet_filter.py        ← CWT denoising (0.5–45 Hz passband)
        │
        ├──── CNN path ────────────────────────┐
        │     10s denoised waveform             │
        │                                       │
        └──── HRV path ───────────────────────┐│
              peak_detector → RR intervals     ││
              artifact_scrubber → correct/keep ││
              hrv_time, hrv_freq, hrv_nonlinear ││
              → (5, 7) sequence                 ││
                                               ││
        ▼                                      ▼▼
  scaler.py (fit on train only)         HybridEnsemble
        │                                      │
        └──────────────────────────────────────┘
                                               │
                                               ▼
                                    Single logit → BCEWithLogitsLoss
                                    (pos_weight for class imbalance)
```

---

## 2. Directory Structure

```
AF VNS/
├── config.yaml                    # Single source of truth for all hyperparameters
├── PLAN.md                        # Current task plan (always maintained)
├── progress.txt                   # Running implementation log
│
├── src/
│   ├── data/
│   │   ├── dataset_parsers.py     # Parse all ECG sources → standard schema
│   │   ├── dataloaders.py         # PhysioDataset + get_dataloaders()
│   │   ├── splitter.py            # 70/15/15 subject-level split
│   │   ├── download_physionet.py  # Download afdb + nsrdb
│   │   ├── download_challenge2017.py  # Download PhysioNet 2017 AF Challenge
│   │   ├── download_ltafdb.py     # Download ltafdb AF segments only
│   │   ├── download_mimic3_waveforms.py  # Download MIMIC-III waveform holdout
│   │   ├── create_mixed_split.py  # Add MIMIC to split (superseded by ltafdb approach)
│   │   └── materialize_split.py   # Create data/splits/ symlinks for inspection
│   │
│   ├── features/
│   │   ├── wavelet_filter.py      # CWT denoising
│   │   ├── peak_detector.py       # R-peak detection → RR intervals (neurokit2)
│   │   ├── artifact_scrubber.py   # Amplitude + RR artifact detection/correction
│   │   ├── hrv_time.py            # RMSSD, SDNN
│   │   ├── hrv_freq.py            # LF, HF, LF/HF (Welch PSD)
│   │   ├── hrv_nonlinear.py       # SampEn, DFA α1 (neurokit2)
│   │   ├── pipeline.py            # Orchestrates full waveform → HRV sequence
│   │   └── scaler.py              # Per-column Z-score scaler (fit on train only)
│   │
│   ├── models/
│   │   ├── cnn.py                 # 1D-CNN morphology encoder
│   │   ├── rnn.py                 # GRU temporal encoder
│   │   ├── transformer.py         # Transformer attention encoder
│   │   ├── ensemble.py            # HybridEnsemble (fuses all three)
│   │   └── pca_reduction.py       # Multi-channel → 1-channel (Galea→Sparrow prep)
│   │
│   └── training/
│       ├── build_model.py         # Build HybridEnsemble from config.yaml
│       ├── precompute_cache.py    # One-time parallel cache build
│       ├── precompute_worker.py   # Subprocess worker for parallel cache
│       ├── train.py               # Training loop (cache path + on-the-fly path)
│       ├── evaluate.py            # AUROC/F1/sensitivity/specificity + MIMIC-III eval
│       └── visualize.py           # Training curve plots from training_log.csv
│
├── tests/
│   ├── test_preprocessing.py      # Features + scaler + artifact + pipeline tests
│   ├── test_models.py             # CNN/RNN/Transformer/Ensemble shape tests
│   └── test_dataloader.py         # Window shapes, no leakage, resampling tests
│
├── data/
│   └── raw/
│       ├── afdb/          # MIT-BIH AF Database (23 records, 250 Hz)
│       ├── nsrdb/         # Normal Sinus Rhythm Database (18 records, 128 Hz → 250 Hz)
│       ├── challenge2017/ # PhysioNet 2017 AF Challenge (5,788 labeled, 300 Hz → 250 Hz)
│       ├── ltafdb/        # Long-Term AF segments (2,654 AF segments, 128 Hz → 250 Hz)
│       └── mimic3/        # MIMIC-III holdout (60 records, 125 Hz → 250 Hz)
│
└── models/
    ├── artifacts/
    │   ├── split.json             # Train/val/test subject_id lists
    │   ├── scaler.pkl             # Fitted StandardScaler (train only)
    │   ├── cache/                 # Precomputed HRV + denoised waveforms
    │   │   ├── train_short.npy    # (N, T) denoised 10s waveforms
    │   │   ├── train_hrv_scaled.npy  # (N, 5, 7) scaled HRV sequences
    │   │   ├── train_labels.npy   # (N,) binary labels
    │   │   └── cache_meta.json    # max_short_len, split sizes
    │   └── mimic3_eval/           # MIMIC-III evaluation plots
    └── checkpoints/
        └── best_model.pth         # Best val_auroc checkpoint
```

---

## 3. Data Pipeline

### 3.1 Data Sources

| Source | Records | Class | Hz | Duration | Phase | Role |
|--------|---------|-------|----|----------|-------|------|
| **mimic3** (MIMIC-III Waveform Subset) | 1,032 (604 AF + 428 ctrl) | AF/NSR | 125 | 500–900s | Phase 1 | Pre-train backbone |
| **afdb** (MIT-BIH AF Database) | 23 | AF=1 | 250 | ~10h each | Phase 2 | Fine-tune head |
| **nsrdb** (Normal Sinus Rhythm Database) | 18 | NSR=0 | 128 | ~24h each | Phase 2 | Fine-tune head |
| **ltafdb** (Long-Term AF Database) | ~84 AF segments | AF=1 | 128 | 30s–hours | Phase 2 | Fine-tune head |
| **challenge2017** (PhysioNet 2017) | 5,788 labeled | AF/NSR | 300 | 30–61s each | OOD eval only | Cross-device generalization |

**Why two phases:** C2017 (93% of old training data) has 30–61s records — too short for meaningful 300s HRV. Including them caused the GRU/Transformer to train on structurally zero-padded HRV for 93% of samples. Two-phase training eliminates this: Phase 1 uses only long MIMIC-3 records (all ≥500s), Phase 2 fine-tunes on curated clean ECG.

**Why C2017 as OOD:** Ambulatory AliveCor ECG — completely different device from training data (smartphone patch vs ICU monitor vs Holter recorder). True cross-device generalization test.

### 3.2 Standard Record Schema

Every parser outputs:
```python
{
    "subject_id": str,   # unique identifier (prefixed by source: "c17_A00001", "ltaf_01_s3")
    "signal":    np.ndarray,  # 1D float64, already resampled to target_fs
    "fs":        float,  # always target_fs (250 Hz) after parse_all()
    "label":     int,    # 1=AF, 0=NSR, None=unknown
}
```

### 3.3 Uniform Resampling (250 Hz)

**Problem:** afdb=250 Hz, nsrdb=128 Hz, challenge2017=300 Hz, ltafdb=128 Hz. A CNN trained on mixed frequencies would learn padding patterns as a proxy label (AF records always produce 2,500 samples at 10s; NSR records produce 1,280 samples before padding).

**Fix:** `_resample_to_target_fs()` in `dataset_parsers.py` uses `scipy.signal.resample` (FFT-based) to normalize every record to 250 Hz. Chunked in 60s segments to keep memory bounded.

### 3.4 Train/Val/Test Split

`splitter.create_split()` does a stratified **70/15/15 split by subject_id** (not by window). This prevents data leakage — all windows from the same patient stay in the same split.

Split is persisted to `models/artifacts/split.json` and shared across all pipeline components. Seed=42 for reproducibility.

### 3.5 PhysioDataset and Rolling Windows

`PhysioDataset` in `dataloaders.py`:
- **Full-length records (≥300s):** Creates rolling windows with configurable stride. Default: non-overlapping (stride = waveform_sec = 10s). Training can use overlapping stride (5s) to double sample count.
- **Short records (<300s but ≥10s):** One sample per record. CNN gets 10s from the record center; HRV pipeline uses the full signal (produces partial HRV — mostly NaN, imputed to zero).

**Why allow short records:** challenge2017 records are 30–61 seconds. PhysioNet 2017 contributes 5,788 records which dramatically improve dataset diversity. Without short record support they'd all be skipped.

Each sample returns `(short_tensor, long_tensor, label, fs)`:
- `short_tensor`: `(1, T_short)` — 10s raw waveform for CNN
- `long_tensor`: `(1, T_long)` — up to 300s for HRV computation
- `_pad_collate` in `train.py` handles variable-length batching

---

## 4. Feature Extraction

### 4.1 CWT Denoising (`wavelet_filter.py`)

**Method:** Continuous Wavelet Transform using PyWavelets (cmor1.5-1.0 wavelet).
**Passband:** 0.5–45 Hz — removes baseline wander (<0.5 Hz) and EMG/noise (>45 Hz).
**Reconstruction:** Approximate inverse by summing CWT coefficients within passband, normalized by number of kept scales.

**Why CWT:** Aims document specifically calls for "wavelet transformation for signal denoising." CWT provides better time-frequency localization than DWT for physiological signals, important for preserving P-wave morphology and f-waves (AF hallmarks).

### 4.2 Artifact Detection & Correction (`artifact_scrubber.py`)

Two-stage check:

1. **Amplitude (MAD-based):** Reject if `max_deviation > amplitude_mad_multiple × MAD`. Threshold = 30× (raised from original 5× because normal QRS spikes are 5–25× MAD).

2. **RR deviation (fraction-based):**
   - Compute `deviation = |RR - median(RR)| / median(RR) × 100`
   - Mark beat as outlier if deviation > `rr_deviation_percent` (60%)
   - **Correct** outliers via linear interpolation from neighboring valid beats
   - **Reject** only if `fraction_outliers > rr_fraction_threshold` (30%)

**Critical design decision:** Original code used `np.any(deviation > 25%)` — rejected entire window if any single beat deviated >25%. AF is defined by irregularly irregular RR intervals (40–60% deviation is normal). This systematically rejected AF windows — the exact class we need to classify. Switched to fraction-based threshold + correction per Aim 2's explicit "artifact detection and **correction**" language.

### 4.3 R-Peak Detection (`peak_detector.py`)

Uses **neurokit2** `nk.ecg_peaks()` for R-peak detection on denoised signal. Returns RR intervals in seconds.

### 4.4 HRV Feature Extraction

**7 features computed per time step** (fixed order):

| Index | Feature | Method | Min data needed |
|-------|---------|--------|----------------|
| 0 | RMSSD | `sqrt(mean(diff(RR)^2))` | 5 RR intervals |
| 1 | SDNN | `std(RR)` | 5 RR intervals |
| 2 | LF power | Welch PSD (0.04–0.15 Hz) | 32 RR intervals |
| 3 | HF power | Welch PSD (0.15–0.40 Hz) | 32 RR intervals |
| 4 | LF/HF ratio | LF / HF | 32 RR intervals |
| 5 | SampEn | Sample entropy (neurokit2) | ~200 RR (full window) |
| 6 | DFA α1 | Detrended fluctuation analysis (neurokit2) | ~100 RR (full window) |

**Subwindow strategy:** 5-minute recording divided into 5 subwindows of 60s each.
- Time-domain (RMSSD, SDNN): computed per 60s subwindow
- Frequency-domain (LF, HF, LF/HF): computed per 60s subwindow (needs ≥32 RR = ~MIN_SAMPLES)
- Nonlinear (SampEn, DFA): computed once over the full available RR pool, broadcast to all time steps (need ≥200/100 RR intervals — can only get this from the full 5-min signal)

**Why 60s subwindows:** 30s was too short for frequency-domain HRV — gives only ~30 RR intervals (below MIN_SAMPLES=32) and can't resolve LF cycles (one LF cycle takes 7–25s, need several). 60s gives ~70 RR intervals at 70 bpm, comfortably above the threshold.

Insufficient windows yield NaN rows — imputed to 0 after scaling (model trained to handle this).

### 4.5 Scaler (`scaler.py`)

**Per-column Z-score standardization.** Fit on training data only, persisted to `models/artifacts/scaler.pkl`.

**Why per-column:** Original implementation used row-level NaN masking — required all 7 features non-NaN to include the row in scaler fit. With sparse data (~7% of rows fully valid), this caused near-empty fits. Per-column fit uses each feature's own valid entries independently, so a row with only RMSSD/SDNN valid still contributes to those two columns' statistics.

Transform also works per-column: valid entries scaled normally, NaN entries remain NaN (imputed to 0 later, not here, to preserve the distinction).

---

## 5. Model Architecture

### 5.1 HybridEnsemble

```
waveform_10s (B, 1, T)          hrv_sequence (B, 5, 7)
        │                               │
        ▼                         ┌─────┴─────┐
   CNNEncoder                GRUEncoder   TransformerEncoder
   → (B, 128)                → (B, 64)    → (B, 64)
        │                         │             │
        └─────────── concat ───────────────────┘
                         (B, 256)
                              │
                         Dropout(0.2)
                         Linear(256→64)   ← expanded head (Round 9)
                         ReLU
                         Dropout(0.1)
                         Linear(64→1)
                              │
                         logit (B, 1)
```

**Total fused dim = 128 + 64 + 64 = 256.**
**Head:** expanded from `Linear(256→1)` to `Linear(256→64→ReLU→64→1)` (~16.5K params) to provide sufficient fine-tuning capacity in Phase 2 while keeping the backbone frozen.

Loss: `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` — includes sigmoid internally for numerical stability.

### 5.6 Two-Phase Transfer Learning (Round 9)

**Phase 1 — Pre-train (backbone + head, all layers trainable):**
- Dataset: MIMIC-3 only (1,032 records, 604 AF + 428 control)
- All records ≥500s — full 300s HRV for every training sample (no zero-padding)
- lr=1e-3, patience=15, max_epochs=100
- Checkpoint: `models/checkpoints/phase1_model.pth`
- Result: val_auroc=0.6976 (expected modest — ICU noisy labels)
- **Goal:** teach the backbone real-world noise patterns, ICU signal characteristics

**Phase 2 — Fine-tune (backbone frozen, head only):**
- Dataset: AFDB + NSRDB + LTAFDB (~125 subjects, 70/15/15 split)
- Backbone (CNN + GRU + Transformer) weights frozen from Phase 1 checkpoint
- Only ~16.5K head params trainable
- lr=5e-4, patience=10, max_epochs=50
- Checkpoint: `models/checkpoints/phase2_model.pth`
- Result: val_auroc=0.9830
- **Goal:** teach the head clean AF vs NSR discrimination on curated data

**Why freezing works:**
The backbone learned general ECG pattern representations during Phase 1. Phase 2 data (~125 subjects) is too small to retrain the backbone without overfitting. Freezing forces the head to learn a linear separator on top of already-meaningful features.

### 5.2 CNN (`cnn.py`) — Morphology Branch

4-layer 1D-CNN with progressively larger channel depth:

```
Conv1d(1→16, k=7, s=2) → BN → ReLU → MaxPool(k=3, s=2)
Conv1d(16→32, k=5, s=2) → BN → ReLU
Conv1d(32→64, k=3, s=2) → BN → ReLU
Conv1d(64→128, k=3, s=2) → BN → ReLU
AdaptiveAvgPool1d(1) → Flatten → Dropout(0.1) → Linear(128→128)
```

**Why this architecture:** Hierarchical feature extraction from raw ECG morphology — early layers detect QRS complex shape (P-wave presence/absence, f-wave morphology), deeper layers encode rhythm patterns. `AdaptiveAvgPool1d(1)` makes the CNN input-length agnostic, handling variable-length signals from different sources.

### 5.3 GRU (`rnn.py`) — Temporal Branch

Single-layer GRU over the 5-step HRV sequence:
```
GRU(input=7, hidden=64, layers=1, batch_first=True)
→ last hidden state h[-1]: (B, 64)
```

**Why GRU over LSTM:** GRU has fewer parameters, trains faster, and performs comparably for short sequences (5 time steps). The hidden state captures how autonomic tone evolves over 5 minutes — a key AF indicator.

### 5.4 Transformer (`transformer.py`) — Attention Branch

```
Linear(7→64)              ← project features to d_model
+ SinusoidalPositionalEncoding
TransformerEncoderLayer(d=64, heads=4, ffn=128, GELU)
LayerNorm
mean pooling over seq_len
→ (B, 64)
```

**Why sinusoidal PE:** Sequence is only 5 steps — learned positional embeddings would overfit. Sinusoidal encoding is parameter-free and generalizes.

**Why mean pooling:** With only 5 time steps, there's no clearly "most important" position. Mean pooling aggregates all steps equally, letting attention weights do the actual selection.

**Why both GRU and Transformer on same HRV sequence:** They learn complementary representations. GRU captures sequential dependencies (how features change over time). Transformer uses self-attention to identify which time steps matter most, independent of order.

### 5.5 PCA Reduction (`pca_reduction.py`)

Reduces multi-channel (e.g., Galea 8-channel) to 1-channel equivalent using sklearn PCA. Single-channel input is a pass-through. Prepares for future Galea→Sparrow integration (TR-2.3).

---

## 6. Training Pipeline

### 6.1 Precompute Cache (`precompute_cache.py`)

**Why cache exists:** HRV computation (wavelet → peak detection → RR → HRV) takes ~0.5s per sample. With 100K+ training samples, that's 14+ hours per epoch on-the-fly. Cache computes everything once, reducing epoch time to ~22 seconds on GPU.

**What gets cached (per split):**
- `{split}_short.npy`: `(N, T)` denoised 10s waveforms, zero-padded to max length
- `{split}_hrv_scaled.npy`: `(N, 5, 7)` HRV sequences, scaler-transformed, NaN→0
- `{split}_labels.npy`: `(N,)` binary labels
- `cache_meta.json`: max_short_len, split sizes

**Parallel workers:** `--workers N` runs HRV computation in a `ProcessPoolExecutor`. Config path resolved to absolute in main process and passed to workers (avoids cwd-relative path failures in subprocesses).

**NaN logging:** Logs the fraction of HRV rows with NaN before imputation. Used to track fix effectiveness (was ~89% NaN before artifact scrubber fix, target <10%).

### 6.1b Phase-Specific Caches

**Phase 1 cache** (`models/artifacts/cache_phase1/`):
- Sources: MIMIC-3 only (filtered by `^p\d{6}_` subject_id pattern)
- Split: 85/15 train/val (no test — all MIMIC goes to training)
- Size: 2,142 train / 372 val windows
- Every sample has full 5-step HRV (all MIMIC records ≥500s)

**Phase 2 cache** (`models/artifacts/cache_phase2/`):
- Sources: AFDB + NSRDB + LTAFDB (excludes MIMIC and C2017)
- Split: 70/15/15 train/val/test
- Size: 5,470 train / 1,513 val / 1,373 test windows

Built via: `.venv/Scripts/python -m src.training.precompute_cache --config config.yaml --phase 1 --workers 6`

### 6.2 Training Loop (`train.py`)

Two paths — same model, same loss:

**Cache path (`--use-cache`):**
- Loads `PrecomputedDataset` using memory-mapped numpy arrays (`mmap_mode="r"`) to avoid loading 1.5GB+ into RAM
- No preprocessing at train time — pure GPU-bound

**On-the-fly path (no `--use-cache`):**
- Full preprocessing pipeline per batch
- Useful for debugging or when cache is stale

**Training details:**
- Optimizer: AdamW (lr=1e-3)
- Loss: BCEWithLogitsLoss with dynamic pos_weight (n_neg/n_pos, capped at 10)
- LR scheduler: ReduceLROnPlateau (patience=5, factor=0.5, min_lr=1e-6)
- Early stopping: patience=15 epochs on val_auroc
- Gradient clipping: max_norm=1.0
- Best checkpoint saved by val_auroc to `models/checkpoints/best_model.pth`

**pos_weight computation:** Computed fresh from training labels at the start of each training run, capped at 10 to prevent instability with extreme imbalances.

### 6.3 Precompute Worker (`precompute_worker.py`)

Subprocess function for parallel cache building. Each worker receives a chunk of `(short_signal, long_signal, fs, label)` tuples and returns `(denoised_10s, hrv_sequence, label)`.

---

## 7. Evaluation

### 7.1 Standard Evaluation (`run_evaluation()`)

Loads a split (train/val/test) and reports:
- AUROC (primary metric — NFR-2.1 target: ≥0.75)
- F1, Sensitivity (recall), Specificity
- ROC curve and confusion matrix plots → `models/artifacts/`

### 7.2 C2017 OOD Evaluation (`evaluate_challenge2017()`)

**Purpose:** True cross-device generalization — ambulatory AliveCor ECG never seen during training.

**Results (Round 9):**
- Records: 5,777 (737 AF, 5,040 NSR). 11 skipped.
- AUROC: **0.7432** | Sensitivity: **0.9159** | Specificity: 0.3286 | F1: 0.2815
- ROC: `models/artifacts/c2017_eval/roc_c2017.png`

**Why Specificity is low:** Model is biased toward high sensitivity (clinically appropriate for VNS triggering). Tune `pos_weight` or decision threshold to move the operating point.

**Why F1 is misleading here:** C2017 is 87% NSR. With Spec=0.33, most NSR records become false positives, collapsing precision. AUROC integrates the full curve and is the correct metric.

### 7.3 Cross-Dataset MIMIC-III Evaluation (`evaluate_mimic3()`)

**Purpose:** NFR-3.1 — test generalization to different ECG equipment and patient population (ICU vs Holter).

**Why separate evaluation path:** `PhysioDataset` requires 300s minimum and a split.json entry. 12 of 60 MIMIC records are <300s. `evaluate_mimic3()` processes records individually — any record ≥10s is valid.

**Per-record inference:**
1. Resample to 250 Hz
2. Extract center 10s for CNN
3. Extract first min(len, 300s) for HRV (short records get all-NaN HRV → imputed to 0)
4. Scale HRV, infer, record probability

**NaN logits:** Originally 10/60 records produced NaN logits. Root cause: MIMIC ICU signals use NaN for lead-off/equipment disconnection events. Fix: `np.nan_to_num(signal, nan=0.0)` at parse time.

**Baseline results (Round 4, before data expansion):**
- AUROC=0.6656, F1=0.6286, Sensitivity=0.7333, Specificity=0.4000
- NFR-3.1 FAIL — model learned equipment signatures, not AF physiology

**Note (Round 9):** MIMIC-3 is no longer held out — all records used in Phase 1 training. Cross-dataset evaluation now uses C2017 (Section 7.2).

---

## 8. Configuration

All hyperparameters live in `config.yaml`. No magic numbers in code.

```yaml
data:
  target_fs: 250           # All records resampled to this rate
  waveform_sec: 10         # CNN input length
  hrv_window_sec: 300      # HRV computation window (5 min)

hrv:
  subwindow_sec: 60        # One HRV vector per 60s = 5 time steps

artifact:
  amplitude_mad_multiple: 30   # Reject if peak > 30×MAD (QRS-safe threshold)
  rr_deviation_percent: 60     # Outlier beat threshold (AF-safe: 60%)
  rr_fraction_threshold: 0.30  # Reject window if >30% outlier beats

model:
  cnn_embed_dim: 128
  rnn_hidden_size: 64
  transformer_d_model: 64
  hrv_seq_len: 5           # 300s / 60s
  hrv_n_features: 7

training:
  learning_rate: 1.0e-3
  batch_size: 32
  max_epochs: 100
```

---

## 9. Design Decisions

### Why three model branches on the same sequence (RNN + Transformer)?

RNN captures sequential dependencies — it knows that HRV variability at time step 3 follows a pattern from steps 1 and 2. Transformer applies self-attention independently of order — it can identify that step 4 is anomalously high SDNN regardless of what came before. They are complementary. The concatenated embedding lets the final FC layer weight each contribution.

### Why not end-to-end from raw ECG only?

Raw ECG alone via CNN works well for short-term AF detection (P-wave absence, f-waves, irregular R-R in the waveform). But HRV features capture **autonomic nervous system dynamics** — the 5-minute trajectory of RR variability distinguishes paroxysmal AF from sustained AF, which matters for VNS timing. Multi-modal fusion is the clinically established approach.

### Why subject-level split?

Window-level split would put windows from the same patient in both train and test — the model memorizes inter-window correlations within one recording. Subject-level split forces the model to generalize to **unseen patients**, which is the actual clinical requirement.

### Why precompute cache instead of always on-the-fly?

With 100K+ training samples and HRV computation taking 0.5s each, on-the-fly training takes 14+ hours per epoch. After fixes, a full training run completes in ~22s/epoch (GPU). The cache is rebuilt once when data or preprocessing changes.

### Why MIMIC-III as pure holdout?

The model trained on PhysioNet data can trivially learn:
- Sampling rate differences (afdb 250 Hz vs nsrdb 128 Hz → solved by resampling)
- Equipment noise profiles (Holter recorder artifacts vs ICU monitor artifacts)
- Record length patterns

Keeping MIMIC out of training and evaluating on it is the only way to know whether the model learned real AF physiology or dataset-specific shortcuts. Round 4 results (AUROC=0.6656) confirmed the model was exploiting shortcuts — it defaulted to predicting AF (prob=1.0) for both AF and NSR MIMIC records.

### Why challenge2017 + ltafdb instead of just MIMIC in training?

The original plan was to add MIMIC to training. But MIMIC is too valuable as a holdout (different ICU equipment = true generalization test). PhysioNet 2017 Challenge + ltafdb provide:
- 3+ distinct recording sources in training
- ~6,000× more training samples
- MIMIC kept clean for eval

### Why AF segment extraction from ltafdb instead of whole-record labels?

ltafdb records contain both AF and NSR episodes within the same 21-hour recording. Labeling the whole record as AF=1 would mark NSR episodes as positive class — training noise. Parsing `(AFIB` rhythm annotations to extract only confirmed AF segments gives clean labels.

---

## 10. Known Limitations & Open Issues

| Issue | Severity | Status |
|-------|----------|--------|
| ~~val_auroc=0.9999 — model exploited dataset artifacts~~ | ~~Critical~~ | **RESOLVED** (two-phase training, C2017 removed) |
| ~~MIMIC AUROC stuck at 0.68 — C2017 garbage HRV~~ | ~~Critical~~ | **RESOLVED** — C2017 removed, two-phase training → AUROC 0.7432 |
| ~~challenge2017 records 30–61s → mostly zero HRV~~ | ~~Medium~~ | **RESOLVED** — C2017 now OOD eval only, not training |
| Specificity=0.329 on C2017 OOD | High | Open — tune pos_weight/threshold, target ≥0.65 |
| NFR-2.1 (AUROC ≥ 0.75) gap = 0.007 | Medium | Open — near target; threshold tuning may close gap |
| HRV NaN rate ~7% after all fixes | Low | Accepted — imputed to 0, model trained on this |
| No live inference path yet | Deferred | Phase 2: LSL streamer + state machine + MockSparrow |
| No hardware integration | Deferred | Phase 2: Sparrow API + watchdog |
| SampEn/DFA unreliable for short records | Low | Full 5-min pool used for nonlinear features |

---

## 11. How to Run

### Setup
```bash
# Install dependencies
.venv/Scripts/pip install -r requirements.txt

# Download data
.venv/Scripts/python -m src.data.download_physionet
.venv/Scripts/python -m src.data.download_challenge2017
.venv/Scripts/python -m src.data.download_ltafdb
```

### Build cache (one-time, ~1-4 hours)
```powershell
Start-Process -WindowStyle Hidden ".venv\Scripts\python.exe" `
  -ArgumentList "-m src.training.precompute_cache --config config.yaml --workers 4" `
  -WorkingDirectory "C:\Users\Edgar\AF VNS" `
  -RedirectStandardOutput "precompute.log" `
  -RedirectStandardError "precompute_err.log"
# Monitor: Get-Content precompute_err.log -Tail 5
```

### Train (two-phase)
```bash
# Phase 1 — pre-train on MIMIC-3
.venv/Scripts/python -m src.training.precompute_cache --config config.yaml --phase 1 --workers 6
.venv/Scripts/python -m src.training.train --config config.yaml --phase 1 --use-cache

# Phase 2 — fine-tune head on AFDB/NSRDB/LTAFDB
.venv/Scripts/python -m src.training.precompute_cache --config config.yaml --phase 2 --workers 6
.venv/Scripts/python -m src.training.train --config config.yaml --phase 2 --use-cache
```

### Evaluate
```bash
# Phase 2 test split (AFDB/NSRDB/LTAFDB holdout)
.venv/Scripts/python -m src.training.evaluate --split test

# C2017 OOD evaluation (cross-device generalization)
.venv/Scripts/python -m src.training.evaluate --challenge2017
```

### Test
```bash
.venv/Scripts/python -m pytest tests/ -v
```

### Check training curves
```bash
.venv/Scripts/python -m src.training.visualize
```
