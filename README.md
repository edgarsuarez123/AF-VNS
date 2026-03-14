# RS-Personalized AI-Driven Adaptive Stimulation — Aim 2, Phase 1

Offline-trainable, online-capable AF detection pipeline for auricular vagus nerve stimulation. Interfaces with Spark Biomedical Sparrow Link.

## Architecture

**Hybrid Ensemble:** CNN (10s raw waveform morphology) + GRU (5-min HRV sequence) + Transformer (attention over HRV) → fused logit → BCEWithLogitsLoss.

| Branch | Input | Dim |
|--------|-------|-----|
| CNN | 10s @ 250 Hz (2500 samples) | 128-d embedding |
| GRU | 5 × 7 HRV features (5 min / 60s steps) | 64-d hidden |
| Transformer | 5 × 7 HRV features | 64-d encoding |

HRV features (7): RMSSD, SDNN, LF, HF, LF/HF, SampEn, DFA α1.

---

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# CUDA (recommended):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only:
# pip install torch torchaudio

pip install -r requirements.txt
```

Verify: `python -c "import torch, mne, neurokit2, pylsl, yaml, sklearn, pywt; print('OK')"`

---

## Data

### PhysioNet (primary training data)

```powershell
.venv\Scripts\python -m src.data.download_physionet
```

Downloads to `data/raw/afdb/` (23 AF records, 250 Hz) and `data/raw/nsrdb/` (18 NSR records, resampled to 250 Hz). All signals are normalized to 250 Hz at load time to prevent sampling-rate bias in the CNN.

### MIMIC-III (cross-dataset generalization test — NFR-3.1)

Place exported files under `data/raw/mimic3/`. Expected format: CSV with columns `subject_id`, `time`, `ecg` (or `ppg`), optional `label`; or EDF files. If the directory is empty or missing, the pipeline skips MIMIC-III. This dataset is reserved for holdout evaluation only — not used in training.

### Split

70/15/15 train/val/test by `subject_id` (no patient leakage). Auto-created on first run if `models/artifacts/split.json` is missing.

---

## Workflow

### 1. Build precompute cache (one time, ~6 hrs on 4 cores)

**Run in background so it survives SSH disconnect (Windows):**

```powershell
Start-Process -WindowStyle Hidden ".venv\Scripts\python.exe" `
  -ArgumentList "-m src.training.precompute_cache --config config.yaml --workers 4 --chunk-size 8" `
  -WorkingDirectory "C:\Users\Edgar\AF VNS" `
  -RedirectStandardOutput "precompute.log" `
  -RedirectStandardError "precompute_err.log"
```

Check progress:
```powershell
Get-Content precompute_err.log -Tail 5
```

**Do not use `--workers 0` (all-core)** — Windows spawns full memory copies per worker and will OOM. Use `--workers 4` with `--chunk-size 8`.

### 2. Train

```powershell
.venv\Scripts\python -m src.training.train --use-cache
```

Target: AUROC ≥ 0.75 (NFR-2.1). Checkpoint saved to `models/checkpoints/best_model.pth`.

On-the-fly (no cache, slow): `python -m src.training.train`

### 3. Evaluate

```powershell
.venv\Scripts\python -m src.training.evaluate --split test
# Add --no-plots to skip figure generation
```

Outputs AUROC, F1, sensitivity, specificity. ROC + confusion matrix → `models/artifacts/`.

### 4. Visualize training curves

```powershell
.venv\Scripts\python -m src.training.visualize
```

Reads `training_log.csv`, saves `models/artifacts/training_curves.png`.

---

## Tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

**33/33 passing** (as of 2026-03-13). Covers preprocessing, HRV pipeline, artifact correction, uniform resampling, model shapes, dataloader integrity.

---

## Key Config Values (`config.yaml`)

| Key | Value | Why |
|-----|-------|-----|
| `data.target_fs` | 250 | Uniform Hz — prevents CNN sampling-rate bias |
| `hrv.subwindow_sec` | 60 | ~70 RR intervals per step (enough for freq features) |
| `model.hrv_seq_len` | 5 | 300s / 60s = 5 time steps |
| `artifact.amplitude_mad_multiple` | 30 | Normal QRS spikes are 5-25× MAD; only flag lead-off/saturation |
| `artifact.rr_deviation_percent` | 60 | AF has irregular RR (40-60% is normal) |
| `artifact.rr_fraction_threshold` | 0.30 | Reject window only if >30% of beats are outliers |
| `training.batch_size` | 32 | — |
| `training.max_epochs` | 100 | AUROC contingency |

---

## Pipeline Fixes Applied (2026-03-13)

Five critical data pipeline issues were identified and fixed before the cache rebuild:

| # | Issue | Fix |
|---|-------|-----|
| 1 | AF=250Hz / NSR=128Hz — CNN learns sampling rate as label proxy | Resample all signals to 250Hz in `dataset_parsers.py` |
| 2 | `np.any(dev_pct > 60)` rejects AF windows (AF IS irregular RR) | `correct_rr_intervals()` interpolates outlier beats; fraction-based rejection |
| 3 | SampEn/DFA always NaN — 60s gives ~70 RR, needs ~200/100 | Compute nonlinear features on full 5-min RR pool (~350 intervals), broadcast to all timesteps |
| 4 | NaN→0 with no logging — model can't distinguish missing from zero | Added NaN% logging before `nan_to_num` |
| 5 | Scaler transform: row-level NaN mask zeroed valid features | Per-column transform loop (matches `fit_scaler` logic) |

Smoke test result after fixes: both AF and NSR records → 5/5 valid HRV rows, 0 NaN across all 7 features (was ~89% NaN before).

---

## Project Layout

```
src/
  data/           dataset_parsers, dataloaders, splitter, download_physionet
  features/       wavelet_filter, artifact_scrubber, peak_detector, hrv_*, pipeline, scaler
  models/         cnn, rnn, transformer, ensemble, pca_reduction
  hardware/       sparrow_api (mock), watchdog, state_machine  ← deferred
  training/       train, evaluate, visualize, precompute_cache, build_model

tests/            test_preprocessing, test_models, test_dataloader
config.yaml       single source of truth for all hyperparameters and paths
PLAN.md           current task plan with step-by-step status
progress.txt      running implementation log with rationale
```

---

## Current Status (2026-03-14)

| Component | Status |
|-----------|--------|
| Data ingestion (PhysioNet) | Done |
| Preprocessing + HRV pipeline | Done — all 5 bugs fixed |
| Uniform resampling (250Hz) | Done |
| Artifact correction (fraction-based) | Done |
| Full-window nonlinear HRV | Done |
| Hybrid Ensemble model | Done |
| Training engine + cache | Done |
| **Cache rebuild** | **Running** (check `precompute_err.log`) |
| **First training run** | Pending (after cache) |
| MIMIC-III holdout eval (NFR-3.1) | Pending |
| 5-fold CV (TR-1.2) | Pending |
| LSL streamer | Deferred — Live Integration |
| live_inference.py | Deferred — Live Integration |
| Sparrow API + watchdog | Deferred — Live Integration |
| Latency profiler (p99 < 200ms) | Deferred |
| 24h HIL test | Deferred |

---

## Traceability

| Requirement | Implementation |
|-------------|----------------|
| FR-1.1 LSL ≥250 Hz | `config lsl.expected_hz`; `lsl_streamer` (deferred) |
| FR-1.2 Multi-dataset | `download_physionet`, `dataset_parsers` |
| FR-2.1 CWT denoising | `wavelet_filter.py` |
| FR-2.2 HRV windows | `dataloaders`, `hrv_*`, `pipeline` |
| FR-3.1 CNN morphology | `cnn.py` |
| FR-3.2 RNN temporal | `rnn.py` |
| FR-3.3 Transformer | `transformer.py` |
| FR-4.2 Safety limits | `config watchdog`; `watchdog.py` (deferred) |
| NFR-2.1 AUROC ≥0.75 | `evaluate.py`; target after cache rebuild |
| NFR-3.1 <5% drop MIMIC | MIMIC-III holdout eval (pending) |
| TR-1.1 Holdout by subject | `splitter`, `get_dataloaders` |
| TR-2.3 PCA / channel strip | `pca_reduction.py`, tests |

---

Update `config.yaml` watchdog and artifact thresholds per Sparrow/FDA limits and validation.
