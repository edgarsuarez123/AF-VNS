# RS-Personalized AI-Driven Adaptive Stimulation — Aim 2, Phase 1

Offline-trainable, online-capable **AF (atrial fibrillation) vs NSR (normal sinus rhythm)** detection pipeline for auricular vagus nerve stimulation. Interfaces with Spark Biomedical Sparrow Link (mock in Phase 1; hardware deferred to Live Integration).

---

## Goals of the Project

- **Core adaptive AI:** Build an inference pipeline that ingests multi-modal physiological data (ECG/PPG), processes HRV via wavelet transforms, runs a Hybrid Ensemble (CNN + RNN + Transformer), and outputs state-machine controlled trigger commands to the Sparrow Link pulse generator.
- **Validation targets:** AUROC ≥ 0.75 on held-out test data (NFR-2.1); &lt;5% performance drop on cross-dataset (MIMIC-III / Challenge 2017) evaluation (NFR-3.1).
- **Reproducibility:** Single `config.yaml` for all hyperparameters and paths; train/val/test split by subject (no patient leakage); optional two-phase transfer learning (Phase 1: MIMIC-III pre-training, Phase 2: AFDB/NSRDB fine-tuning).
- **Later phases:** LSL streaming, live inference, Sparrow API + watchdog, latency &lt;200 ms, 24 h HIL tests (deferred to Live Integration).

---

## Architecture

**Hybrid Ensemble:** CNN (10 s raw waveform morphology) + GRU (5 min HRV sequence) + Transformer (attention over HRV) → fused logit → BCEWithLogitsLoss.

| Branch      | Input                          | Dim           |
|------------|----------------------------------|---------------|
| CNN        | 10 s @ 250 Hz (2500 samples)     | 128-d embedding |
| GRU        | 5 × 7 HRV features (5 min / 60 s steps) | 64-d hidden   |
| Transformer| 5 × 7 HRV features              | 64-d encoding   |

HRV features (7): RMSSD, SDNN, LF, HF, LF/HF, SampEn, DFA α1. All signals are resampled to 250 Hz so the CNN does not see sampling-rate bias.

---

## Running the Project on Another Computer

### 1. Clone and environment

```powershell
git clone <repo-url> "AF VNS"
cd "AF VNS"

py -3.11 -m venv .venv
.\.venv\Activate.ps1
```

**CUDA (recommended):**

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**  
`pip install torch torchaudio` (then install rest of deps).

```powershell
pip install -r requirements.txt
```

Verify:

```powershell
.venv\Scripts\python -c "import torch, mne, neurokit2, pylsl, yaml, sklearn, pywt; print('OK')"
```

Use the project’s Python for all commands: **`.venv\Scripts\python`** (or `python` if the venv is activated). Do not use system Python.

### 2. Download data

**PhysioNet (required for training):**

```powershell
.venv\Scripts\python -m src.data.download_physionet
```

This fills `data/raw/afdb/` (MIT-BIH AF) and `data/raw/nsrdb/` (Normal Sinus Rhythm). No credentials needed.

**Optional — MIMIC-III (for Phase 1 pre-training / cross-dataset eval):**  
See [Downloading MIMIC-III data](#downloading-mimic-iii-data) below. If `data/raw/mimic3/` is missing or empty, the pipeline skips MIMIC-III.

**Optional — Challenge 2017 (out-of-distribution evaluation only):**

```powershell
.venv\Scripts\python -m src.data.download_challenge2017
```

Puts data in `data/raw/challenge2017/`. Used only for `evaluate --challenge2017`, not for training.

### 3. Tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

Fix any failures before training.

### 4. Precompute cache (one-time, long run)

Builds denoised 10 s waveforms and HRV features so training uses cached arrays instead of on-the-fly preprocessing.

**Foreground:**

```powershell
.venv\Scripts\python -m src.training.precompute_cache --config config.yaml --workers 4
```

**Background (e.g. SSH-friendly on Windows):**

```powershell
Start-Process -WindowStyle Hidden ".venv\Scripts\python.exe" `
  -ArgumentList "-m src.training.precompute_cache --config config.yaml --workers 4" `
  -WorkingDirectory "C:\Users\Edgar\AF VNS" `
  -RedirectStandardOutput "precompute.log" `
  -RedirectStandardError "precompute_err.log"
```

Check progress: `Get-Content precompute_err.log -Tail 5`

Do **not** use `--workers 0` (all cores) on Windows; use e.g. `--workers 4` to avoid OOM.

**Two-phase transfer learning:** To build Phase 1 (MIMIC) and Phase 2 (AFDB/NSRDB) caches:

```powershell
.venv\Scripts\python -m src.training.precompute_cache --config config.yaml --workers 4 --phase 1
.venv\Scripts\python -m src.training.precompute_cache --config config.yaml --workers 4 --phase 2
```

### 5. Train

**Single-phase (legacy, all data):**

```powershell
.venv\Scripts\python -m src.training.train --use-cache
```

**Two-phase:**

```powershell
.venv\Scripts\python -m src.training.train --use-cache --phase 1
.venv\Scripts\python -m src.training.train --use-cache --phase 2 --phase1-checkpoint models/checkpoints/phase1_model.pth
```

Best checkpoint: `models/checkpoints/best_model.pth` (legacy) or `phase1_model.pth` / `phase2_model.pth`.

### 6. Evaluate

```powershell
.venv\Scripts\python -m src.training.evaluate --split test
.venv\Scripts\python -m src.training.evaluate --mimic3
.venv\Scripts\python -m src.training.evaluate --challenge2017
```

Use `--no-plots` to skip figure generation. ROC and confusion matrices go to `models/artifacts/` (e.g. `mimic3_eval/`, `c2017_eval/`).

### 7. Training curves

```powershell
.venv\Scripts\python -m src.training.visualize
```

Reads `training_log.csv`, writes `models/artifacts/training_curves.png`.

---

## Downloading MIMIC-III Data

MIMIC-III is used for **Phase 1 pre-training** and/or **cross-dataset evaluation** (NFR-3.1). The pipeline expects data under `data/raw/mimic3/`.

### Option A — Automated download (WFDB waveforms + labels)

1. **PhysioNet access**
   - Request access to **MIMIC-III Waveform Database Matched Subset**: [PhysioNet MIMIC-III Waveforms](https://physionet.org/content/mimic3wdb-matched/).
   - For the script’s AF/non-AF selection you also need **MIMIC-III Clinical Database** (for `DIAGNOSES_ICD.csv`): [PhysioNet MIMIC-III](https://physionet.org/content/mimiciii/). Complete the credentialing and accept the use agreement for both.

2. **Get DIAGNOSES_ICD.csv**
   - From the MIMIC-III Clinical Database export, copy `DIAGNOSES_ICD.csv` into `data/raw/mimic3/`.  
   - (The repo does not ship this file; it is subject to PhysioNet’s license.)

3. **Run the download script**

   Set credentials (or pass on the command line):

   ```powershell
   $env:PHYSIONET_USER = "your_physionet_username"
   $env:PHYSIONET_PASS = "your_physionet_password"
   ```

   Then:

   ```powershell
   .venv\Scripts\python -m src.data.download_mimic3_waveforms --out-dir data/raw/mimic3 --diagnoses-csv data/raw/mimic3/DIAGNOSES_ICD.csv --n-af 30 --n-control 30
   ```

   This downloads WFDB records into `data/raw/mimic3/` and writes `.label` files (1 = AF, 0 = control) so `dataset_parsers.py` can use them.

### Option B — Manual placement

Place your own files in `data/raw/mimic3/` in one of these forms:

- **WFDB:** For each record, put `<name>.hea`, `<name>.dat`, and optionally `<name>.label` (single line: `1` or `0`). Record names must be unique (e.g. `p000020_00001`).
- **CSV:** Columns `subject_id`, `time`, `ecg` (or `ppg`), and optional `label`. One file per subject or one combined file; parser uses `subject_id` to group.
- **EDF:** One EDF per subject; the parser will use the first ECG-like channel.

If the directory is missing or empty, the pipeline skips MIMIC-III and continues without it.

---

## Data Summary

| Source        | Path                    | Role |
|---------------|-------------------------|------|
| MIT-BIH AF    | `data/raw/afdb/`        | Training + eval |
| Normal Sinus  | `data/raw/nsrdb/`       | Training + eval |
| MIMIC-III     | `data/raw/mimic3/`      | Phase 1 pre-training and/or cross-dataset eval |
| Challenge 2017| `data/raw/challenge2017/` | OOD evaluation only (not training) |

Splits are by **subject_id** (70/15/15 or phase-specific). Split files: `models/artifacts/split.json`, `phase1_split.json`, `phase2_split.json`.

---

## Key Config Values (`config.yaml`)

| Key | Value | Purpose |
|-----|-------|---------|
| `data.target_fs` | 250 | Uniform sampling for CNN (avoids rate bias) |
| `hrv.subwindow_sec` | 60 | One HRV vector per 60 s → 5 steps over 5 min |
| `model.hrv_seq_len` | 5 | 300 s / 60 s |
| `artifact.rr_deviation_percent` | 60 | AF has irregular RR; avoid rejecting true AF windows |
| `artifact.rr_fraction_threshold` | 0.30 | Reject window only if &gt;30% of beats are outliers |
| `training.batch_size` | 32 | |
| `training.max_epochs` | 100 | AUROC contingency |
| `evaluation.threshold` | 0.5 | Classification threshold (tune after threshold analysis) |

---

## What’s New (Recent Additions)

- **Two-phase transfer learning:** Phase 1 = MIMIC-III only (85/15 train/val); Phase 2 = AFDB/NSRDB (and optionally LTAFDB) with frozen backbone and fine-tuned head. Phase splits and caches: `phase1_split.json`, `phase2_split.json`, `cache_phase1/`, `cache_phase2/`.
- **Challenge 2017 OOD evaluation:** `evaluate --challenge2017` on 30–61 s single-lead ECG; uses Phase 2 checkpoint and scaler; outputs to `models/artifacts/c2017_eval/`.
- **Configurable evaluation threshold:** `evaluation.threshold` and sensitivity target in `config.yaml`; threshold analysis in `evaluate.py` to tune operating point.
- **Pipeline fixes (historical):** Uniform 250 Hz resampling, fraction-based artifact rejection for AF, full-window nonlinear HRV (SampEn/DFA), per-column scaler transform, NaN handling with logging.

---

## Project Layout

```
src/
  data/           dataset_parsers, dataloaders, splitter, download_physionet,
                  download_mimic3_waveforms, download_challenge2017
  features/       wavelet_filter, artifact_scrubber, peak_detector, hrv_*, pipeline, scaler
  models/         cnn, rnn, transformer, ensemble, pca_reduction
  hardware/       sparrow_api (mock), watchdog, state_machine  ← deferred
  training/       train, evaluate, visualize, precompute_cache, build_model

tests/            test_preprocessing, test_models, test_dataloader, test_evaluate, ...
config.yaml       single source of truth for paths and hyperparameters
PLAN.md           current task plan and step status
progress.txt      implementation log and rationale
```

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
| NFR-2.1 AUROC ≥0.75 | `evaluate.py`; threshold tuning |
| NFR-3.1 &lt;5% drop cross-dataset | MIMIC-III / C2017 holdout eval |
| TR-1.1 Holdout by subject | `splitter`, `get_dataloaders` |
| TR-2.3 PCA / channel strip | `pca_reduction.py`, tests |

---

## Current Status

- **Data & preprocessing:** PhysioNet ingestion, 250 Hz resampling, HRV pipeline, artifact handling, phase splits.
- **Models & training:** Hybrid Ensemble, two-phase training, precomputed cache, augmentation.
- **Evaluation:** Test split, MIMIC-III, Challenge 2017, threshold analysis, ROC/confusion plots.
- **Deferred:** LSL streamer, `live_inference.py`, Sparrow API + watchdog, latency profiler, 24 h HIL test.

Update `config.yaml` (watchdog, artifact thresholds) per Sparrow/FDA limits and validation. See `PLAN.md` for the latest step list and “Resume From Here.”
