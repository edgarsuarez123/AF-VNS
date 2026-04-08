# Tinnitus aVNS — AI Infrastructure Report
**Branch:** `feature/tinnitus-avns` | **Date:** 2026-04-08 | **Phase:** SBIR Phase I

---

## 1. Overview

This document describes the AI infrastructure powering the tinnitus auricular vagus nerve stimulation (aVNS) pipeline — compute environment, software stack, model architecture, data flow, training infrastructure, and real-time inference design.

The system implements a **tri-fold synchronized stimulation gate**: a stimulation pulse fires only when cardiac diastole, respiratory exhalation, and EDA arousal-in-band are simultaneously detected. Three separate ML models power these three gates.

---

## 2. Compute Environment

| Component | Value |
|-----------|-------|
| OS | Windows 10 Pro (64-bit) |
| CPU | Intel/AMD (host inference) |
| GPU | CUDA-capable (training only) |
| Python | 3.11 (`.venv/Scripts/python.exe`) |
| Shell | PowerShell (SSH via Tailscale VPN from mobile) |
| Remote access | Termius → SSH → `100.87.72.29` |

**Background process management:** No tmux available on Windows. Long-running training jobs launched via `Start-Process -WindowStyle Hidden` with stdout/stderr redirected to log files. This allows training to survive SSH session disconnects — critical given the mobile-primary workflow.

```powershell
Start-Process -WindowStyle Hidden ".venv\Scripts\python.exe" `
  -ArgumentList "-m src.training.tinnitus_phase_train --model-section phase_model_v2" `
  -WorkingDirectory "C:\Users\Edgar\AF VNS" `
  -RedirectStandardOutput "train_v2.log" `
  -RedirectStandardError "train_v2_err.log"
```

Progress monitoring: `Get-Content train_v2_err.log -Tail 10`

---

## 3. Software Stack

### 3.1 Core Dependencies

| Library | Role |
|---------|------|
| `torch` (PyTorch) | CNN training and inference |
| `sklearn` | Gradient boosting arousal classifier |
| `numpy` | Array operations, windowing, label generation |
| `scipy` | Signal resampling (`scipy.signal.resample`), Butterworth filter (`sosfilt`) |
| `neurokit2` | PPG peak detection, EDA decomposition (cvxEDA), respiratory peak finding |
| `wfdb` | BIDMC PhysioNet record parsing (WFDB format) |
| `joblib` | Classifier serialization (`.pkl` checkpoint) |
| `pyyaml` | Config loading |
| `matplotlib` | Validation result plots |

### 3.2 Project Structure

```
C:\Users\Edgar\AF VNS\
├─ config_tinnitus.yaml          ← single source of truth for all hyperparameters
├─ src/
│   ├─ data/
│   │   └─ tinnitus_parsers.py   ← BIDMC + WESAD ingestion, TinnitusRecordDict schema
│   ├─ features/
│   │   ├─ ppg_filter.py         ← Butterworth bandpass denoising
│   │   ├─ ppg_phase_labels.py   ← diastole detection (systolic peak + dicrotic notch)
│   │   ├─ ppg_resp.py           ← PPG-derived respiration (RIIV / RIFV / baseline wander)
│   │   └─ eda.py                ← EDA tonic/phasic decomposition, arousal calibration
│   ├─ models/
│   │   ├─ phase_detector.py     ← 1D CNN architecture + factory
│   │   ├─ arousal_gate.py       ← EDA ring buffer, rule-based + classifier gating
│   │   ├─ arousal_classifier.py ← GBT wrapper + factory
│   │   └─ tinnitus_closed_loop.py ← tri-fold pipeline, streaming inference
│   └─ training/
│       ├─ tinnitus_precompute_cache.py   ← offline label generation + caching
│       ├─ tinnitus_precompute_arousal.py ← EDA feature extraction for GBT
│       ├─ tinnitus_phase_train.py        ← CNN training loop
│       ├─ tinnitus_train_arousal.py      ← GBT training script
│       └─ tinnitus_replay_validation.py  ← offline pipeline simulation
├─ models/
│   ├─ checkpoints/              ← trained model files
│   └─ artifacts/                ← caches, splits, replay results
├─ tests/                        ← 462+ automated tests
└─ data/raw/tinnitus avns/       ← raw BIDMC + WESAD signal files
```

### 3.3 Configuration Management

All hyperparameters live in `config_tinnitus.yaml`. No magic numbers in code. Multiple named sections allow A/B experiments without code changes:

```yaml
# Example: running 3 training variants without touching Python
phase_training:          # default
  task_weight_dia: 0.5

phase_training_ls:       # F14 label smoothing experiment
  task_weight_dia: 0.5
  label_smoothing: 0.1

phase_training_dia_moderate:   # F14 diastole-bias experiment
  task_weight_dia: 0.6
  label_smoothing: 0.1
```

Scripts accept `--model-section` / `--config-section` flags to select which YAML block to use at runtime.

---

## 4. Data Ingestion Layer

### 4.1 Schema

All records, regardless of source, are normalized to `TinnitusRecordDict`:

```python
TinnitusRecordDict = TypedDict("TinnitusRecordDict", {
    "subject_id":   str,
    "session_id":   str,
    "ppg_signal":   np.ndarray,   # 1D, float64, at ppg_fs
    "ppg_fs":       float,        # Hz
    "resp_signal":  Optional[np.ndarray],   # reference resp if available
    "resp_fs":      Optional[float],
    "eda_signal":   Optional[np.ndarray],   # 4 Hz
    "eda_fs":       Optional[float],
    "temp_signal":  Optional[np.ndarray],   # 4 Hz, wrist skin temp (WESAD only)
    "temp_fs":      Optional[float],
    "label":        int,          # 0=baseline, 1=stress
    "condition":    str,          # "baseline", "stress", "amusement", etc.
})
```

`validate_tinnitus_record()` checks all required keys before any record enters the pipeline.

### 4.2 Parser: BIDMC

- Format: WFDB (PhysioNet) binary records + header files
- Extraction: PLETH channel (PPG) + RESP channel (impedance pneumography)
- Native rate: 125 Hz (both signals)
- No resampling required
- 53 records parsed; all pass schema validation

### 4.3 Parser: WESAD

- Format: Python pickle per subject (`S{N}/S{N}.pkl`)
- Extraction:
  - `signal["wrist"]["BVP"]` → PPG @ 64 Hz
  - `signal["wrist"]["EDA"]` → EDA @ 4 Hz
  - `signal["wrist"]["TEMP"]` → Skin temperature @ 4 Hz
  - `signal["chest"]["Resp"]` → Chest belt @ 700 Hz
- Segmentation: Split by contiguous label epochs; skip undefined (label=0) and epochs <30s
- PPG resampled 64→125 Hz via `scipy.signal.resample` at cache generation time
- 15 subjects → 75 epoch records

---

## 5. Feature Engineering Layer

### 5.1 PPG Denoising

```
Input:  raw PPG (any fs ≥ 64 Hz)
Filter: Butterworth bandpass 0.5–8 Hz
        Order: 4
        Form:  second-order sections (SOS) — numerically stable at narrow bands
Output: filtered PPG, same shape and fs
```

Why SOS over ba-form: At narrow bandpass ratios (0.5–8 Hz relative to 125 Hz Nyquist of 62.5 Hz), the ba-form transfer function has near-cancelling numerator/denominator coefficients causing floating-point instability. SOS chains second-order biquad sections instead.

### 5.2 Diastole Label Generation

```
Input:  clean PPG, fs
Step 1: neurokit2.ppg_findpeaks() → systolic peak indices
Step 2: For each beat:
          search_start = peak_i + 0.30 × beat_interval
          search_end   = peak_{i+1} - 100 ms
          notch_idx    = argmin(clean_ppg[search_start:search_end])
Step 3: Sample-level labels:
          diastole = 1 for notch_idx → (next_peak - 100ms guard)
          systole  = 0 elsewhere
Step 4: Majority vote within 200ms frames → 5 Hz label array
Output: (n_frames,) int8 array, 5 Hz
```

### 5.3 Exhalation Label Generation

Three methods, configurable per run:
- **RIIV:** PPG peak amplitudes interpolated → bandpass 0.1–0.5 Hz → peaks = end-expiration
- **RIFV:** Inter-beat intervals interpolated → bandpass 0.1–0.5 Hz → RSA-based
- **Baseline wander:** Direct PPG lowpass → bandpass 0.1–0.5 Hz

All output: `(n_frames,)` int8 at 5 Hz, matching diastole label interface exactly.

### 5.4 EDA Feature Extraction

Per 60-second window (30s or 15s hop):

| Feature | Computation |
|---------|------------|
| `tonic_scl_mean` | `mean(tonic_SCL)` in window |
| `tonic_scl_std` | `std(tonic_SCL)` in window |
| `phasic_mean` | `mean(phasic_SCR)` in window |
| `max_scr_amplitude` | `max(phasic_SCR)` in window |
| `scr_rate` | SCR peak count / window_sec × 60 |
| `skin_temp_mean` | `mean(temp_signal)` in window (v2 only) |

Decomposition: `neurokit2.eda_phasic(method="cvxeda")` with highpass fallback.

---

## 6. Caching Infrastructure

### 6.1 Design

Raw signals are processed once offline; results cached as numpy arrays. Training reads only cached files — no signal processing at training time.

```
tinnitus_precompute_cache.py
  input:  raw records (TinnitusRecordDict)
  output: per-record .npy files in cache_dir/

Per record:
  {record_id}_ppg.npy       shape: (n_samples, 1)       or (n_samples, 3) for v2
  {record_id}_labels.npy    shape: (n_frames, n_tasks)
  {record_id}_quality.npy   shape: (n_frames, n_tasks)  NaN fractions

manifest.json: maps record_ids → file paths + metadata
tinnitus_phase_split.json: train/val/test subject lists (70/15/15)
```

### 6.2 Cache Inventory

| Cache | Contents | Windows | Channels |
|-------|----------|---------|----------|
| `cache_tinnitus_phase/` | 1-ch PPG + diastole/exh labels | 350,624 | 1 |
| `cache_tinnitus_phase_v2/` | 3-ch PPG+VPG+APG + labels | 350,624 | 3 |
| `cache_tinnitus_exh/` | 2-ch PPG+RIIV, 6s windows | ~same subjects | 2 |
| `cache_tinnitus_arousal/` | 5-feature EDA vectors | 1,398 | 5 |
| `cache_tinnitus_arousal_v2/` | 6-feature EDA vectors | ~2,800 (15s hop) | 6 |

### 6.3 Subject-Level Split

```json
{
  "train": ["S2", "S3", "S4", "S6", "S8", "S10", "S13", "S14", "S17", ...bidmc_ids...],
  "val":   ["S5", "S11", "S15", ...],
  "test":  ["S7", "S9", "S16", ...]
}
```

No subject appears in more than one partition — prevents data leakage from subject-specific physiological signatures.

---

## 7. Model Architecture

### 7.1 Phase Detector CNN

**Purpose:** Multi-task binary classification — given 2s of PPG, output per-frame probability of diastole AND exhalation.

```
Input:  (B, C, 250)   C=1 (v1) or C=3 (v2: PPG + VPG + APG)

Backbone:
  Conv1d(C→16,  k=7, stride=5, bias=False) → BN → ReLU
  Conv1d(16→32, k=5, stride=2, bias=False) → BN → ReLU
  Conv1d(32→48, k=3, stride=2, bias=False) → BN → ReLU
  AdaptiveAvgPool1d(10)

Head:
  Dropout(0.1)
  Conv1d(48→n_tasks, k=1)
  Permute(0, 2, 1)

Output: (B, 10, 2)   10 frames × [diastole_logit, exhalation_logit]
Parameters: ~7,500 (v1), ~8,200 (v2)
```

**Why this architecture:**
- Strided convolutions replace pooling — fewer hyperparameters, learnable downsampling
- AdaptiveAvgPool forces exactly 10 output frames regardless of input length variation
- 1×1 conv head is equivalent to a per-frame linear classifier — minimal overfitting risk at low data volume
- Total 7,500 parameters fits comfortably in embedded microcontroller RAM (<64 KB)

### 7.2 Dedicated Exhalation Detector

```
Input:  (B, 2, 750)   2 channels (PPG + RIIV), 6s at 125 Hz
Output: (B, 30, 1)    30 frames × [exhalation_logit]
Same backbone; longer window captures a full respiratory cycle at 10 bpm minimum
```

### 7.3 Arousal Classifier

```
sklearn.GradientBoostingClassifier
  n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8

Input:  (N, 5) or (N, 6) EDA feature vectors
Output: (N,) binary labels + predict_proba() for soft thresholding

Preprocessing (fit on train set, applied to val/test):
  StandardScaler
  NaN imputation: per-column training median
```

**Why GBT over neural net:** 1,398 training windows is too small for reliable neural net training. GBT is sample-efficient, requires no architecture search, and provides interpretable feature importances. Training time <5 seconds.

### 7.4 Checkpoint Inventory

| File | Model | Accuracy |
|------|-------|----------|
| `tinnitus_phase_detector.pth` | v1 CNN (1-ch) | dia=72%, exh=51% |
| `tinnitus_phase_detector_v2.pth` | v2 CNN (3-ch VPG/APG) | dia=75.6%, exh=54% |
| `tinnitus_exh_detector.pth` | Dedicated exh (2-ch, 6s) | exh=54.6% |
| `tinnitus_phase_ls.pth` | F14 label smoothing | dia=72%, exh=51% |
| `tinnitus_phase_dia_mod.pth` | F14 diastole-bias | dia=72%, exh=51% |
| `tinnitus_phase_exh_mod.pth` | F14 exhalation-bias | dia=72%, exh=51% |
| `arousal_classifier.pkl` | v1 GBT (5-feat) | acc=85.7%, AUROC=0.927 |
| `arousal_classifier_v2.pkl` | v2 GBT (6-feat, 15s hop) | — |

---

## 8. Training Infrastructure

### 8.1 Phase Detector Training Loop

```
PhaseDetectorDataset
  → reads manifest.json
  → loads {record_id}_ppg.npy + {record_id}_labels.npy
  → yields (B, C, 250) windows with 200ms stride on-the-fly (no full preload)

Training:
  optimizer: Adam(lr=1e-3)
  loss:      BCEWithLogitsLoss per task
  total:     task_weight_dia × loss_dia + task_weight_exh × loss_exh
  scheduler: none (early stopping handles LR implicitly)
  early stop: patience=15 (tracks avg_acc = mean(dia_acc, exh_acc) on val set)
  grad clip:  max_norm=1.0

Validation (every epoch):
  run_phase_validation() → dia_acc, exh_acc, avg_acc, val_loss
  Checkpoint: save if avg_acc improves

Output:
  best checkpoint → models/checkpoints/{name}.pth
  training curves → models/artifacts/training_curves.png
```

### 8.2 Arousal Classifier Training

```
tinnitus_train_arousal.py
  → loads cache_tinnitus_arousal/ feature vectors
  → splits by tinnitus_arousal_split.json
  → fits StandardScaler on train
  → fits GradientBoostingClassifier on train
  → evaluates on val + test: accuracy, F1, AUROC, confusion matrix
  → saves classifier + scaler → arousal_classifier.pkl
  → saves metrics → arousal_classifier_metrics.json
```

### 8.3 Experiment Management Pattern

Each training variant is:
1. Named config section added to `config_tinnitus.yaml`
2. Launched with `--model-section` or `--config-section` flag
3. Output to a differently-named checkpoint (e.g., `tinnitus_phase_ls.pth`)
4. Results logged to `TINNITUS_PLAN.md` + `progress.txt`
5. Committed to git with descriptive message

No external experiment tracking tool (MLflow, W&B) — YAML + git + plan files serve this purpose given the solo development context.

---

## 9. Real-Time Inference Pipeline

### 9.1 Streaming Architecture

`TinnitusClosedLoopPipeline` is designed for streaming operation — the caller drives timing by calling `feed()` at whatever rate the hardware delivers samples.

```python
pipeline = build_tinnitus_closed_loop_pipeline("config_tinnitus.yaml", device="cpu")
pipeline.calibrate_eda(baseline_eda, eda_fs)   # one-time calibration

# Streaming loop (hardware calls this at 125 Hz for PPG, 4 Hz for EDA):
while True:
    ppg_chunk  = acquire_ppg(chunk_size=125)    # 1 second of PPG
    eda_chunk  = acquire_eda(chunk_size=4)       # 1 second of EDA
    temp_chunk = acquire_temp(chunk_size=4)      # 1 second of temp
    events = pipeline.feed(ppg_chunk, eda_chunk, temp_samples=temp_chunk)
    for event in events:
        trigger_stimulator(event.amplitude, event.frequency, event.pulse_width)
```

### 9.2 Fast Path Detail

Runs every `inference_stride_ms=100` ms (10 Hz):

```
1. Extract last 250 samples from PPG ring buffer
2. denoise_ppg() → clean window
3. [v2 only] compute VPG = np.diff(clean, prepend) / (1/fs)
              compute APG = np.diff(VPG, prepend) / (1/fs)
              z-normalize each channel independently
              stack → (1, 3, 250) tensor
4. PhaseDetector(tensor) → (1, 10, 2) logits
5. sigmoid(logits) → [dia_probs (10,), exh_probs (10,)]
6. Consecutive-frame gate:
     dia_window = dia_probs[-N:]     N = consecutive_frames_required = 3
     exh_window = exh_probs[-N:]
     dia_ok  = all(dia_window > 0.65)
     exh_ok  = all(exh_window > 0.50)
7. Tri-fold check:
     if dia_ok AND exh_ok AND arousal_gate.is_in_band():
         fire TinnitusStimEvent
```

### 9.3 Slow Path Detail

Runs every `slow_window_sec=60` seconds:

```
1. Extract last 7500 samples (60s of PPG)
2. denoise_ppg()
3. get_rr_intervals(ppg, fs=125, signal_type="ppg")
4. correct_rr_intervals(rr)  ← reject outliers >60% deviation
5. autonomic_state.compute(rr) → {lf_hf_ratio, norm_hf, sampen, dfa_alpha1}
6. stim_recommender.recommend(state) → update amplitude/frequency targets
```

### 9.4 Latency Budget

| Step | Budget | Measured |
|------|--------|---------|
| PPG denoising (250 samples) | <5 ms | ~0.5 ms |
| CNN forward pass (CPU) | <10 ms | ~2 ms |
| Gate logic | <1 ms | <0.5 ms |
| EDA update (per chunk) | <10 ms | ~3 ms |
| **Total fast path** | **<50 ms** | **~3–4 ms** |

Meets SBIR <50 ms timing accuracy requirement with ~12× margin.

### 9.5 EDA Arousal Gate

```
ArousalGate
  ├─ ring_buffer: deque(maxlen=240)     ← 60s × 4 Hz
  ├─ calibrate(eda, fs)                 ← fit threshold from baseline window
  ├─ update(eda_chunk, fs)              ← append samples, recompute arousal state
  │   ├─ decompose_eda()
  │   ├─ compute_arousal_in_band()
  │   └─ [if classifier]: compute 5/6-feature vector → predict_proba()
  ├─ is_in_band()                       ← returns bool or raises if uncalibrated
  │   ├─ [classifier mode]: return predict_proba()[1] < threshold
  │   └─ [rule mode]: return low_thresh ≤ SCL ≤ high_thresh
  └─ [F21] sliding recalibration:
      every 60s: EMA blend (α=0.3) of new 5-min baseline estimate
```

---

## 10. Validation Infrastructure

### 10.1 Unit + Integration Tests

462+ automated tests, organized per module:

```
tests/
├─ test_tinnitus_parsers.py         (36 tests — parsing, schema, record counts)
├─ test_ppg_filter.py               (9 tests — bandpass behavior, edge cases)
├─ test_ppg_phase_labels.py         (17 tests — peak detection, notch search)
├─ test_ppg_resp.py                 (31 tests — RIIV/RIFV/baseline, interface match)
├─ test_eda.py                      (35 tests — decomposition, calibration, WESAD validation)
├─ test_arousal_gate.py             (8 tests — buffer, calibration, is_in_band)
├─ test_arousal_classifier.py       (11 tests — fit, predict, NaN, checkpoint I/O)
├─ test_phase_train.py              (3 tests — config, forward pass shape, factory)
├─ test_tinnitus_closed_loop.py     (15 tests — gating, latency, factory, v2)
└─ test_tinnitus_replay_validation.py (22 tests — chunking, metrics, ground truth)
```

Run: `.venv/Scripts/python -m pytest tests/ -v`

### 10.2 Offline Replay Validation

`tinnitus_replay_validation.py` streams all 75 WESAD epoch records through the full closed-loop pipeline, comparing stimulation events to independently computed ground-truth labels.

```
For each subject × condition:
  1. Build pipeline (fresh model + arousal gate per subject)
  2. Calibrate EDA from baseline epoch
  3. Chunk records into 1s streaming windows
  4. feed() each chunk → collect TinnitusStimEvents
  5. Compare event timestamps to GT label grid
  6. Compute: precision, recall, stim rate, per-gate breakdown

Output:
  models/artifacts/replay_validation/wesad_replay_results.json
  models/artifacts/replay_validation/wesad_replay_summary.png
```

Supports `--use-v2` flag (loads v2 checkpoints) and `--compare` flag (prints F16 vs F22 table).

---

## 11. Infrastructure Decisions & Trade-offs

| Decision | Choice | Trade-off |
|----------|--------|-----------|
| Precompute all labels offline | ✓ Done | Disk space vs. training speed — cache is ~3 GB but training is 10× faster |
| Single config YAML | ✓ Done | Verbosity vs. reproducibility — every experiment is fully specified |
| No external ML tracking (W&B, MLflow) | Git + plan files | Simpler solo workflow vs. less queryable experiment history |
| CPU inference | ✓ Done | ~2 ms on CPU — no GPU required in deployment device |
| No threading in pipeline | Caller-driven streaming | Simpler correctness guarantees vs. requires external scheduler |
| joblib for GBT checkpoint | ✓ Done | Sklearn native format — fast, no torch dependency for arousal classifier |
| SOS Butterworth filter | ✓ Done | Numerically stable vs. ba-form at narrow bands |
| Subject-level train/val/test split | ✓ Done | Prevents leakage vs. smaller effective dataset |

---

*Generated from `feature/tinnitus-avns` — commit `d501233`*
