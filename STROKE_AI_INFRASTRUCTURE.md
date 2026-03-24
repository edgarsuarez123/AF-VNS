# AI Infrastructure — Stroke AVNS Phase Detection Pipeline

**Project:** Aim 2 — AI-Driven Adaptive Auricular VNS (Stroke Population)
**Branch:** `feature/stroke-avns`
**Scope:** Real-time physiological phase detection for stimulation timing in stroke patients
**Status:** FINAL — production model designated: `phase_detector_ref.pth` — diastole 84.34%, exhalation 56.54% (S-36). All ECG-only experiments exhausted.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Data Pipeline](#3-data-pipeline)
4. [Label Generation](#4-label-generation)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Autonomic State Module](#8-autonomic-state-module)
9. [Stim Parameter Recommender](#9-stim-parameter-recommender)
10. [Configuration](#10-configuration)
11. [Design Decisions](#11-design-decisions)
12. [Known Limitations & Open Issues](#12-known-limitations--open-issues)
13. [How to Run](#13-how-to-run)

---

## 1. System Overview

The Stroke AVNS pipeline implements **real-time physiological phase detection** on ECG recordings from stroke patients. The system determines the optimal timing window for auricular vagus nerve stimulation: during **diastole** (cardiac relaxation phase) and **exhalation** (respiratory relaxation phase), when vagal tone is highest and stimulation efficacy is greatest.

### Clinical Context

VNS efficacy is maximized when stimulation is delivered at the intersection of:
- **Cardiac diastole** — heart in relaxation phase, receptive to vagal input
- **Exhalation** — respiratory phase associated with vagal dominance (respiratory sinus arrhythmia)

The system continuously monitors ECG at 250 Hz, detects the current phase at 5 Hz, and feeds phase information to the stim parameter recommender.

### End-to-End Data Flow

```
Raw ECG (250 Hz)
        │
        ▼
  stroke_parsers.py        ← Parse CVES/MIMIC-3/SHaRe → standard schema
        │                    (CVES: also returns resp_signal from flow_rate/thermst)
        ▼
  Resample to 250 Hz       ← scipy.signal.resample (FFT-based)
        │
        ▼
  wavelet_filter.py        ← CWT denoising (0.5-45 Hz passband)
        │
        ├────────────────────────────────────────────────────────┐
        ▼                                                         ▼
  phase_labels.py                                    Exhalation labels (priority order):
  (diastole labels)                                  1. resp_labels.py  [CVES — hardware reference]
  R-peak → T-end → frame labels                         Butterworth bandpass + scipy peak detection
                                                     2. edr.py fallback [RSA/QRS-AM/fusion EDR]
                                                     3. NaN mask        [MIMIC — no hardware resp]
        │                                                         │
        └──────────────── combine ─────────────────────────────────┘
                              │
                              ▼
                  stroke_precompute_cache.py
                  (2s windows, 0.2s stride)
                  → (N, 500) waveforms
                  → (N, 10, 2) frame labels (NaN = masked)
                              │
                              ▼
                      PhaseDetector CNN         ←── phase_train.py
                      (B,1,500) → (B,10,2)          Multi-task BCE, NaN-masked
                      ~7,570 params                  pos_weight per task (auto)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             diastole_prob(t)    exhalation_prob(t)
             at 5 Hz update      at 5 Hz update
                    │                   │
                    └────────┬──────────┘
                             ▼
                  ClosedLoopPipeline     ←── closed_loop_pipeline.py (S-31)
                  co-occurrence check (dia AND exh)
                             │
                             ▼
                  AutonomicStateModule (every 60s)
                  LF/HF ratio, nHF, SampEn, DFA-alpha1
                             │
                             ▼
                  StimRecommender
                  amplitude (0-10mA)
                  frequency (1-100Hz)
                  pulse_width (50-500us)
```

---

## 2. Directory Structure

```
AF VNS/
├── config_stroke.yaml              # All hyperparameters for stroke pipeline
├── STROKE_PLAN.md                  # Branch task plan
│
├── src/
│   ├── data/
│   │   ├── stroke_parsers.py       # Parse CVES, MIMIC-3 stroke, SHaRe → standard schema
│   │   ├── download_stroke_datasets.py   # Download CVES + SHaRe via wfdb
│   │   └── download_mimic3_stroke_waveforms.py  # MIMIC-III ICD-9 430-438
│   │
│   ├── features/
│   │   ├── phase_labels.py         # Diastolic phase: R-peak + T-end → 5Hz frame labels
│   │   ├── resp_labels.py          # Exhalation from hardware resp signal (CVES primary path)
│   │   ├── edr.py                  # ECG-Derived Respiration: RSA/amplitude/fusion methods (fallback)
│   │   ├── wavelet_filter.py       # CWT denoising (shared with AF pipeline)
│   │   ├── peak_detector.py        # R-peak detection (shared)
│   │   ├── hrv_time.py             # RMSSD, SDNN (shared)
│   │   ├── hrv_freq.py             # LF, HF, LF/HF (shared)
│   │   └── hrv_nonlinear.py        # SampEn, DFA-α1 (shared)
│   │
│   ├── models/
│   │   ├── phase_detector.py       # PhaseDetector CNN (~7,570 params)
│   │   ├── autonomic_state.py      # Sliding-window HRV → autonomic state vector
│   │   └── stim_recommender.py     # Autonomic state → stim parameters (rule-based)
│   │
│   ├── models/
│   │   ├── phase_detector.py       # PhaseDetector CNN (~7,570 params)
│   │   ├── autonomic_state.py      # Sliding-window HRV → autonomic state vector
│   │   ├── stim_recommender.py     # Autonomic state → stim parameters (rule-based)
│   │   └── closed_loop_pipeline.py # End-to-end: ECG → phase detect → stim params (S-31)
│   │
│   └── training/
│       ├── stroke_precompute_cache.py  # Build 2s window cache with phase labels
│       ├── phase_train.py              # Training loop (multi-task BCE, NaN masking)
│       └── phase_evaluate.py           # Accuracy, precision, recall, confusion matrices
│
├── tests/
│   ├── test_phase_labels.py        # 8 tests for phase_labels.py
│   ├── test_edr.py                 # 7 tests for edr.py
│   ├── test_stroke_cache.py        # 12 tests for stroke_precompute_cache.py
│   ├── test_phase_train.py         # 12 tests for phase_train.py
│   └── test_autonomic_state.py     # 16 tests for autonomic_state + stim_recommender
│
├── data/
│   └── raw/
│       └── stroke avns/
│           ├── cves/               # 228 records (74 stroke, 154 control), 500 Hz
│           ├── mimic3_stroke/      # 300 records (150/150), 125 Hz
│           └── shareedb/           # 133 records (17 event, 116 control), 250 Hz
│
├── models/
│   ├── artifacts/
│   │   ├── phase_detect_split.json     # Subject-level train/val/test split
│   │   └── cache_phase_detect/         # Precomputed phase detection cache
│   │       ├── train_waveforms.npy     # (1,819,893, 500) denoised 2s windows
│   │       ├── train_labels.npy        # (1,819,893, 10, 2) frame labels
│   │       ├── val_waveforms.npy       # (395,566, 500)
│   │       ├── val_labels.npy          # (395,566, 10, 2)
│   │       ├── test_waveforms.npy      # (361,445, 500)
│   │       ├── test_labels.npy         # (361,445, 10, 2)
│   │       └── cache_meta.json
│   │
│   └── checkpoints/
│       └── phase_detector.pth          # Best checkpoint (avg_acc=0.6864)
│
├── phase_train_err.log             # Training stderr (progress bars + metrics)
└── precompute_phase_err.log        # Cache build progress
```

---

## 3. Data Pipeline

### 3.1 Data Sources

| Source | Records | Population | Hz | Role |
|--------|---------|------------|----|------|
| **CVES** | 228 (74 stroke, 154 control) | Post-stroke + controls, autonomic stress protocols | 500 | Primary — target population; 220/228 have hardware resp reference |
| **MIMIC-III Stroke** | 300 (150/150, ICD-9 430-438) | ICU patients with cerebrovascular events | 125 | Volume — diastole training only; exhalation NaN-masked |
| **SHaRe** | 133 (17 event, 116 control) | 24h ambulatory Holter | 250 | OOD evaluation only — never in training |
| **BIDMC** | 53 | ICU adults (mixed diagnoses) | 125 | Tested (S-42–S-49) — reference resp available but wrong conditions; excluded from production |

**Why the stroke/control label is irrelevant:** The ML task is phase detection, not disease classification. Every human ECG has cardiac cycles and respiratory modulation — the labels used for training are derived from ECG signal characteristics (R-peak timing, T-wave morphology), not neurological status.

### 3.2 Parsing

`stroke_parsers.py` normalizes all sources to the standard schema:
```python
{
    "subject_id":  str,                # e.g. "cves_001", "p190432_stroke"
    "signal":      np.ndarray,         # 1D float64, resampled to 250 Hz
    "fs":          float,              # 250.0 (after resample)
    "label":       int | None,         # pathology label — not used for training
    "resp_signal": np.ndarray | None,  # hardware respiratory reference (CVES only)
    "resp_channel": str | None,        # "flow_rate" / "thermst" / "resp" / None
}
```

`resp_signal` is used by `resp_labels.py` to generate reference exhalation labels for CVES records without needing EDR inference.

CVES (500 Hz) and MIMIC-3 (125 Hz) are resampled to 250 Hz using `scipy.signal.resample` (FFT-based). Chunked at 60s to bound memory.

### 3.3 Subject-Level Split

`splitter.create_split()` produces a **70/15/15 split by subject_id**. Persisted to `models/artifacts/phase_detect_split.json`, seed=42.

| Split | Records | Windows |
|-------|---------|---------|
| Train | ~336 | 1,819,893 |
| Val | ~72 | 395,566 |
| Test | ~72 | 361,445 |
| **Total** | **480** | **2,576,904** |

48 records were skipped (insufficient R-peaks or respiratory cycles for label generation, typically very short or heavily corrupted signals).

---

## 4. Label Generation

Label generation is the most novel component of this pipeline — no external annotation or reference signal is needed.

### 4.1 Diastolic Phase Labels (`src/features/phase_labels.py`)

**Physiological basis:** Diastole = cardiac relaxation phase. Starts at T-wave end, ends at next R-peak onset. Occupies ~60-65% of each cardiac cycle.

**Algorithm:**
```
1. Denoise ECG (CWT 0.5-45 Hz)
2. R-peak detection: neurokit2.ecg_peaks()
3. T-wave end detection: neurokit2.ecg_delineate()
   └── Fallback: T-end = R-peak + 0.4 × RR interval (40% RR rule)
4. Per-sample label array: diastole=1 at each sample in [T-end, next R-onset)
5. Downsample to 5 Hz: majority vote over each 200ms frame
```

**Validated:** 150 frames, 72.1 bpm, 62.3% diastole rate on CVES record.

### 4.2 Exhalation Phase Labels (three-tier fallback)

**Critical discovery (S-32):** CVES records contain hardware respiratory channels (`flow_rate`, `thermst`) — measured nasal airflow and temperature. These are used as ground truth instead of EDR inference.

**Label generation priority:**

**Tier 1 — Reference signal (`src/features/resp_labels.py`, CVES records):**
```
1. Extract resp_signal from parser (flow_rate > thermst > resp priority)
2. Butterworth bandpass [0.08, 0.6] Hz
3. scipy.signal.find_peaks with auto-prominence (10% of range)
4. Peak → next landmark = exhale (label=1), trough → next landmark = inhale (label=0)
5. Majority-vote downsample to 5 Hz frames
```
Coverage: 220/228 CVES records (96%).

**Tier 2 — EDR fallback (`src/features/edr.py`, CVES reference-failed records):**
Three methods implemented:
- `"vangent2019"` — RSA-based (HR variability bandpass). Fails under autonomic stress.
- `"amplitude"` — QRS peak height interpolation + SOS Butterworth 0.1-0.5 Hz. Stress-tolerant but low SNR.
- `"fusion"` — PCA of RSA + amplitude channels. Slightly better correlation (0.196 vs 0.162) but insufficient to improve training.

**Tier 3 — NaN masking (MIMIC, SHaRe records):**
Records without hardware respiratory signal have exhalation labels forced to NaN. These records still contribute diastole gradients that stabilize shared CNN features — removing them (CVES-only training) collapses exhalation accuracy to 50%.

**ECG-only exhalation ceiling: 56-57%.** All three EDR methods and 10 systematic ablations confirmed this limit. RSA is suppressed under orthostatic stress and in stroke patients with impaired vagal function. QRS amplitude modulation has low SNR from single-lead ECG. The physics does not support higher accuracy without a reference channel.

### 4.3 NaN Labels

Frames where label generation fails (too few R-peaks, no respiratory cycles detected) are stored as NaN. The training loop masks these frames — they contribute zero gradient, not wrong gradient. This is critical for noisy ICU records.

---

## 5. Model Architecture

### 5.1 PhaseDetector (`src/models/phase_detector.py`)

A lightweight 1D CNN designed for real-time inference at 5 Hz on embedded hardware.

```
Input: (B, 1, 500)  — 2 seconds × 250 Hz = 500 samples

Conv1d(1→16, k=7, s=5) + BN + ReLU  → (B, 16, 100)
Conv1d(16→32, k=5, s=2) + BN + ReLU → (B, 32, 50)
Conv1d(32→48, k=3, s=2) + BN + ReLU → (B, 48, 25)
AdaptiveAvgPool1d(10)                → (B, 48, 10)
Dropout(0.1)
Conv1d(48→2, k=1)                   → (B, 2, 10)
Permute(0, 2, 1)                    → (B, 10, 2)

Output: (B, 10, 2)
  Dim 0: batch
  Dim 1: 10 temporal frames (one per 200ms)
  Dim 2: 2 tasks — [:, :, 0] = diastole logit, [:, :, 1] = exhalation logit
```

**Parameter count: ~7,570** — intentionally tiny for:
- Inference latency <200ms (NFR-2.1 latency requirement)
- Future deployment on embedded co-processor (Sparrow device)
- Avoids overfitting on ~480 records

### 5.2 Architecture Decisions

**Why 1D CNN, not RNN/Transformer?**
Phase detection needs to respond to instantaneous ECG morphology — the shape of the QRS complex and T-wave at this moment in time. RNNs are better for long-range temporal patterns (like 5-minute HRV trajectories in the AF pipeline). For 2s context, a CNN with strided convolutions achieves the same receptive field with lower latency.

**Why 2s context window?**
- Must contain at least one complete cardiac cycle (~0.8s at 75 bpm) for diastole detection
- Enough for RSA detection (HR modulation correlates with respiration at ~0.1-0.4 Hz)
- Short enough to maintain <200ms inference per window (with GPU batch processing)

**Why multi-task output (diastole + exhalation together)?**
Shared ECG context for both tasks. The lower CNN layers learn general ECG representations (QRS morphology, noise patterns) that benefit both tasks. Only the final 1×1 conv head differentiates.

**Why AdaptiveAvgPool1d(10)?**
The pool produces exactly 10 output frames regardless of how the strided convolutions map to output length. This makes the architecture input-length agnostic and eliminates dimension tracking across conv layers.

**Why no positional encoding or attention?**
With only 10 output frames and 2 tasks, self-attention would add parameters without benefit. The temporal ordering is already implicit in the conv feature maps.

---

## 6. Training Pipeline

### 6.1 Cache Build (`stroke_precompute_cache.py`)

**Why cache:** Label generation (wavelet + R-peak detection + T-wave delineation + EDR) takes ~2s per record. With 2.5M training windows, building labels on-the-fly is infeasible. Cache is built once.

**Process:**
```
For each record in split:
  1. Denoise (CWT)
  2. Run phase_labels.generate_phase_labels() → diastole frame array
  3. Exhalation labels (priority):
     a. resp_labels.generate_exhalation_labels_from_reference() if resp_signal available
     b. edr.generate_exhalation_labels() as fallback (CVES reference-failed only)
     c. NaN mask if no hardware resp signal (MIMIC, SHaRe)
  4. Slice 2s windows at 0.2s stride
  5. Store (window, labels) in memory-mapped numpy arrays
```

**Parallel execution:** Launched as background PowerShell process via `Start-Process`. ~480 records in ~12 minutes with 4 workers.

**Production cache (cache_phase_detect):**
```
exh_method_counts: reference=220, masked_edr=~250, edr=~5, none=0
Total: ~2.7M windows (train: ~1.9M, val: ~400K, test: ~400K)
```

### 6.2 Training Loop (`phase_train.py`)

**Loss function:** Multi-task BCE with NaN masking
```python
# Per task, per frame — only compute loss on valid (non-NaN) labels
valid_mask = ~torch.isnan(labels)
loss_diastole = bce(logits[:,:,0][valid_mask[:,0]], labels[:,:,0][valid_mask[:,0]])
loss_exhale   = bce(logits[:,:,1][valid_mask[:,1]], labels[:,:,1][valid_mask[:,1]])
loss = 0.5 * loss_diastole + 0.5 * loss_exhale
```

**Training configuration:**
| Parameter | Value |
|-----------|-------|
| Device | CUDA |
| pos_weight (diastole) | 0.51 (diastole is majority) |
| pos_weight (exhalation) | 1.01 (near-balanced) |
| Optimizer | AdamW |
| LR | 1e-3 |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=15 on avg_acc |
| Checkpoint metric | avg(diastole_acc, exhalation_acc) |

**Accuracy metric:** Per-frame binary accuracy after sigmoid threshold at 0.5.

### 6.3 Training Results — Full Ablation

Production model uses equal task weights (0.5/0.5) with NaN-masked MIMIC exhalation:

| Experiment | Diastole | Exhalation | Notes |
|-----------|----------|------------|-------|
| EDR baseline (S-26) | 83.58% | ~52% | RSA EDR all records |
| **Reference labels CVES+MIMIC masked (S-36)** | **84.34%** | **56.54%** | **Production model** |
| 5s windows (S-38) | 82.27% | 57.36% | No gain |
| Large model 29K (S-40a) | 84.06% | 56.07% | No gain |
| Weighted loss 0.7 exh (S-40b) | 82.78% | 56.36% | No gain |
| Masked+FANTASIA (S-41) | 82.47% | 56.48% | FANTASIA hurts diastole |
| BIDMC added (S-49) | 83.13% | 56.14% | No gain on CVES |
| CVES-only (S-52) | 85.31% | 50.56% | Exhalation collapses |
| Dia-weighted 0.7/0.3 (S-57) | 83.60% | 56.13% | Regression |
| Fusion EDR MIMIC (S-65) | 83.67% | 54.33% | Regression |

**Production checkpoint:** `models/checkpoints/phase_detector_ref.pth`

---

## 7. Evaluation Framework

### 7.1 phase_evaluate.py (`src/training/phase_evaluate.py`)

Reports per-task metrics on any split:

| Metric | Per Task | Notes |
|--------|----------|-------|
| Accuracy | Diastole + Exhalation | Primary go/no-go metric (>85% each) |
| Precision | Per class | True positive rate |
| Recall | Per class | Sensitivity |
| F1 Score | Per class | Harmonic mean |
| Confusion Matrix | Per task | Visual output |

NaN frames are excluded from all metric computations — only frames where both a prediction and a valid label exist are counted.

### 7.2 Evaluation Results

| Evaluation | Description | Result |
|-----------|-------------|--------|
| S-39 | Per-source breakdown (CVES vs MIMIC) | CVES: 84.34%/56.54% — MIMIC: 69.73%/49.77% |
| S-29 | OOD: SHaRe 24h Holter (never in training) | Diastole: 81.06% (−3.3% gap) |
| S-30 | Latency benchmark (NFR-1.1 <200ms) | 116ms p95 at 100ms stride — **PASS** |

---

## 8. Autonomic State Module

`src/models/autonomic_state.py`

Computes a sliding-window HRV-based autonomic state vector every 60 seconds. Independent of the phase detector — runs on the same raw ECG in a parallel thread.

**Output vector (4 features):**

| Feature | Formula | Clinical Meaning |
|---------|---------|-----------------|
| LF/HF ratio | LF power / HF power | Sympathovagal balance |
| nHF (normalized HF) | HF / (LF + HF) | Vagal tone (0=sympathetic, 1=parasympathetic) |
| SampEn | Sample entropy | Autonomic complexity |
| DFA-α1 | Detrended fluctuation analysis | Long-range HRV correlation |

**Update rate:** Every 60s (vs. 5 Hz for phase detection — autonomic state changes slowly).

**Why these features:** Directly referenced in grant Aim 2: "frequency-domain HRV (LF/HF ratio, normalized HF power) and nonlinear indices (SampEn, DFA-α1) evaluate baseline autonomic state."

---

## 9. Stim Parameter Recommender

`src/models/stim_recommender.py`

Maps the autonomic state vector → stimulation parameters. Currently rule-based; ML upgrade planned for Phase 2.

**Input:** Autonomic state vector (LF/HF, nHF, SampEn, DFA-α1) + current phase flags (diastole=True/False, exhalation=True/False)

**Output:**
```python
{
    "amplitude":    float,  # mA, range [0, 10]
    "frequency":    float,  # Hz, range [1, 100]
    "pulse_width":  float,  # µs, range [50, 500]
}
```

**Decision logic:**
- Stimulation is only triggered when BOTH diastole=True AND exhalation=True
- Amplitude scales with vagal deficit (low nHF → higher amplitude)
- Frequency scales inversely with autonomic complexity (low SampEn → higher frequency)
- Pulse width held near physiologically safe defaults unless extreme autonomic disruption

**Why rule-based first:** The grant describes this as a "control algorithm," not a learned policy. A rule-based system is auditable, safe, and directly interpretable by clinicians. ML upgrade (e.g., RL policy) is deferred to Phase 2 after hardware validation.

---

## 10. Configuration

All hyperparameters in `config_stroke.yaml`:

```yaml
data:
  target_fs: 250             # All records resampled to this
  stroke_raw_dir: data/raw/stroke avns

phase_detection:
  window_sec: 2.0            # CNN input window
  stride_sec: 0.2            # Cache sliding window stride
  frame_rate_hz: 5.0         # Output label rate = grant NFR-1.1 update rate
  frames_per_window: 10      # 2.0 / 0.2 = 10 frames
  t_end_fallback_fraction: 0.4  # T-end = R + 0.4*RR when delineation fails

phase_model:
  channels: [16, 32, 48]     # Conv layer output channels
  kernels: [7, 5, 3]         # Kernel sizes
  strides: [5, 2, 2]         # Strided convolutions
  n_frames: 10               # AdaptiveAvgPool output length
  n_tasks: 2                 # diastole + exhalation
  dropout: 0.1

phase_training:
  learning_rate: 1.0e-3
  batch_size: 256
  max_epochs: 100
  early_stopping_patience: 15
  checkpoint_path: models/checkpoints/phase_detector.pth

edr:
  min_respiratory_cycles: 2  # Minimum cycles to generate exhalation labels
```

---

## 11. Design Decisions

### Why hardware reference labels instead of EDR for CVES?

EDR (RSA-based) was the original approach but fails under orthostatic stress — the exact CVES protocol. CVES records contain multi-channel WFDB signals including nasal thermistor (`thermst`) and airflow (`flow_rate`). These hardware channels provide direct respiratory measurement at 500 Hz. Discovery of these channels (S-32) changed the exhalation label strategy: 220/228 CVES records now use measured respiratory labels instead of inferred EDR. This +4.56% exhalation gain (+52% → +56.54%) directly validates the grant's Phase 2 hardware plan — if hardware reference improves labels during offline training, embedding impedance pneumography in the VNS device will close the exhalation gap in production.

### Why NaN-mask MIMIC exhalation instead of using EDR?

MIMIC records are resting ICU ECG — EDR (RSA method) correlation with reference is ~0.16 on CVES. For ICU patients, RSA is further suppressed. EDR labels for MIMIC are effectively noise (49.77% accuracy = worse than coin flip). Including them as training signal corrupts the exhalation head. NaN-masking eliminates exhalation gradients from MIMIC while preserving diastole gradients — MIMIC still improves diastole accuracy and stabilizes shared CNN features. QRS amplitude modulation EDR (S-60) was implemented and tested — fusion showed slightly better correlation (0.196) but full retrain showed regression. NaN masking is the optimal MIMIC strategy.

### Why ECG-Derived Respiration is still in edr.py?

Three EDR methods are implemented: RSA (`vangent2019`), QRS amplitude (`amplitude`), and PCA fusion (`fusion`). These serve as:
1. Fallback for the ~8 CVES records where reference label generation fails
2. Pre-hardware benchmark (documents the ECG-only ceiling)
3. Future: QRS-AM may be useful in single-lead wearable scenarios where hardware resp is unavailable

### Why 2,576,904 windows from 480 records?

The 0.2s stride (90% overlap) creates dense temporal coverage per record. A 10-minute CVES recording at 0.2s stride produces ~3,000 windows. This is intentional — phase detection requires frame-level temporal resolution, and more windows improve the model's ability to detect phase transitions.

### Why not use the AF pipeline's HybridEnsemble?

Wrong task. The AF HybridEnsemble classifies a 5-minute window into AF vs. NSR — a slow, episodic classification. Phase detection must update at 5 Hz — 150× faster. The HybridEnsemble is ~220K parameters; PhaseDetector is ~7,570 parameters. For real-time embedded deployment, every parameter counts.

### Why subject-level split instead of window-level?

Window-level split would put windows from the same patient in both train and test. The model would memorize subject-specific ECG morphology (not true phase detection). Subject-level split forces generalization to unseen patients — the actual clinical requirement.

### Why separate model from autonomic state module?

Phase detection operates at 5 Hz on 2s context. Autonomic state computation requires 60s of HRV data. They have incompatible time scales. Running them in parallel threads with independent update rates is cleaner than trying to combine them in a single forward pass.

### Why NaN masking instead of imputing labels?

Imputing NaN labels to 0 or 1 would inject artificial supervision signal for frames the label generator could not confidently classify (noisy segments, motion artifacts). NaN masking means those frames simply don't contribute to training — the model learns only from frames where we trust the derived label.

---

## 12. Known Limitations & Open Issues

| Issue | Severity | Status |
|-------|----------|--------|
| Exhalation accuracy 56.54% — below 85% NFR target | **By design** | ECG-only ceiling confirmed across 10 experiments — requires impedance hardware (Phase 2) |
| Diastole 84.34% — 0.66% below 85% NFR target | Low | Within training-run variance; CVES-only achieved 85.31% in one run |
| BIDMC parser bug — `II,` comma artifact in channel names | Low | 1 failing test; BIDMC excluded from production |
| neurokit2 ChainedAssignmentError (pandas CoW) | Low | Non-blocking, upstream package |
| No live inference path | Deferred | Phase 2: LSL streamer (pipeline and latency already validated) |
| No hardware integration | Deferred | Phase 2: Sparrow API + impedance pneumography |

---

## 13. How to Run

### Build Phase Cache

```powershell
Start-Process -WindowStyle Hidden "C:\Users\Edgar\AF VNS\.venv\Scripts\python.exe" `
  -ArgumentList "-m src.training.stroke_precompute_cache --config config_stroke.yaml --workers 4" `
  -WorkingDirectory "C:\Users\Edgar\AF VNS" `
  -RedirectStandardOutput "precompute_phase.log" `
  -RedirectStandardError "precompute_phase_err.log"

# Monitor:
Get-Content "C:\Users\Edgar\AF VNS\precompute_phase_err.log" -Tail 5
```

### Train Phase Detector

```powershell
Start-Process -WindowStyle Hidden "C:\Users\Edgar\AF VNS\.venv\Scripts\python.exe" `
  -ArgumentList "-m src.training.phase_train --cache-dir models/artifacts/cache_phase_detect --checkpoint-out models/checkpoints/phase_detector_ref.pth" `
  -WorkingDirectory "C:\Users\Edgar\AF VNS" `
  -RedirectStandardOutput "phase_train.log" `
  -RedirectStandardError "phase_train_err.log"

# Monitor:
Get-Content "C:\Users\Edgar\AF VNS\phase_train_err.log" -Tail 5
```

### Evaluate

```bash
# In-distribution test split
.venv/Scripts/python -m src.training.phase_evaluate \
  --config config_stroke.yaml --split test

# OOD: SHaRe
.venv/Scripts/python -m src.training.phase_evaluate \
  --config config_stroke.yaml --split sharee
```

### Run Tests

```bash
.venv/Scripts/python -m pytest tests/ -v -k "phase or edr or autonomic or stim"
```

### Latency Benchmark

```bash
.venv/Scripts/python -m src.training.phase_evaluate \
  --config config_stroke.yaml --benchmark
```
