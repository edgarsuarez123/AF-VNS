# Phase 1 Implementation Plan: RS-Personalized AI-Driven Adaptive Stimulation (Aim 2)

## Goal

**Core Adaptive AI Functionality:** Develop an offline-trainable, online-capable inference pipeline using PyTorch that ingests multi-modal physiological data, processes HRV via wavelet transforms, runs a Hybrid Ensemble (CNN + RNN + Transformer), and outputs state-machine controlled trigger commands to the Sparrow Link pulse generator.

---

## Progress Summary

| Category | Status |
|----------|--------|
| **Overall** | 20/30 tasks (67%) |
| Environment & Infrastructure | 4/4 tasks (100%) ✅ |
| Data Foundation (Ingestion) | 4/5 tasks (80%) |
| Preprocessing & Feature Extraction | 7/7 tasks (100%) ✅ |
| Hybrid AI Models (PyTorch) | 5/5 tasks (100%) ✅ |
| Training & Validation Engine | 0/4 tasks (0%) |
| Live Inference & Hardware API | 0/5 tasks (0%) |

- [x] Environment & Infrastructure - 4/4 tasks (100%)
- [x] Data Foundation (Ingestion) - 4/5 tasks (80%; LSL deferred to Live Integration)
- [x] Preprocessing & Feature Extraction - 7/7 tasks (100%) ✅
- [x] Hybrid AI Models (PyTorch) - 5/5 tasks (100%) ✅
- [ ] Training & Validation Engine - 0/4 tasks (0%)
- [ ] Live Inference & Hardware API - 0/5 tasks (0%)

---

## Phase 1 Scope: MVP AI & Inference System

### ✅ What's Included

#### 1. Environment & Infrastructure

- [x] **Repository Setup** — Initialize Git, `.gitignore` (exclude `.edf` and `.csv` datasets).
- [x] **Dependency Management** — Create `requirements.txt`: torch, torchaudio, mne, neurokit2, pylsl, scikit-learn, pandas, numpy, scipy, **PyWavelets** (pywt).
- [x] **Directory Structure** — Scaffold `src/data`, `src/features`, `src/models`, `src/hardware`, `src/training`, `tests/`, `models/artifacts/`, `models/checkpoints/`.
- [x] **Config Management** — Create `config.yaml` as single source of truth: learning rate, batch size; **LSL expected rate (250 Hz)**; **HRV window lengths (30 s, 5 min)**; **wavelet params** (family, scale range); **watchdog max duration & max intensity** (Sparrow/FDA limits); paths for scaler (`models/artifacts/scaler.pkl`) and checkpoint (`models/checkpoints/best_model.pth`).

#### 2. Data Foundation (Ingestion)

- [x] **Dataset Downloader** — `src/data/download_physionet.py` using wfdb: **MIT-BIH AF**, **PhysioNet Normal Sinus Rhythm**; MIMIC-III reads from configured dir (warn if missing). Channel mapping documented in parser/README.
- [x] **EDF/CSV Parsers** — `src/data/dataset_parsers.py` converts WFDB, EDF, and MIMIC-III CSV/EDF into standard schema (subject_id, signal, fs, label).
- [x] **PyTorch DataLoaders** — `src/data/dataloaders.py`: `PhysioDataset` with configurable 10 s and 5 min windows; `get_dataloaders()` from config and split.
- [x] **Data Splitter** — `src/data/splitter.py`: 70/15/15 by subject_id; `create_split()` / `load_split()`; persist to `models/artifacts/split.json`.
- [ ] **Galea LSL Receiver** — `src/data/lsl_streamer.py` (deferred to Live Integration phase). Buffer **≥250 Hz** (FR-1.1); verify stream rate, fail fast or warn if below.

#### 3. Preprocessing & Feature Extraction

- [x] **Signal Denoising** — `src/features/wavelet_filter.py`: CWT (PyWavelets) to remove baseline wander **&lt;0.5 Hz** and high-frequency noise. Specify wavelet family (e.g. `'cmor'` or `'mexh'`), scale range; keep params in `config.yaml`.
- [x] **Artifact Rejection** — `src/features/artifact_scrubber.py`: reject windows where amplitude exceeds **X× MAD** or R–R differs from local median by **&gt;Y%**; X, Y in config.
- [x] **R-Peak Detection** — `src/features/peak_detector.py` utilizing neurokit2 (e.g. `nk.ecg_peaks`) to find exact R–R intervals.
- [x] **Time-Domain HRV** — `src/features/hrv_time.py`: RMSSD, SDNN from R–R intervals.
- [x] **Frequency-Domain HRV** — `src/features/hrv_freq.py`: LF/HF ratio via Welch's method (`scipy.signal`).
- [x] **Non-Linear HRV** — `src/features/hrv_nonlinear.py`: e.g. sample entropy and/or DFA α1 from R–R intervals; integrate into feature pipeline and DataLoader (per original Aim 2).
- [x] **Feature Standardizer** — `src/features/scaler.py`: Z-score with `sklearn.preprocessing.StandardScaler`; **fit on training set only**; persist to `models/artifacts/scaler.pkl`. Live pipeline must load this same scaler for inference.

#### 4. Hybrid AI Models (PyTorch)

- [x] **CNN Morphology Module** — `src/models/cnn.py`: 1D-CNN (`nn.Conv1d`) for 10-second raw waveform feature extraction. Returns 128-dim embedding.
- [x] **RNN Temporal Module** — `src/models/rnn.py`: GRU network (`nn.GRU`) processing 5-minute sequences of HRV features. Returns 64-dim hidden state.
- [x] **Transformer Attention Module** — `src/models/transformer.py`: `nn.TransformerEncoder` handling parameter optimization and attention weighting across the sequence.
- [x] **Hybrid Ensemble Logic** — `src/models/ensemble.py`: The wrapper `nn.Module` that concatenates CNN, RNN, and Transformer outputs into a final Fully Connected (FC) classification layer (`nn.Linear` → `nn.Sigmoid`).
- [x] **Dimensionality Reduction** — `src/models/pca_reduction.py` using `sklearn.decomposition.PCA` to map Galea 8-channel data to Sparrow 1-channel equivalent.

#### 5. Hardware Actuation & Safety

- [ ] **Sparrow Link API** — `src/hardware/sparrow_api.py` with mockable HTTP/Bluetooth (e.g. `def trigger_stim(duration, intensity):`). **All** stimulation requests (including MockSparrowAPI in Phase 1) must go through the watchdog—no direct `trigger_stim()` from the AI layer.
- [ ] **Safety Watchdog** — `src/hardware/watchdog.py`: Intercepts every API call; enforces **max duration** and **max intensity** read from `config.yaml` (Sparrow/FDA limits). Throws `SafetyViolationException` on override attempt.
- [ ] **State Machine** — `src/hardware/state_machine.py`: IDLE → SENSING → ANALYZING → STIMULATING | COOLDOWN. `live_inference.py` drives the state machine; LSL buffer feeds SENSING/ANALYZING.

---

### ❌ What's Deferred to Phase 2

- **Embedded C++ Firmware:** All AI code currently runs on an external PC/Cloud instance.
- **Physical Sparrow Link Integration:** Phase 1 uses a `MockSparrowAPI` to log outputs to the console instead of firing physical hardware.
- **Live Human Trials:** Only algorithm benchmarking on datasets and Galea lab data.

---

## Prerequisites & Risks

- **Dataset access:** Obtain PhysioNet/MIMIC-III credentials; confirm download scripts run (wfdb, etc.) by end of Week 1. If delayed, document fallback (e.g. BIDMC, LTAF) per risk matrix.
- **LSL unavailable:** Support a **replay mode**: stream from a recorded file (EDF/CSV) at 250 Hz into the same buffer interface so `live_inference.py` can be tested without Galea hardware.

---

## Implementation Order (Step-by-Step)

### First (Weeks 1–2): Data & Pipelines

**Setup & Ingestion**

- [x] Run `pip install -r requirements.txt`.
- [x] Execute `download_physionet.py` to populate `data/raw/` (MIT-BIH AF, Normal Sinus, MIMIC-III dir checked).
- [x] Write `dataset_parsers.py` to convert records to standard schema (subject_id, signal, fs, label).
- [x] Implement **data splitter** (70/15/15 by Patient ID); use it before any training.

**Preprocessing Engine**

- [x] Complete `wavelet_filter.py` (CWT params in config) and verify signal plots (matplotlib).
- [x] Complete `peak_detector.py`, `hrv_time.py`, `hrv_freq.py`, `hrv_nonlinear.py`.
- [x] Create **`tests/test_preprocessing.py`** (PyTest): HRV calculations match NeuroKit2 benchmarks; CWT output sanity checks.

---

### Then (Weeks 3–4): Core Deep Learning

**Model Architectures**

- [x] Implement `cnn.py`, `rnn.py`, and `transformer.py`.
- [x] Assemble `ensemble.py`.
- [x] Write a dummy forward pass test: `tensor(batch=32, seq=300, channels=1)` → model → output shape `(32, 1)`.

**Training Loop**

- [ ] Create `src/training/train.py`: BCEWithLogitsLoss, AdamW; **fit scaler on training set only**; persist to `models/artifacts/scaler.pkl`.
- [ ] Create `src/training/evaluate.py`: AUROC, F1, Sensitivity, Specificity.
- [ ] Train on PhysioNet until AUROC ≥ 0.75. Save best weights to `models/checkpoints/best_model.pth`. **Contingency:** If AUROC &lt; 0.75 after N epochs (N in config), run hyperparameter sweep (LR, batch size); document in training log; escalate if still below.

---

### Next (Week 5): Cross-Validation & Simulation

**Validation Testing**

- [ ] Run `evaluate.py` on MIMIC-III holdout. Verify F1/AUROC degradation **&lt;5%** (NFR-3.1).
- [ ] **5-fold cross-validation** (TR-1.2): Implement and run on MIT-BIH; document per-fold and mean AUROC/F1.
- [ ] **Noise Injection Test** (TR-1.3): Add motion-artifact-like noise to validation set; re-test; target ≥80% accuracy during noise.

**Dimensionality Reduction**

- [ ] Train PCA mappings (multi-channel → single-channel); TR-2.3: strip 80% channels, verify ≥75% baseline accuracy.

---

### Finally (Weeks 6–8): Live Integration & Control

**LSL Real-Time Stream**

- [ ] Implement `lsl_streamer.py`. **Verify LSL stream ≥250 Hz** (check nominal_srate or timestamps); warn/fail if below.
- [ ] Write `live_inference.py`: LSL (or replay file) → Preprocessing (**load `models/artifacts/scaler.pkl`**) → PyTorch model → state machine → console (and MockSparrow via watchdog). No direct `trigger_stim()` from model.

**Hardware & Safety Wrapper**

- [ ] Implement `watchdog.py` (limits from `config.yaml`).
- [ ] Implement `state_machine.py`.
- [ ] **Explicit tests:** `tests/test_watchdog.py` (over-limit command → SafetyViolationException); `tests/test_state_machine.py` (valid transitions only); `tests/test_dataloader.py` (window shapes, no patient leakage across splits).
- [ ] **Latency profiling (TR-2.1, NFR-1.1):** Run 10,000 simulated heartbeats through pipeline; cProfile (or equivalent); report **p50/p95/p99**; **pass if p99 &lt; 200 ms**.
- [ ] **24-hour HIL test (TR-2.2):** Mock Sparrow; log every command; confirm no safety interlock violations.

---

## Acceptance Criteria for Phase 1

1. **pytest** suite passes with **&gt;85%** code coverage for `src/features/` and `src/hardware/` (exclude or separately scope `src/training/` and `src/models/` if desired).
2. Training logs prove **AUROC ≥ 0.75** on MIT-BIH test split (NFR-2.1 Phase I).
3. Latency profiler: full cycle (buffer → preprocess → forward → command) **p99 &lt; 200 ms** (NFR-1.1).
4. System connects to local LSL (or replay stream) and prints real-time probability scores without crashing.

---

## Traceability (FR/NFR/TR → Implementation)

| Requirement | Implementation / Test |
|-------------|------------------------|
| FR-2.1 (CWT) | `wavelet_filter.py`, `tests/test_preprocessing.py` |
| FR-2.2 (HRV windows) | `dataloaders.py` + config window lengths, `hrv_time.py`, `hrv_freq.py` |
| FR-4.2 (Safety) | `watchdog.py`, `tests/test_watchdog.py` |
| NFR-1.1 (&lt;200 ms) | Latency profiler script, 10k-beat run, p99 |
| NFR-2.1 (AUROC ≥0.75) | `evaluate.py`, training logs |
| NFR-3.1 (&lt;5% drop) | MIMIC-III holdout evaluation |
| TR-1.1 (holdout) | Data splitter, `train.py` / `evaluate.py` |
| TR-1.2 (5-fold) | Week 5 CV task |
| TR-1.3 (noise) | Noise injection test |
| TR-2.1 (latency) | Profiler + 10k heartbeats |
| TR-2.2 (HIL) | 24 h Mock Sparrow test |
| TR-2.3 (dim reduction) | PCA test, 80% channels stripped |

---

## Phase 1 Exit Checklist

- [ ] All Phase 1 FRs/NFRs/TRs mapped to a test or script (see Traceability).
- [x] `config.yaml` contains: LSL rate (250 Hz), HRV window lengths (30 s, 5 min), wavelet params, artifact thresholds (X, Y), watchdog max duration & intensity, scaler path, checkpoint path.
- [x] README or runbook: how to run training, evaluation, latency profiler, and `live_inference.py` (LSL or replay).
- [ ] Four acceptance criteria above signed off.
