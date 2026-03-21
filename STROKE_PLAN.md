# Stroke AVNS — Aim 2 Phase Detection Pipeline

Branch: `feature/stroke-avns`

## Grant Context (Aim 2)

The grant describes Aim 2 as: a hybrid CNN/RNN system that processes 250Hz HRV
time-series and multi-frequency impedance data continuously. The system must:

1. **Detect diastolic phase** — R-R interval trajectory identifies diastole in real time
2. **Detect exhalation phase** — thoracic impedance identifies exhalation (no impedance
   hardware yet → use ECG-Derived Respiration as proxy)
3. **Assess autonomic state** — frequency-domain HRV (LF/HF ratio, normalized HF power)
   and nonlinear indices (SampEn, DFA-α1) evaluate baseline autonomic state
4. **Adaptive stim output** — dynamically adjust amplitude (0-10mA), frequency (1-100Hz),
   pulse width (50-500µs) at 5Hz update rate

**Go/no-go milestone:** >85% classification accuracy for diastolic phase AND exhalation
phase detection, with total closed-loop system latency <200ms. (NFR-1.1 and NFR-2.1)

---

## Course Correction (2026-03-19)

Steps 2-12 built a **stroke vs. control binary classifier**. This was the wrong task.
Stroke does not have a direct ECG signature like AF does — the model never learned
(Phase 1 val_auroc=0.57, Phase 2 in-dist AUROC=0.60, OOD AUROC=0.41).

The grant's ML task is **real-time physiological phase detection on stroke patients**,
not detecting whether someone had a stroke. The stroke patients are the population,
not the classification target.

### What is reusable
- ECG parsers (CVES, MIMIC-3, SHaRe) — data loading works
- HRV feature extraction pipeline (LF/HF, SampEn, DFA-α1) — already built
- CNN architecture — repurpose for phase detection on short windows
- Training loop, caching, evaluation framework — structure stays
- config_stroke.yaml — extend with phase detection settings

### What is NOT reusable
- StrokeHybridEnsemble / StrokeResponderHead — wrong output (binary stroke/control)
- stroke_train.py / stroke_evaluate.py — wrong task and labels
- Precomputed caches — wrong windowing and labels

---

## Architecture

```
ECG (250Hz) ─────┬── Phase Detector CNN ──── diastole logit (5Hz)
                  │                      └── exhalation logit (5Hz)
                  │
                  └── HRV Window (60s) ──── Autonomic State Module
                                            ├── LF/HF ratio
                                            ├── normalized HF power
                                            ├── SampEn
                                            └── DFA-α1
                                            ↓
                                        Stim Parameter Recommender
                                            ├── amplitude (0-10mA)
                                            ├── frequency (1-100Hz)
                                            └── pulse_width (50-500µs)
```

**Phase Detector:** Lightweight 1D CNN on 2s ECG context (500 samples @ 250Hz).
Outputs diastole + exhalation probability at every 200ms frame (5Hz update rate).
Must run in <200ms including feature extraction.

Architecture (PhaseDetector, ~7.5K params):
```
(B,1,500) → Conv1d(1→16,k=7,s=5)+BN+ReLU → (B,16,100)
          → Conv1d(16→32,k=5,s=2)+BN+ReLU → (B,32,50)
          → Conv1d(32→48,k=3,s=2)+BN+ReLU → (B,48,25)
          → AdaptiveAvgPool1d(10)          → (B,48,10)
          → Dropout(0.1) → Conv1d(48→2,k=1) → permute → (B,10,2)
```

**Autonomic State Module:** Sliding-window HRV features (existing pipeline).
Updates every 60s. Feeds the stim parameter recommender (control algorithm, not ML).

### Ground Truth Label Generation

**Diastolic phase labels:**
- R-peak detection via neurokit2 `nk.ecg_peaks()` or Pan-Tompkins
- T-wave end detection via `nk.ecg_delineate()`
- Diastole = T-wave end to next R-peak onset
- Label each 200ms frame as diastole=1 or systole=0

**Exhalation phase labels (ECG-Derived Respiration):**
- No impedance hardware → derive respiration from ECG
- neurokit2 `nk.ecg_rsp()` extracts respiratory signal from R-peak amplitude modulation
  and respiratory sinus arrhythmia (RSA)
- Detect respiratory peaks/troughs → label exhale=1, inhale=0
- EDR accuracy vs. reference is typically 80-90% — this is the ceiling for our model

### Datasets

| Dataset | Records | Use | Notes |
|---------|---------|-----|-------|
| CVES | 228 (74 stroke, 154 control) | Primary — target population doing autonomic tests | Sit-stand, tilt protocols → varied HR/breathing |
| MIMIC-3 stroke | 300 (150/150) | Volume — ICU ECGs | Noisier, but more data |
| SHaRe | 133 (17 event, 116 control) | OOD evaluation | 24h Holter ECGs |

All three contain ECG from which diastolic phase and EDR can be derived.
The stroke/control label is irrelevant for phase detection — every ECG has cardiac
cycles and respiratory modulation regardless of pathology.

---

## Steps

### Label Generation & Data Prep

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| S-13 | `src/features/phase_labels.py` — generate diastolic phase labels from ECG (R-peak + T-end detection → per-frame labels at 5Hz) | **Done** | 2026-03-20 |
| S-14 | `src/features/edr.py` — ECG-Derived Respiration: extract respiratory signal, detect inhale/exhale phases, generate per-frame labels at 5Hz | **Done** | 2026-03-20 |

#### S-14 Sub-steps

| # | Action | Status |
|---|--------|--------|
| 14a | Add `edr:` config section to `config_stroke.yaml` | Done |
| 14b | Implement `extract_edr()` — nk.ecg_rsp + rsp_findpeaks | Done |
| 14c | Implement `generate_exhalation_labels()` — orchestrator | Done |
| 14d | Run existing tests — 136 pass, 1 flaky (unrelated) | Done |
| 14e | Smoke test — 300 frames, 12 cycles, 13.5 bpm, 42.9% exhale | Done |
| 14f | Git commit + update progress.txt | Done |
| S-15 | Tests for phase_labels.py + edr.py (synthetic + real CVES records) | **Done** | 2026-03-20 |
| S-16 | Rewrote `stroke_precompute_cache.py` — precompute 2s ECG windows + 5Hz frame-level diastole/exhalation labels; replaces superseded binary classifier cache | **Done** | 2026-03-20 |
| S-17 | Tests for stroke_precompute_cache.py (12 tests — cache shapes, meta, splits, process_record, NaN preservation) | **Done** | 2026-03-20 |

#### S-13 Sub-steps

| # | Action | Status |
|---|--------|--------|
| 13a | Add `phase_detection:` config to `config_stroke.yaml` | Done |
| 13b | Implement `get_rpeak_indices()` | Done |
| 13c | Implement `get_twave_offsets()` + 40% RR fallback | Done |
| 13d | Implement `_build_sample_phase_array()` | Done |
| 13e | Implement `_downsample_to_frames()` | Done |
| 13f | Implement `generate_phase_labels()` | Done |
| 13g | Run existing tests — 136 pass, 0 fail | Done |
| 13h | Smoke test — 150 frames, 72.1bpm, 62.3% diastole, 0% fallback | Done |
| 13i | Git commit + update progress.txt | Done |

### Phase Detection Model

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| S-18 | `src/models/phase_detector.py` — PhaseDetector CNN: 3 Conv1d+BN+ReLU blocks (1→16→32→48, strides 5/2/2), AdaptiveAvgPool1d(10), 1×1 conv head → (B,10,2). ~7.5K params. | **Done** | 2026-03-20 |
| S-19 | `config_stroke.yaml` — add `phase_model:` section (channels, kernels, strides, n_frames, n_tasks, dropout) | **Done** | 2026-03-20 |
| S-20 | `src/training/phase_train.py` — training loop for phase detector (multi-task BCE loss with NaN masking, 5Hz frame-level labels) + config additions (phase_training section, checkpoint path) | **Done** | 2026-03-20 |
| S-21 | Tests for phase_train.py (12 tests: dataset, loss NaN masking, pos_weight, validation accuracy, smoke) | **Done** | 2026-03-20 |

### Autonomic State + Stim Recommender

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| S-22 | `src/models/autonomic_state.py` — sliding-window HRV feature extractor producing autonomic state vector (LF/HF, nHF, SampEn, DFA-α1) | **Done** | 2026-03-20 |
| S-23 | `src/models/stim_recommender.py` — maps autonomic state → stim params (amplitude, frequency, pulse_width); rule-based initially, ML later | **Done** | 2026-03-20 |
| S-24 | Tests for autonomic_state.py + stim_recommender.py (16 tests: nHF, compute, NaN, rules, clipping, factories) | **Done** | 2026-03-20 |

### Training, Evaluation & Latency

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| S-25 | Precompute phase labels for CVES + MIMIC-3 (background) | **Done** | 2026-03-20 |
| S-26 | Train phase detector on CVES + MIMIC-3 combined | **Done** | 2026-03-20 — early stop Epoch 50, best avg_acc=68.64% |
| S-27 | `src/training/phase_evaluate.py` — accuracy, per-class precision/recall for diastole + exhalation; confusion matrices | **Done** | 2026-03-20 |
| S-28 | Evaluate + diagnose exhalation bottleneck | **Running** | 2026-03-21 — see sub-steps below |
| S-29 | OOD evaluation on SHaRe — report generalization | Pending | |
| S-30 | Latency benchmark — single-window inference <200ms end-to-end | Pending | |
| S-31 | Integration: phase detector + autonomic state + stim recommender end-to-end | Pending | |

### Previous Steps (Stroke vs. Control — completed but superseded)

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| S-2 | config_stroke.yaml | Done | Reusable — extend with phase detection config |
| S-3–5 | stroke_parsers.py (MIMIC-3, SHaRe, CVES) | Done | Reusable — ECG loading still needed |
| S-7 | parse_cerevasc_dir() | Done | Reusable |
| S-8 | stroke_dataloaders.py | Done | Superseded — new windowing needed |
| S-9 | stroke_precompute_cache.py | Done | Superseded — new label generation |
| S-10–11 | StrokeResponderHead, StrokeHybridEnsemble | Done | Superseded — wrong output head |
| S-12 | stroke_train.py + tests | Done | Superseded — wrong task |
| — | stroke_evaluate.py | Done | Superseded — wrong metrics |
| — | SHAREE_EVENT_PATIENTS corrected | Done | Still useful for OOD eval |

---

## Data Downloads (completed 2026-03-18)

### Directory Reorg
1. Created `data/raw/AF avns/` and `data/raw/stroke avns/` — done
2. Moved afdb, nsrdb, mimic3, ltafdb, challenge2017 under `AF avns/` — done
3. Updated config.yaml raw_dir + mimic3_subdir — done
4. Updated hardcoded paths in download scripts — done

### Stroke Downloads
5. Created `download_stroke_datasets.py` (cves + shareedb via wfdb) — done
6. Created `download_mimic3_stroke_waveforms.py` (ICD9 430-438) — done
7. CVES download complete — 228 records (74 stroke, 154 control)
8. SHaRe download complete — 133 records (17 event, 116 control)
9. MIMIC-III stroke download complete — 300 records (150/150)

---

## S-28 Sub-steps — Diagnose & Fix Exhalation

| # | Action | Status |
|---|--------|--------|
| 28a | Fix 3 bugs in phase_evaluate.py (naming, config fields, checkpoint loading) | Done |
| 28b | Add --dataset cves/mimic to stroke_precompute_cache.py | Done |
| 28c | Add --cache-dir, --source-breakdown to phase_evaluate.py | Done |
| 28d | Add --cache-dir, --checkpoint-out to phase_train.py | Done |
| 28e | Evaluate combined test split: diastole 83.58%, exhalation 51.98% | Done |
| 28f | Build CVES-only + MIMIC-only caches | Running |
| 28g | Per-source evaluation (CVES vs MIMIC exhalation accuracy) | Pending |
| 28h | CVES-only retrain (if diagnosis confirms MIMIC noise) | Pending |
| 28i | Longer 5s window (if CVES-only retrain insufficient) | Pending |

---

## Resume From Here

**Current state (2026-03-21 11:05) — S-28 diagnostic in progress:**
- **S-26 DONE**: Training finished — early stop Epoch 50, best avg_acc=68.64%
  - Diastole: 85.42% (val), 83.58% (test)
  - Exhalation: 51.83% (val), 51.98% (test) — **near random chance**
- **S-28 IN PROGRESS**: Fixed evaluation bugs, added per-source CLI
  - Building CVES-only and MIMIC-only caches (background)
  - Next: run per-source eval to confirm MIMIC noise hypothesis
  - Then: CVES-only retrain → if insufficient, try 5s window
