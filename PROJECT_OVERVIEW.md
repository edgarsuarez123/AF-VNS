# AF VNS — Project Overview

**RS-Personalized AI-Driven Adaptive Stimulation, Aim 2, Phase 1**
*Offline-trainable, online-capable AI pipeline for auricular vagus nerve stimulation, triggered by AF (atrial fibrillation) vs. NSR (normal sinus rhythm) classification from ECG/HRV.*

Status as of this writing: **Round 9 complete, Round 10 in progress.** Best out-of-distribution result: **AUROC = 0.7432** on PhysioNet Challenge 2017 (ambulatory ECG, never seen in training), with clinically strong sensitivity (91.6%) and known-weak specificity (32.9%) that Round 10 is addressing.

---

## 1. The Problem This Solves

Non-invasive auricular vagus nerve stimulation (aVNS) is a candidate therapy for atrial fibrillation, but existing devices use **fixed stimulation parameters** — they stimulate on a timer or continuously, regardless of the patient's actual cardiac rhythm at that moment. This is a blunt instrument: it can't tell whether a patient is currently in AF or NSR, so it can't adapt *when* or *how strongly* to stimulate to that patient's real-time physiological state.

**Aim 2's premise:** an AI that reads a patient's ECG/HRV in real time, classifies their rhythm (AF vs. NSR), and only then decides whether to trigger stimulation, should outperform fixed-parameter stimulation — provided the classifier is accurate and generalizes to real patients (not just the lab data it was trained on).

**Phase 1's job** (this codebase) is to prove that the classification half of that loop works *offline*, on large public datasets, before any hardware is involved:

- Can a model tell AF from NSR using ECG waveform morphology + heart rate variability (HRV)?
- Does it still work on data from a completely different device/hospital/population than it trained on (the real clinical requirement — a device shipped to patients will always see "OOD" data relative to its training set)?
- Is it fast enough (`<200ms`) and safe enough (bounded stimulation, watchdog limits) to eventually drive real hardware?

The hardware/live-stimulation half (Sparrow Link pulse generator, LSL streaming from the Galea headset, a safety watchdog and state machine) is explicitly **deferred** to a later "Live Integration" phase — Phase 1 only builds and validates the brain, using a mock device.

---

## 2. What The System Actually Does

End to end, in one sentence: **it takes a chunk of raw ECG, denoises it, extracts a 10-second waveform snippet plus a 5-minute sequence of heart-rate-variability features, runs both through a three-branch neural network, and outputs a single probability that the patient is in AF right now.**

```
Raw ECG (WFDB / EDF / CSV, from 5 different public datasets)
        │
        ▼
  Parse → standard schema → resample all sources to 250 Hz
        │
        ▼
  Split by PATIENT (not by window) into train/val/test — no leakage
        │
        ▼
  Precompute cache (build once, train many times fast)
        │
   ┌────┴─────────────────────────────┐
   ▼                                   ▼
CWT wavelet denoise               CWT wavelet denoise
   │                                   │
10s raw waveform                  R-peak detection → RR intervals
   │                              → artifact correction/rejection
   │                              → time/freq/nonlinear HRV
   │                              → (5 steps × N features) sequence
   │                                   │
   ▼                                   ▼
 CNNEncoder (128-d)      GRUEncoder (64-d)   TransformerEncoder (64-d)
   │                              │                    │
   └──────────────── concat (256-d) ────────────────────┘
                              │
                    Dropout → Linear → ReLU → Dropout → Linear
                              │
                      single logit → sigmoid → P(AF)
```

This same pipeline is used for training (with labels, from public annotated databases) and would be used for live inference (frame-by-frame, no label — deferred to Phase 2/Live Integration).

---

## 3. Architecture

### 3.1 Why a hybrid ensemble (CNN + GRU + Transformer), not one model

Each branch captures something the others don't:

| Branch | Input | Captures | Why this architecture |
|---|---|---|---|
| **CNN** (`src/models/cnn.py`) | 10s raw waveform @ 250 Hz | Waveform **morphology**: P-wave presence/absence, fibrillatory (f-)waves, QRS shape, short-timescale RR irregularity | 4-layer 1D-CNN, `Conv1d(1→16→32→64→128)` with stride-2 downsampling, BatchNorm, ReLU, `AdaptiveAvgPool1d(1)` at the end so it's agnostic to exact input length (needed because sources have different native sampling rates before resampling) |
| **GRU** (`src/models/rnn.py`) | 5-step sequence of HRV features (one vector per 60s over 5 minutes) | **Sequential/temporal** evolution of autonomic tone — how HRV changes step to step | Single-layer GRU chosen over LSTM: fewer parameters, trains faster, and with only 5 timesteps the extra LSTM gating buys nothing. Supports `pack_padded_sequence` so it ignores zero-padded timesteps on short records |
| **Transformer** (`src/models/transformer.py`) | Same 5-step HRV sequence | **Attention-weighted** relationships between timesteps, independent of order — e.g. "timestep 4's SDNN spike matters regardless of what came before" | 1 encoder layer, 4 heads, `d_model=64`, sinusoidal (not learned) positional encoding — learned PE would overfit on a 5-token sequence — masked mean pooling over valid timesteps only |
| **Fusion head** (`src/models/ensemble.py`) | concat of all three embeddings (256-d) | Final AF/NSR decision | `Dropout(0.2) → Linear(256→64) → ReLU → Dropout(0.1) → Linear(64→1)` → single logit for `BCEWithLogitsLoss`. The 64-unit hidden layer (added in Round 9) exists specifically to give the classifier head enough capacity to fine-tune independently of the frozen backbone (see §3.2) |

Running GRU and Transformer on the *identical* HRV sequence is deliberate: they're complementary lenses on the same 5 numbers-over-time, and letting the final linear layer learn how much to trust each is cheaper and more robust than hand-designing which one is "right."

Raw-ECG-only (CNN alone) was considered but rejected: it captures short-term morphology well but misses the **autonomic nervous system dynamics** (the multi-minute trajectory of RR variability) that distinguish paroxysmal from sustained AF — relevant to *when* to stimulate, not just *whether* the current 10s looks like AF.

### 3.2 Two-phase transfer learning (the current architecture, since Round 9)

This is the single biggest architectural decision in the project's history, so it gets its own section.

**The problem it solves:** the largest source of training data, PhysioNet Challenge 2017 (5,788 records), consists of 30–61 second ambulatory ECG snippets. The model's HRV branch needs a 300-second (5-minute) window. A 30–61s record can fill at most 1 of 5 HRV timesteps — the other 4 are structurally zero. Since C2017 was 93% of all training data, the GRU/Transformer branches spent nearly all of training looking at mostly-zero-padded input. This was diagnosed as the primary reason AUROC plateaued around 0.68 for several rounds.

**The fix — split training into two phases with different data and different frozen/trainable layers:**

- **Phase 1 (pre-train, all layers trainable).** Train the *entire* network — CNN, GRU, Transformer, head — on MIMIC-III ICU waveform records only (1,032 records, all ≥500s, so every training sample gets a full, non-padded 5-step HRV sequence). Labels come from ICD-9 diagnosis codes (AF = code 42731), so they're noisier than a cardiologist annotation, but the recordings are real-world, noisy ICU signal — exactly the noise profile a deployed device will actually see. **Goal: teach the backbone noise-robust, general ECG representations, not clean AF/NSR discrimination.** Result: val AUROC = 0.6976 (expected to be modest, given noisy labels — that's not the point of this phase).
- **Phase 2 (fine-tune, backbone frozen).** Freeze the CNN/GRU/Transformer weights from the Phase 1 checkpoint. Train *only* the ~16.5K-parameter classification head on curated, clean, expert-annotated AF/NSR data — MIT-BIH AFDB + NSRDB + Long-Term AF Database segments (~125 subjects, subject-level 70/15/15 split). **Goal: teach the head a clean decision boundary on top of already-meaningful features**, without enough data to overfit a much larger from-scratch model. Result: val AUROC = 0.9830 on this held-out curated test set — credible, not a red flag, because the backbone was frozen (minimal overfitting surface) and the split is patient-level.
- **PhysioNet Challenge 2017 is removed from training entirely** and repurposed as the **sole out-of-distribution (OOD) evaluation set** — a smartphone/AliveCor-patch ECG, a device category the model has never trained on, from patients it has never seen. This is the metric that matters for "will this generalize to a real patient wearing a real device": **AUROC = 0.7432**, sensitivity = 91.6%, specificity = 32.9%.

Why freezing the backbone works here specifically: it learned general ECG representations from 1,032 real clinical recordings in Phase 1; the curated Phase 2 dataset (~125 subjects) is far too small to retrain a network this size without overfitting, so forcing only the head to adapt is the right capacity trade-off.

### 3.3 AUROC trajectory across development rounds

The project's history is a chain of diagnosing *why* an initially "too good to be true" 0.9999 AUROC was actually a sign of a broken pipeline, and fixing the pipeline one root cause at a time until the number became trustworthy and, eventually, good. See §6 for detail; the headline numbers on the true OOD benchmark (MIMIC-III, later Challenge 2017):

| Round | What changed | OOD AUROC |
|---|---|---|
| 1st in-sample run | 41 records, no cross-dataset check | 0.9999 (bogus — see §6.1) |
| Round 4 | First honest MIMIC-III cross-dataset eval | 0.6656 |
| Round 5 | 4 data sources, HRV pipeline bug fixes | 0.6022 |
| Round 6 | + attention masking, + augmentation | 0.6333 |
| Round 7 | + MIMIC mixed into training | 0.6747 |
| Round 8 | + label smoothing, + per-source stride | 0.6777 |
| **Round 9** | **Two-phase transfer learning; switch OOD benchmark to Challenge 2017** | **0.7432** |
| Round 10 (in progress) | Threshold tuning, +2 HRV features, +MIMIC controls in Phase 2, partial backbone unfreeze | target: close remaining 0.007 gap to NFR-2.1 (≥0.75) and raise specificity toward ≥0.65 |

---

## 4. Data Pipeline

### 4.1 Sources

| Source | Role | Records | Native Hz | Duration | Why it's used this way |
|---|---|---|---|---|---|
| **MIMIC-III Waveform DB** | Phase 1 pre-training | 1,032 (604 AF + 428 control) | 125 | 500–900s | Real ICU noise; long enough for full HRV; labels from ICD-9 codes |
| **MIT-BIH AF Database (afdb)** | Phase 2 fine-tuning | 23 | 250 | ~10h each | Gold-standard, cardiologist-annotated AF |
| **PhysioNet Normal Sinus DB (nsrdb)** | Phase 2 fine-tuning | 18 | 128 | ~24h each | Gold-standard healthy controls |
| **Long-Term AF Database (ltafdb)** | Phase 2 fine-tuning | ~84 extracted AF segments | 128 | 30s–hours | Only the annotated `(AFIB` rhythm segments are extracted from each 21h recording — labeling a whole record AF=1 would mislabel its NSR episodes |
| **PhysioNet Challenge 2017** | **OOD evaluation only, never trained on** | 5,777–5,788 labeled | 300 | 30–61s | AliveCor smartphone ECG — a genuinely different device/population; the only honest test of cross-device generalization |

All signals are resampled to a uniform **250 Hz** (`scipy.signal.resample`, chunked) before anything else touches them. This closes a real bug found during development: at native rates, AF-labeled records (afdb, 250 Hz) always produced more raw samples per 10s window than NSR-labeled records (nsrdb, 128 Hz) before padding — the CNN could trivially learn "sample count" as a proxy for the label instead of learning AF physiology.

### 4.2 Splitting

`splitter.py` creates a **70/15/15 train/val/test split by `subject_id`, not by window**, with a fixed seed (42), persisted to `models/artifacts/split.json` (and phase-specific variants for two-phase training). Splitting by window instead of patient would let the model see windows from the same patient in both train and test, memorize per-patient idiosyncrasies (equipment, individual anatomy) instead of general AF physiology, and report an inflated score. This is the single most important anti-leakage guard in the codebase and is enforced in tests (`test_dataloader.py`).

### 4.3 Windowing

- **CNN input:** 10 seconds of denoised waveform.
- **HRV input:** up to 300 seconds (5 minutes), divided into 60-second sub-windows → up to 5 HRV feature vectors per record (`model.hrv_seq_len = 5`).
- Records shorter than 300s (all of Challenge 2017, some MIMIC-III) still produce a sample: the CNN gets its 10s from the record center, and the HRV sequence gets however many complete 60s sub-windows fit (often just one), with the remaining timesteps left NaN → zero-imputed → **masked out** of the GRU/Transformer via `hrv_lengths` (see §5).

---

## 5. Feature Extraction

| Stage | Module | What it does | Key decision |
|---|---|---|---|
| Denoising | `wavelet_filter.py` | Continuous Wavelet Transform (PyWavelets, `cmor`), reconstructs signal keeping only 0.5–45 Hz scales | CWT chosen over DWT for finer time-frequency localization — needed to preserve P-wave/f-wave morphology, which is exactly what the CNN is supposed to see |
| R-peak detection | `peak_detector.py` | `neurokit2.ecg_peaks()` → R–R intervals | — |
| Artifact handling | `artifact_scrubber.py` | Amplitude check (reject if peak > 30×MAD) + RR-interval check | **Not naive rejection.** Beats deviating >60% from local median RR are *corrected* by interpolation from neighbors; the whole window is only *rejected* if >30% of beats are outliers. This replaced an earlier version that rejected a window if *any* beat deviated >25% — which systematically threw away AF windows, since AF is *defined* by irregularly-irregular RR intervals of 40–60%. That bug alone caused ~89% NaN rates in early HRV features (see §6.2) |
| HRV features (time-domain) | `hrv_time.py` | RMSSD, SDNN (+ pNN50, CoV added in the in-progress Round 10) | Computed per 60s sub-window; needs only 5 RR intervals |
| HRV features (frequency-domain) | `hrv_freq.py` | LF power, HF power, LF/HF ratio via Welch PSD | Needs ≥32 RR intervals (lowered from 64); 60s sub-windows chosen specifically to clear this bar (~70 RR intervals at 70bpm) — 30s windows only produced ~30–50 beats and left 99.97% of frequency features NaN in early rounds |
| HRV features (nonlinear) | `hrv_nonlinear.py` | Sample Entropy (SampEn), DFA α1 | Computed **once over the full 5-minute RR pool** (not per sub-window) and broadcast to every timestep — SampEn needs ~200 RR intervals, DFA needs ~100, far more than any 60s sub-window can supply |
| Scaling | `scaler.py` | Per-column Z-score, fit on train split only, persisted to `scaler.pkl` | Per-column, not per-row: an earlier version required *all* features valid in a row to use it for fitting, which nearly emptied the fit given how sparse HRV features are; per-column fitting lets each feature use its own valid entries independently |

Current fixed feature order (`pipeline.py`, `FEATURE_ORDER`, 7 features — Round 10 is in the process of extending this to 9 with pNN50/CoV): `rmssd, sdnn, lf, hf, lf_hf_ratio, sampen, dfa_alpha1`.

---

## 6. Development History: How the Model Got Here

This project's real engineering value is in the sequence of *diagnose-a-shortcut, fix-the-root-cause, re-measure* iterations below — worth documenting because each is a specific, generalizable lesson for any small-dataset medical ML pipeline.

### 6.1 The 0.9999 AUROC red flag (Round 1)

First training run on 41 raw PhysioNet records hit val AUROC = 0.999999. Suspiciously perfect for a hard clinical problem with so little data. Root cause, confirmed by later cross-dataset testing: the model wasn't learning AF physiology, it was learning **which database a record came from** (afdb vs. nsrdb differ in native sampling rate, equipment, and recording length in ways that correlate perfectly with the label in a 2-source dataset). Lesson institutionalized into the project: **never trust in-distribution AUROC on a small, few-source dataset — always cross-validate on data from a genuinely different institution/device before believing the number.**

### 6.2 The artifact scrubber was rejecting the positive class

The original artifact rejection logic discarded any HRV window where *any single* R-R interval deviated more than 25% from the local median. AF is, by clinical definition, "irregularly irregular" — 40–60% RR deviation is *normal for AF*, not noise. This threshold was quietly deleting the majority of true-AF training windows before the model ever saw them, producing ~89% NaN rates in time-domain HRV features. Fixed by (a) raising the threshold to 60%, (b) switching from "reject if any beat is bad" to "reject only if >30% of beats are bad," and (c) correcting outlier beats by interpolation instead of discarding the whole window — recovering time-domain feature coverage from ~11% to ~60–80%.

### 6.3 Sampling-rate bias

AF-source data (afdb) was natively 250 Hz; NSR-source data (nsrdb) was natively 128 Hz. Before uniform resampling, this meant every AF window had more raw samples than every NSR window after padding — a free, physiology-independent way for the CNN to cheat. Fixed by resampling every source to 250 Hz at parse time.

### 6.4 Infinite entropy poisoning the whole feature

`neurokit2`'s sample-entropy function occasionally returns `inf` (when no template match exists) for a small number of windows. The scaler's NaN-only filter didn't catch `inf`, so it entered the mean/variance computation, producing `mean = inf`, `scale = NaN`, and zeroing out the sample-entropy feature across the *entire* dataset — silently, with no error. Fixed by clamping non-finite values to NaN immediately at the source and switching the scaler's filter from "not NaN" to "is finite" everywhere.

### 6.5 Domain shift from a single-institution dataset

Cross-dataset evaluation on held-out MIMIC-III repeatedly showed the model making **polarized, overconfident, and wrong** predictions (probability ≈ 0 or 1) on ICU data — a strong signature of a model that memorized source-specific noise rather than learning transferable AF physiology. This motivated, in order: expanding to more sources (Challenge 2017, Long-Term AFDB), attention masking so the RNN/Transformer stop treating zero-padded timesteps as real data, on-the-fly signal augmentation (noise, amplitude scaling, baseline wander, dropout, powerline interference — simulating real-world electrode conditions), label smoothing to reduce overconfident OOD predictions, and finally the two-phase transfer-learning architecture (§3.2) that produced the current best result.

### 6.6 Streaming precompute to avoid OOM

Once the dataset grew past ~400,000 candidate windows across 4 sources, the original cache-builder — which loaded every raw signal from every database into RAM before processing any of them — ran out of memory partway through a multi-hour build. Rewritten to stream one record at a time (`iter_all_records()` generator) and switched from a 10s overlapping stride to a non-overlapping 300s stride, cutting peak RAM from ~14 GB to ~350 MB and window count from ~400K near-duplicate windows to ~14K independent ones.

---

## 7. Training Pipeline

- **Precomputed cache** (`precompute_cache.py`, `precompute_worker.py`): denoising + HRV extraction is the expensive step (~0.5s/sample); with 100K+ samples that's 14+ hours per *epoch* done on the fly. Precomputing once (optionally parallelized across CPU cores) and memory-mapping the resulting `.npy` arrays (`mmap_mode="r"`, so multi-GB arrays aren't fully loaded into RAM) brings epoch time down to ~20–25 seconds on a single GPU.
- **Loss:** `BCEWithLogitsLoss` with a **dynamic `pos_weight`** (`n_negative / n_positive`, capped at 10) recomputed from the actual training split each run, so class imbalance is handled by the loss function rather than by discarding real data.
- **Optimizer / schedule:** AdamW, `lr=1e-3`, `ReduceLROnPlateau` (halves LR after 5 epochs without val-AUROC improvement, floor `1e-6`), gradient clipping at `max_norm=1.0` (medical signal batches can occasionally produce outlier gradients), early stopping at 15 epochs' patience on val AUROC, best checkpoint always retained.
- **Label smoothing** (ε=0.1, training loss only — validation keeps hard labels): added because the model was outputting probability ≈0 or ≈1 on nearly every OOD sample; smoothing discourages that overconfidence without touching how validation/eval metrics are computed.
- **Augmentation** (`augmentation.py`, training split only): Gaussian noise (20–40 dB SNR), amplitude scaling (0.8–1.2×), baseline wander (0.1–0.5 Hz), brief signal dropout (loose-electrode simulation), and powerline interference — all standing in for real-world conditions an earpiece sensor will actually encounter (sweat, movement, poor contact) that lab-grade recordings don't have.
- **Two-phase runs:** Phase 1 and Phase 2 each have their own cache directory, split file, scaler, and checkpoint (see `config.yaml` `paths.phase1_*` / `phase2_*`), so the two training regimes never mix state.

---

## 8. Evaluation

Three distinct evaluations answer three distinct questions:

1. **In-distribution test split** (`evaluate.py --split test`) — Phase 2's held-out AFDB/NSRDB/LTAFDB subjects. Answers: "does the fine-tuned head correctly separate clean, curated AF vs. NSR?" (Currently: AUROC ≈ 0.98.)
2. **MIMIC-III evaluation** (`evaluate.py --mimic3`) — historically the cross-dataset generalization check; superseded as the primary OOD metric once MIMIC-III became Phase 1 training data (Round 9), but the evaluation path (`evaluate_mimic3()`, a standalone per-record loop that tolerates variable-length records without needing `PhysioDataset`'s 300s minimum) remains available.
3. **Challenge 2017 OOD evaluation** (`evaluate.py --challenge2017`) — **the metric that matters most.** Ambulatory, single-lead, smartphone-patch ECG from a device and patient population the model has never trained on. This is what NFR-2.1 (AUROC ≥ 0.75) and NFR-3.1 (<5% cross-dataset degradation) are actually judged against. Current result: **AUROC 0.7432, sensitivity 0.9159, specificity 0.3286, F1 0.2815** (F1 is misleading here — C2017 is 87% NSR, so low specificity collapses precision; AUROC and sensitivity are the metrics that reflect the clinical use case).
4. **Threshold analysis** (`evaluate.py`, added Round 10): sweeps the decision threshold and reports Youden's-J-optimal operating point, so the sensitivity/specificity trade-off can be tuned deliberately (e.g., "give me the highest specificity subject to sensitivity ≥ 0.80") instead of defaulting to a fixed 0.5 cutoff that happens to suit neither.

**Why sensitivity > specificity matters clinically here:** a false negative means "AF is present but stimulation is not triggered" (the therapy fails to act when needed); a false positive means "stimulation triggers when the patient is actually in NSR" (an unnecessary but comparatively low-risk stimulation). The current model's 91.6% sensitivity / 32.9% specificity split reflects that asymmetry, though the low specificity is flagged as an open problem — Round 10 exists specifically to raise specificity toward ≥0.65 without giving back the sensitivity gain.

---

## 9. Key Design Decisions (Summary)

| Decision | Alternative considered | Why this way |
|---|---|---|
| Split by patient, not by window | Random window split | Prevents the model from memorizing per-patient signal quirks and reporting an inflated score |
| Single `config.yaml` for every hyperparameter/path | Hardcoded constants scattered in code | Reproducibility, and a single place to update safety-relevant thresholds (artifact rejection, watchdog limits) for FDA-facing traceability |
| CWT (not DWT) for denoising | Discrete Wavelet Transform | Better time-frequency localization, preserves the fine waveform features (P-wave/f-wave) the CNN depends on |
| Correct-then-conditionally-reject artifacts | Reject on any RR outlier | AF *is* RR irregularity; rejecting on it deletes the target class |
| Two-phase transfer learning | Single-phase joint training on all sources | Removes the 93%-zero-padded-HRV problem caused by mixing short (C2017) and long records in one training set; lets a noisy-but-plentiful source (MIMIC) teach robustness and a clean-but-scarce source (AFDB/NSRDB/LTAFDB) teach discrimination |
| Challenge 2017 as pure OOD eval, never trained on | Include C2017 in training for more data | The only way to measure true cross-device generalization is to never let the model see that device during training |
| GRU + Transformer on the same HRV sequence (not one or the other) | Pick one | They're complementary (sequential vs. attention-based) and cheap to fuse; letting the head learn the weighting beats hand-picking |
| Dynamic `pos_weight`, recomputed per run | Fixed class weight, or oversampling | Adapts automatically as the training-data mix changes across rounds without manual retuning |
| Precompute + memory-map cache | Compute features on the fly every epoch | ~2500x epoch-time reduction (14h → ~22s), makes rapid iteration possible on a single desktop GPU |
| MockSparrow / all hardware deferred | Build against real hardware from the start | Validates whether the *algorithm* is sound (the primary research risk) before spending time on hardware integration, which has its own independent risk (Sparrow/FDA limits, LSL streaming, latency) |

---

## 10. Current Status & Open Issues

| Item | Status |
|---|---|
| NFR-2.1: AUROC ≥ 0.75 on OOD (Challenge 2017) | 0.7432 — gap of 0.007, closing via Round 10 |
| Sensitivity on OOD | 0.9159 — clinically strong |
| Specificity on OOD | 0.3286 — **known weak point**, actively being addressed |
| In-distribution (Phase 2 test split) | AUROC ≈ 0.98 |
| Test suite | 61+ tests passing as of the last full run cited in the log (growing each round) |
| LSL streaming, live inference, Sparrow API, watchdog, state machine | **Deferred** — not built yet; scoped for "Live Integration" phase after algorithm validation is complete |
| Latency profiling (<200ms target), 24h hardware-in-the-loop test | **Deferred** to Live Integration |

**Round 10 (in progress)** is tackling the specificity gap and the residual AUROC gap via four changes, in increasing order of effort/risk: (1) threshold tuning — free, no retraining; (2) two new time-domain HRV features (pNN50, coefficient of variation) plus MIMIC-derived healthy controls added to Phase 2 fine-tuning data; (3) partial (rather than total) freezing of the CNN backbone in Phase 2, so the last convolutional block can adapt slightly to the clean fine-tuning distribution instead of staying fully fixed.

---

## 11. Requirements Traceability

| Requirement | Meaning | Where it's implemented |
|---|---|---|
| FR-1.1 | LSL streaming ≥250 Hz | `config.lsl.expected_hz`; streamer deferred |
| FR-1.2 | Multi-institutional datasets | `download_*.py`, `dataset_parsers.py` |
| FR-2.1 | Wavelet denoising | `wavelet_filter.py` |
| FR-2.2 | HRV windows (30s/5min) | `hrv_*.py`, `pipeline.py`, `dataloaders.py` |
| FR-3.1/3.2/3.3 | CNN / RNN / Transformer branches | `cnn.py` / `rnn.py` / `transformer.py` |
| FR-4.2 | Stimulation safety limits | `config.watchdog`; `watchdog.py` deferred |
| NFR-2.1 | AUROC ≥ 0.75 | `evaluate.py` |
| NFR-3.1 | <5% cross-dataset degradation | Challenge 2017 / MIMIC-III eval |
| TR-1.1 | Subject-level holdout | `splitter.py`, `get_dataloaders()` |
| TR-2.3 | Multi-channel → single-channel (Galea → Sparrow) | `pca_reduction.py` |

---

## 12. How To Reproduce

```powershell
# One-time setup
.venv\Scripts\python -m src.data.download_physionet         # afdb + nsrdb
.venv\Scripts\python -m src.data.download_challenge2017      # OOD eval set
.venv\Scripts\python -m src.data.download_ltafdb              # extra AF segments
# MIMIC-III requires PhysioNet credentialing — see README.md

.venv\Scripts\python -m pytest tests/ -v

# Two-phase build + train
.venv\Scripts\python -m src.training.precompute_cache --config config.yaml --phase 1 --workers 4
.venv\Scripts\python -m src.training.train --use-cache --phase 1
.venv\Scripts\python -m src.training.precompute_cache --config config.yaml --phase 2 --workers 4
.venv\Scripts\python -m src.training.train --use-cache --phase 2 --phase1-checkpoint models/checkpoints/phase1_model.pth

# Evaluate
.venv\Scripts\python -m src.training.evaluate --split test          # in-distribution
.venv\Scripts\python -m src.training.evaluate --challenge2017        # true OOD metric
```

Full setup instructions (including MIMIC-III credentialing) are in `README.md`; the running engineering log with every round's raw numbers and rationale is in `progress.txt`; the live task list is `PLAN.md`.
