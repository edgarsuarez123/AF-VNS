# Stroke AVNS — Pipeline Plan

## Context

Branch: `feature/stroke-avns`

This is **not** a classification problem. The AF pipeline asks: is this ECG AF or NSR?
The stroke pipeline asks: did the autonomic nervous system respond to stimulation, how strongly,
and did stimulation condition (bi-fold synchronized, cardiac-gated, open-loop) produce different
responses? Labels are population categories (stroke vs control) or stimulation conditions.
Evaluation is HRV trajectory analysis — does RMSSD increase from baseline during stim, and does
it differ across conditions?

**Everything is additive. AF pipeline is never touched.**

---

## Why the Backbone Transfers

HybridEnsemble (CNN + GRU + Transformer) extracts per-60s-subwindow features:
RMSSD, SDNN, LF, HF, LF/HF, SampEn, DFA α1 — exactly the HRV Endpoint Report biomarkers
for aVNS response. The backbone learned to extract these under real-world noise. Only the
head, loss, and evaluation layer change.

---

## Datasets

| Dataset | Role | Records | Status |
|---------|------|---------|--------|
| `mimic3_stroke/` | Phase 1 pre-training | 200 stroke + 200 control (≥500s) | Downloaded (partial) — blocked on PhysioNet creds |
| `cerevasc/` | Phase 2 fine-tuning | 60 stroke + 60 control, 24h ECG | NOT downloaded — needs PhysioNet Class 2 credentials |
| `shareedb/` | OOD evaluation | 139 hypertensive, 17 event patients | Downloaded ✓ |

**Credential blocker:** Request PhysioNet Class 2 access at physionet.org/content/cerevasc/
for both `cerevasc` and `shareedb` (Italian Holter, may already be Class 2).

---

## New Files (all additive — nothing in AF codebase is modified)

| File | Purpose |
|------|---------|
| `src/data/stroke_parsers.py` | StrokeRecordDict schema + parse_cerevasc_dir(), parse_mimic3_stroke_dir(), parse_sharee_dir(), parse_pilot_session() |
| `src/data/stroke_dataloaders.py` | StrokeDataset (population label) + StrokeSessionDataset (epoch-aware, pilot data) |
| `src/models/stroke_head.py` | StrokeResponderHead (binary BCE), StrokeClassificationHead (3-way CE), StrokeRegressionHead (MSE/Huber) |
| `src/models/stroke_ensemble.py` | StrokeHybridEnsemble — shares CNN/GRU/Transformer encoders, pluggable head, independent from HybridEnsemble |
| `src/training/stroke_train.py` | --phase (1 or 2), --head (responder/classifier/regression) |
| `src/training/stroke_evaluate.py` | AUROC for cerevasc + SHAREE OOD; within-subject RMSSD delta + Wilcoxon for pilot data |
| `src/training/stroke_precompute_cache.py` | Writes cache_stroke_phase1/, cache_stroke_phase2/, stroke_phase1_scaler.pkl, stroke_phase2_scaler.pkl |
| `src/data/download_cerevasc.py` | Download cerevasc via wfdb (Class 2) |
| `config_stroke.yaml` | Copy of config.yaml + stroke: section (epoch durations, conditions, n_conditions) — AF config.yaml untouched |

---

## StrokeRecordDict Schema

```python
{
    "subject_id": str,
    "signal": np.ndarray,
    "fs": float,
    "label": int,           # stroke=1, control=0 (population); condition label for pilot
    "session_id": str,      # e.g. "ses01_bifold"
    "epoch_type": str,      # "baseline" | "stim" | "recovery"
    "condition": str | None # "bifold" | "cardiac_gated" | "open_loop" | None
}
```

---

## Two-Phase Training

**Phase 1** — Full backbone + StrokeResponderHead on MIMIC-3 stroke.
All layers trainable. Target val AUROC on MIMIC-3 holdout: 0.65–0.70 (ICU labels noisy).

**Phase 2** — Freeze backbone, fine-tune head only on cerevasc.
Load Phase 1 checkpoint. ~16.5K head params trainable.
Target val AUROC on cerevasc test split (stroke vs control): ≥0.70.

**OOD** — SHAREE after Phase 2.
Label: 17 event patients = 1, 122 no-event = 0 (testing cross-device generalization).
Target AUROC ≥0.65.

**Pilot data (future)** — StrokeSessionDataset + StrokeRegressionHead.
Leave-one-subject-out CV. Deliverable: within-subject RMSSD delta across 3 stim conditions
with paired Wilcoxon signed-rank test.

**Backbone init:** Try both AF phase2_model.pth and stroke Phase 1 checkpoint as starting
points for Phase 2. Use whichever gives better cerevasc val AUROC. Document in progress.txt.

---

## Steps

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| 1 | Audit downloaded data — record counts, signal types, label availability per dataset | Pending | |
| 2 | Create config_stroke.yaml (copy config.yaml + stroke: section + path overrides) | Done | 2026-03-19 |
| 3 | Write StrokeRecordDict schema + stroke_parsers.py skeleton | Done | 2026-03-19 |
| 4 | Implement parse_mimic3_stroke_dir() — reuse _select_ecg_channel, _resample_to_target_fs | Done | 2026-03-19 |
| 5 | Implement parse_sharee_dir() — inspect SHAREE channel headers first (Italian Holter lead names) | Done | 2026-03-19 |
| 6 | Write download_cerevasc.py (pending Class 2 credentials) | Pending | |
| 7 | Implement parse_cerevasc_dir() once data is available | Pending | |
| 8 | Write StrokeDataset in stroke_dataloaders.py (rolling window, population label) | Done | 2026-03-19 |
| 9 | Write stroke_precompute_cache.py (Phase 1 cache from MIMIC-3 stroke) | Done | 2026-03-19 |
| 10 | Write stroke_head.py — StrokeResponderHead (binary BCE) | Done | 2026-03-19 |
| 11 | Write stroke_ensemble.py — StrokeHybridEnsemble with pluggable head | Done | 2026-03-19 |
| 12 | Write stroke_train.py Phase 1 (MIMIC-3 stroke, StrokeResponderHead) | Pending | |
| 13 | Write stroke_evaluate.py — AUROC + sensitivity/specificity | Pending | |
| 14 | Tests for parsers, dataset, cache builder | Pending | |
| 15 | Run Phase 1 cache rebuild (background) | Pending | |
| 16 | Run Phase 1 training (background) | Pending | |
| 17 | Phase 2 cache rebuild on cerevasc (pending credentials) | Pending | |
| 18 | Phase 2 training — compare AF backbone init vs stroke Phase 1 init | Pending | |
| 19 | OOD evaluation on SHAREE | Pending | |
| 20 | StrokeSessionDataset + StrokeRegressionHead (pilot data, future) | Pending | |

---

## What Stays Untouched

`src/features/`, `src/models/cnn.py`, `src/models/rnn.py`, `src/models/transformer.py`,
`src/models/ensemble.py`, `src/training/train.py`, `src/training/evaluate.py`,
`src/training/precompute_cache.py`, `config.yaml`, all AF caches, both AF checkpoints,
all existing tests. AF checkpoints may be loaded **read-only** as backbone init — never overwritten.

---

## Resume From Here

**Current state (2026-03-19) — Step 12 next:**
- Steps 2–5 complete: config_stroke.yaml, stroke_parsers.py, 8/8 tests passing
- Step 8 complete: stroke_dataloaders.py, 6/6 tests passing
- Step 9 complete: stroke_precompute_cache.py, 6/6 tests passing
- Steps 10–11 complete: stroke_head.py + stroke_ensemble.py (StrokeResponderHead, StrokeHybridEnsemble, build_stroke_model), 10/10 tests passing
- cerevasc ✗ blocked on Class 2 credentials — request at physionet.org/content/cerevasc/
- Step 12 next: stroke_train.py Phase 1 (MIMIC-3 stroke, StrokeResponderHead)
