# AF VNS — Round 5: NaN Fix + Data Expansion + Mixed-Source Training

## Context

Rounds 1–4 complete. Current state:
- 41 PhysioNet training records → val_auroc=0.9999 (fake — learned equipment signature, not AF physiology)
- MIMIC-III cross-dataset eval: AUROC=0.6667, F1=0.6667 → **NFR-3.1 FAIL**
- 10/60 MIMIC records had NaN logits (ICU missing-data NaN values in WFDB p_signal)
- **Dataset is too small:** 23 AF + 18 NSR training records. Industry minimum: 500+ per class

**Root problem:** Model cannot generalize because it only saw 2 recording sources (afdb + nsrdb). Any new ECG equipment looks foreign. Need diverse training data.

**Strategy:** Keep all 60 MIMIC as holdout. Download PhysioNet 2017 AF Challenge (5,788 labeled) + Long-Term AF Database (2,654 AF segments from 84 records) for balanced training. Dynamic pos_weight handles residual imbalance.

---

## Status

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| 19 | NaN fix + clean baseline metrics | Done | 2026-03-14 |
| 20 | Download PhysioNet 2017 AF Challenge + parser | Done | 2026-03-14 |
| 20b | Download ltafdb + AF segment parser | Done | 2026-03-14 |
| 20c | Add pos_weight for class imbalance | Done | 2026-03-14 |
| 21 | Fix OOM: streaming precompute + stride fix | Done | 2026-03-15 |
| 21b | Delete stale split, rebuild cache | In Progress | |
| 22 | Retrain + cross-dataset eval | Not Started | |

---

## Step 21: Fix OOM — Streaming Precompute + Stride Fix (Done 2026-03-15)

**Problem:** Cache rebuild crashed with OOM — `parse_all()` loaded all ~9 GB of raw signals into RAM simultaneously. Additionally, stride=10s with 300s HRV window created ~245K near-duplicate overlapping windows (96.7% overlap).

**Root causes & fixes:**

1. **OOM** — Added `iter_all_records()` generator to `dataset_parsers.py`: yields one record at a time instead of loading all into a list. Peak memory: ~350 MB (was ~14 GB).

2. **Window explosion** — Added `stride_sec: 300` to config.yaml: non-overlapping 5-min windows. Reduces ~245K windows to ~14K independent windows. Cache time: ~45 min (was 17+ hour crash).

3. **Stale split** — Old `split.json` had 2,654 ltafdb subjects despite `max_ltaf_segments: 84` cap. Added `collect_all_subject_ids()` for lightweight split creation without loading signals. Added `create_split_from_ids()` to splitter.

4. **Rewrote `precompute_cache.py`** — Streams records via `iter_all_records()`, builds windows per-record, supports parallel workers via existing `process_chunk`, accumulates only small processed results (~100 MB total).

**Tests:** 54/54 passing (37 existing + 17 new in `tests/test_precompute.py`).

---

## Step 21b: Delete Stale Split + Rebuild Cache

1. Delete `models/artifacts/split.json` (stale — 2,654 ltaf subjects)
2. Run: `.venv/Scripts/python -m src.training.precompute_cache --config config.yaml --workers 4`
3. Precompute auto-creates new split.json with correct IDs (84 ltaf, 5,788 c17, 41 original)
4. Verify cache files in `models/artifacts/cache/`

---

## Step 22: Retrain + cross-dataset eval

- Train: `.venv/Scripts/python -m src.training.train --use-cache`
- Eval: `.venv/Scripts/python -m src.training.evaluate --mimic3`
- Target: MIMIC holdout AUROC significantly above 0.6656 baseline
- NFR-3.1: F1 degradation < 5% from in-distribution test set

---

## Resume From Here

**Current state (2026-03-15):**
- Rounds 1-4 complete. 54/54 tests passing.
- Step 19 done: NaN fix committed, clean MIMIC baseline AUROC=0.6656
- Step 20/20b/20c done: Challenge 2017 + ltafdb downloaded, pos_weight added
- Step 21 done: Streaming precompute fix committed (OOM + stride fix)
- Step 21b next: Delete stale split.json, run cache rebuild with --workers 4

**Next:**
1. Delete stale split: `del models\artifacts\split.json`
2. Run cache: `.venv/Scripts/python -m src.training.precompute_cache --config config.yaml --workers 4`
3. Verify cache: check `models/artifacts/cache/cache_meta.json`
4. Train: `.venv/Scripts/python -m src.training.train --use-cache`
5. Eval: `.venv/Scripts/python -m src.training.evaluate --mimic3`
6. Target: MIMIC holdout AUROC significantly above 0.6656 baseline
