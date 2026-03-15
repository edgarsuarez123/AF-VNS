# AF VNS — Round 6: MIMIC Mixed Training + Attention Masking + Data Augmentation

## Context

Round 5 MIMIC-III AUROC = 0.6022 despite expanding from 41 to 5,915 training records.
Root cause: **domain shift** — all training data is research/portable-grade; MIMIC is ICU bedside.
Adding more PhysioNet data doesn't bridge the gap. The model needs ICU signals during training.

Additionally: 36% of training samples have 4/5 zero-padded HRV timesteps (GRU/Transformer
learn padding shortcuts), and no augmentation exists for real-world noise conditions.

---

## Status

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| 23a | Snapshot 60 MIMIC holdout IDs to JSON | Done | 2026-03-15 |
| 23b | Download script: skip-if-exists + env var creds | Done | 2026-03-15 |
| 23c | Download 400 MIMIC records (manual) | Skipped | — needs creds; current 60 used as holdout only |
| 23d | Add mimic3_holdout config key | Done | 2026-03-15 |
| 23e | Include MIMIC train records in data pipeline | Done | 2026-03-15 |
| 23f | Default evaluate to holdout-only | Done | 2026-03-15 |
| 23g | Tests for MIMIC holdout logic | Done | 2026-03-15 |
| 24a | Save HRV lengths in cache | Done | 2026-03-15 |
| 24b | PrecomputedDataset returns lengths | Done | 2026-03-15 |
| 24c | GRU: pack_padded_sequence | Done | 2026-03-15 |
| 24d | Transformer: src_key_padding_mask | Done | 2026-03-15 |
| 24e | Ensemble passes lengths through | Done | 2026-03-15 |
| 24f | Training loop passes lengths | Done | 2026-03-15 |
| 24g | Evaluate passes lengths | Done | 2026-03-15 |
| 24h | Tests for attention masking | Done | 2026-03-15 |
| 25a | Create augmentation module | Done | 2026-03-15 |
| 25b | Add augmentation config section | Done | 2026-03-15 |
| 25c | Integrate augmentation into PrecomputedDataset | Done | 2026-03-15 |
| 25d | Tests for augmentation | Done | 2026-03-15 |
| 26a | Delete stale cache + rebuild | Not Started | |
| 26b | Train | Not Started | |
| 26c | Evaluate on MIMIC holdout | Not Started | |

---

## Resume From Here

**Current state (2026-03-15):**
- Steps 23-25 complete. 75/75 tests passing.
- Step 23c skipped: downloading 400 MIMIC records requires PhysioNet creds.
  Current 60 records serve as holdout-only for eval. New records downloaded
  later will automatically be included in training (holdout JSON excludes them).
- Step 26 next: delete stale cache, rebuild (includes HRV lengths), train, eval.

**Next:**
1. Delete stale cache: `del models\artifacts\cache\*.npy` + `del models\artifacts\scaler.pkl` + `del models\artifacts\split.json`
2. Rebuild cache: `.venv/Scripts/python -m src.training.precompute_cache --config config.yaml --workers 4`
3. Train: `.venv/Scripts/python -m src.training.train --use-cache`
4. Eval: `.venv/Scripts/python -m src.training.evaluate --mimic3`
5. Target: AUROC > 0.75 on holdout
