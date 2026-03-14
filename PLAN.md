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
| 20b | Download ltafdb + AF segment parser | In Progress | |
| 20c | Add pos_weight for class imbalance | Done | 2026-03-14 |
| 21 | Update split + rebuild cache | Not Started | |
| 22 | Retrain + cross-dataset eval | Not Started | |

---

## Step 19: Fix NaN + clean baseline metrics

**Already done (uncommitted):**
- `dataset_parsers.py`: `np.nan_to_num(signal, nan=0.0)` in `parse_mimic3_wfdb_record()`
- `evaluate.py`: `holdout_path` param + `--holdout` CLI flag
- `create_mixed_split.py`: Mixed split script (will be used differently in Step 21)

**Execute:**
1. Update `progress.txt` section 2.14: NaN root cause + fix
2. Update `PLAN.md` with Round 5 plan
3. Run `pytest tests/ -v` — must still be 37 passing
4. Run clean MIMIC eval: `.venv/Scripts/python -m src.training.evaluate --mimic3`
5. Commit: `fix: sanitize MIMIC ICU NaN signals at parse time; add holdout eval mode`

---

## Step 20: Download PhysioNet 2017 AF Challenge data

**Dataset:** PhysioNet Computing in Cardiology Challenge 2017
- 8,528 records: 771 AF + 5,154 Normal + 2,557 Other + 46 Noisy
- Format: WFDB (.hea + .mat), 300 Hz, 30–61 seconds per record
- Labels: in `REFERENCE.csv` (A=AF, N=Normal, O=Other, ~=Noisy)

**Write:** `src/data/download_challenge2017.py` — download + extract + create .label files
**Modify:** `src/data/dataset_parsers.py` — add `parse_challenge2017_dir()` + add to `parse_all()`

Commit: `feat: add PhysioNet 2017 AF Challenge parser and downloader`

---

## Step 21: Update split + rebuild cache

- Delete `split.json` and let `get_dataloaders(create_split_if_missing=True)` regenerate
- New split includes: afdb + nsrdb + challenge2017 subjects
- MIMIC records stay as pure holdout (not in split.json)
- Rebuild cache in background (~1–4 hours)

Commit: `feat: regenerate split with Challenge 2017, rebuild cache`

---

## Step 22: Retrain + cross-dataset eval

- Train: `.venv/Scripts/python -m src.training.train --use-cache`
- Eval: `.venv/Scripts/python -m src.training.evaluate --mimic3`
- Target: MIMIC holdout AUROC significantly above 0.6667 baseline
- NFR-3.1: F1 degradation < 5% from in-distribution test set

---

## Resume From Here

**Current state (2026-03-14):**
- Rounds 1-4 complete. 37/37 tests passing.
- Step 19 done: NaN fix committed, clean MIMIC baseline AUROC=0.6656
- Step 20 done: Challenge 2017 downloaded (738 AF + 5,050 Normal)
- Step 20b: ltafdb downloading in background (~84 records, ~3GB)
- Step 20c done: pos_weight added to training loop
- After ltafdb: ~3,415 AF vs ~5,068 Normal (1:1.5 ratio, pos_weight handles rest)
- **Next:** Wait for ltafdb download → delete split.json → rebuild cache → train
