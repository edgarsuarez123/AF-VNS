"""
Convert WFDB, EDF, and MIMIC-III raw files into a single in-memory schema.

Standard schema per record: subject_id (str), signal (1D ndarray), fs (float), label (int | None).
ECG/PPG channel mapping: WFDB first channel; EDF/MIMIC per config/comment.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import yaml
from scipy.signal import resample as _scipy_resample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RecordDict = Dict[str, Any]  # subject_id, signal, fs, label

# Lead priority for ECG channel selection in multi-channel MIMIC-III records
_ECG_PRIORITY = ["II", "I", "III", "V", "MCL", "MCL1", "aVR", "AVR", "AVL", "AVF"]


def _select_ecg_channel(sig_names: list, units: list) -> int:
    """Return index of best ECG channel by priority. Fallback: first mV channel, then 0."""
    names_upper = [n.strip().upper() for n in sig_names]
    for lead in _ECG_PRIORITY:
        lead_up = lead.upper()
        if lead_up in names_upper:
            return names_upper.index(lead_up)
    # Fallback: first channel with mV units
    for i, u in enumerate(units):
        if "mv" in u.lower():
            return i
    return 0


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config from project root if needed."""
    import os
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        root = Path(__file__).resolve().parents[2]
        config_path = root / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def _ensure_1d(signal: np.ndarray) -> np.ndarray:
    """Take first channel if 2D, else return 1D."""
    if signal.ndim == 2:
        return signal[0] if signal.shape[0] < signal.shape[1] else signal[:, 0]
    return signal


def _resample_to_target_fs(signal: np.ndarray, fs_orig: float, fs_target: float) -> np.ndarray:
    """Resample signal from fs_orig to fs_target. Chunks long signals to avoid OOM."""
    if abs(fs_orig - fs_target) < 0.1:
        return signal
    ratio = fs_target / fs_orig
    # Chunk at ~60s to keep memory bounded (~15K samples per chunk at 250 Hz)
    chunk_samples = int(fs_orig * 60)
    if len(signal) <= chunk_samples * 2:
        n_target = int(len(signal) * ratio)
        return _scipy_resample(signal, n_target).astype(np.float64)
    # Resample in chunks and concatenate
    chunks = []
    for start in range(0, len(signal), chunk_samples):
        seg = signal[start : start + chunk_samples]
        n_out = int(len(seg) * ratio)
        chunks.append(_scipy_resample(seg, n_out).astype(np.float64))
    return np.concatenate(chunks)


def parse_wfdb_record(record_dir: Path, record_name: str, db_name: str) -> Optional[RecordDict]:
    """Parse one WFDB record; return standard dict or None on failure."""
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required")

    record_path = record_dir / record_name
    if not (record_path.with_suffix(".hea")).exists():
        return None
    try:
        # Pass full path for local files (wfdb accepts path/to/record without extension)
        record = wfdb.rdrecord(str(record_path))
    except Exception:
        return None
    if record is None:
        return None

    signal = record.p_signal if record.p_signal is not None else record.d_signal
    if signal is None:
        return None
    signal = _ensure_1d(np.asarray(signal, dtype=np.float64))

    fs = float(record.fs)
    label: Optional[int] = None
    if db_name == "afdb":
        try:
            ann = wfdb.rdann(str(record_path), "atr")
        except Exception:
            ann = None
        if ann is not None and hasattr(ann, "aux_note"):
            # AF rhythm labels in MIT-BIH AF db are in aux_note (e.g. '(AFIB', '(AFL')
            # symbol contains beat codes (N, V, Q) which never include 'AF'
            notes = "".join(ann.aux_note) if isinstance(ann.aux_note, (list, np.ndarray)) else str(ann.aux_note)
            label = 1 if "AFIB" in notes or "AFL" in notes or "(AF" in notes else 0
        else:
            label = 1  # afdb is AF cohort
    elif db_name == "nsrdb":
        label = 0  # Normal sinus rhythm

    return {
        "subject_id": record_name,
        "signal": signal,
        "fs": fs,
        "label": label,
    }


def parse_mimic3_wfdb_record(record_dir: Path, record_name: str) -> Optional[RecordDict]:
    """Parse one MIMIC-III WFDB record + .label file, selecting best ECG channel."""
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required")

    record_path = record_dir / record_name
    if not record_path.with_suffix(".hea").exists():
        return None
    try:
        record = wfdb.rdrecord(str(record_path))
    except Exception as e:
        logger.warning("Failed to read MIMIC-III WFDB %s: %s", record_name, e)
        return None
    if record is None:
        return None

    signal = record.p_signal if record.p_signal is not None else record.d_signal
    if signal is None:
        return None
    signal = np.asarray(signal, dtype=np.float64)

    # Select best ECG channel (MIMIC records have mixed channels)
    if signal.ndim == 2 and signal.shape[1] > 1:
        ch_idx = _select_ecg_channel(record.sig_name, record.units)
        signal = signal[:, ch_idx]
    else:
        signal = signal.ravel()

    # Sanitize NaN — MIMIC ICU signals use NaN for missing/invalid samples
    signal = np.nan_to_num(signal, nan=0.0)

    fs = float(record.fs)

    # Read label from .label file (written by download_mimic3_waveforms.py)
    label: Optional[int] = None
    label_path = record_dir / f"{record_name}.label"
    if label_path.exists():
        label = int(label_path.read_text().strip())

    return {
        "subject_id": record_name,
        "signal": signal,
        "fs": fs,
        "label": label,
    }


def _wfdb_dir_with_hea(base: Path) -> Optional[Path]:
    """Return base if it contains .hea files, else first subdir that does (e.g. afdb/files or nsrdb/versioned)."""
    if not base.is_dir():
        return None
    if list(base.glob("*.hea")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and list(sub.glob("*.hea")):
            return sub
    return None


def parse_wfdb_dir(record_dir: Path, db_name: str) -> List[RecordDict]:
    """Parse all WFDB records in a directory (or its first subdir that contains .hea)."""
    records = []
    record_dir = _wfdb_dir_with_hea(record_dir)
    if record_dir is None:
        return records
    hea_files = list(record_dir.glob("*.hea"))
    seen = set()
    for hea in hea_files:
        name = hea.stem
        if name in seen:
            continue
        seen.add(name)
        rec = parse_wfdb_record(record_dir, name, db_name)
        if rec is not None:
            records.append(rec)
    return records


def parse_edf_file(edf_path: Path, channel_idx: int = 0) -> Optional[RecordDict]:
    """Parse one EDF file with MNE; primary channel at channel_idx (default ECG/PPG)."""
    try:
        import mne
    except ImportError:
        raise ImportError("mne is required for EDF")

    raw = mne.io.read_raw_edf(str(edf_path), verbose=False)
    data, times = raw.get_data(return_times=True)
    signal = _ensure_1d(data)
    fs = raw.info["sfreq"]
    subject_id = edf_path.stem
    return {
        "subject_id": subject_id,
        "signal": signal.astype(np.float64),
        "fs": float(fs),
        "label": None,
    }


def parse_ltafdb_dir(ltafdb_dir: Path) -> List[RecordDict]:
    """Parse Long-Term AF Database segments (pre-extracted by download_ltafdb.py).

    Each file is an individual AF segment with .label=1, saved as WFDB format.
    Only parses records that have .label files (AF segments only).
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required")

    records = []
    if not ltafdb_dir.is_dir():
        return records

    label_files = sorted(ltafdb_dir.glob("*.label"))
    for label_path in label_files:
        record_name = label_path.stem
        hea_path = ltafdb_dir / f"{record_name}.hea"
        if not hea_path.exists():
            continue
        try:
            record = wfdb.rdrecord(str(ltafdb_dir / record_name))
        except Exception as e:
            logger.warning("Failed to read ltafdb segment %s: %s", record_name, e)
            continue

        signal = record.p_signal if record.p_signal is not None else record.d_signal
        if signal is None:
            continue
        signal = _ensure_1d(np.asarray(signal, dtype=np.float64))
        signal = np.nan_to_num(signal, nan=0.0)
        fs = float(record.fs)
        label = int(label_path.read_text().strip())

        records.append({
            "subject_id": f"ltaf_{record_name}",
            "signal": signal,
            "fs": fs,
            "label": label,
        })

    return records


def parse_challenge2017_record(record_dir: Path, record_name: str) -> Optional[RecordDict]:
    """Parse one PhysioNet 2017 AF Challenge WFDB record (.hea + .mat) with .label file."""
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required")

    record_path = record_dir / record_name
    if not record_path.with_suffix(".hea").exists():
        return None
    try:
        record = wfdb.rdrecord(str(record_path))
    except Exception as e:
        logger.warning("Failed to read Challenge 2017 record %s: %s", record_name, e)
        return None
    if record is None:
        return None

    signal = record.p_signal if record.p_signal is not None else record.d_signal
    if signal is None:
        return None
    signal = _ensure_1d(np.asarray(signal, dtype=np.float64))

    # Sanitize NaN (shouldn't be common but be safe)
    signal = np.nan_to_num(signal, nan=0.0)

    fs = float(record.fs)

    # Read label from .label file
    label: Optional[int] = None
    label_path = record_dir / f"{record_name}.label"
    if label_path.exists():
        label = int(label_path.read_text().strip())

    return {
        "subject_id": f"c17_{record_name}",  # prefix to avoid ID collision with afdb/nsrdb
        "signal": signal,
        "fs": fs,
        "label": label,
    }


def parse_challenge2017_dir(challenge_dir: Path) -> List[RecordDict]:
    """Parse all PhysioNet 2017 AF Challenge records that have .label files (AF or Normal only)."""
    records = []
    if not challenge_dir.is_dir():
        return records

    # Only parse records that have .label files (AF=1 or Normal=0; Other/Noisy excluded)
    label_files = sorted(challenge_dir.glob("*.label"))
    for label_path in label_files:
        record_name = label_path.stem
        rec = parse_challenge2017_record(challenge_dir, record_name)
        if rec is not None:
            records.append(rec)

    return records


def parse_mimic3_csv(csv_path: Path, subject_id_col: str = "subject_id", time_col: str = "time", signal_col: str = "ecg", label_col: Optional[str] = "label", fs: float = 250.0) -> Optional[RecordDict]:
    """Parse one MIMIC-III CSV; columns: subject_id, time, ecg (or ppg), optional label."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    for col in (signal_col,):
        if col not in df.columns and col == "ecg" and "ppg" in df.columns:
            signal_col = "ppg"
            break
    if signal_col not in df.columns:
        logger.warning("CSV %s missing signal column (tried ecg/ppg).", csv_path)
        return None
    signal = np.asarray(df[signal_col], dtype=np.float64)
    if subject_id_col in df.columns:
        sid = str(df[subject_id_col].iloc[0])
    else:
        sid = csv_path.stem
    label = None
    if label_col and label_col in df.columns:
        # Majority vote or last value for segment
        label = int(df[label_col].mode().iloc[0]) if len(df[label_col].dropna()) else None
    return {
        "subject_id": sid,
        "signal": signal,
        "fs": fs,
        "label": label,
    }


def parse_mimic3_dir(mimic3_dir: Path, fs_default: float = 250.0) -> List[RecordDict]:
    """Parse MIMIC-III directory: WFDB (.hea+.dat+.label), CSV, or EDF."""
    records = []
    if not mimic3_dir.is_dir():
        return records
    seen_hea = set()
    for f in mimic3_dir.iterdir():
        if f.suffix.lower() == ".hea":
            name = f.stem
            # Skip non-record files (e.g. DIAGNOSES_ICD.csv also in directory)
            if name.upper().startswith("DIAGNOSES"):
                continue
            if name in seen_hea:
                continue
            seen_hea.add(name)
            rec = parse_mimic3_wfdb_record(mimic3_dir, name)
            if rec is not None:
                records.append(rec)
        elif f.suffix.lower() == ".csv" and not f.stem.upper().startswith("DIAGNOSES"):
            rec = parse_mimic3_csv(f, fs=fs_default)
            if rec is not None:
                records.append(rec)
        elif f.suffix.lower() == ".edf":
            rec = parse_edf_file(f)
            if rec is not None:
                records.append(rec)
    return records


def parse_all(config_path: str = "config.yaml") -> List[RecordDict]:
    """Parse all configured sources (WFDB afdb/nsrdb, MIMIC-III) into standard list."""
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    mimic3_subdir = Path(data_cfg.get("mimic3_subdir", "data/raw/mimic3"))

    all_records: List[RecordDict] = []

    afdb_path = raw_dir / "afdb"
    if afdb_path.is_dir():
        afdb_recs = parse_wfdb_dir(afdb_path, "afdb")
        all_records.extend(afdb_recs)
        logger.info("Parsed %d records from afdb", len(afdb_recs))
    nsrdb_path = raw_dir / "nsrdb"
    if nsrdb_path.is_dir():
        nsr_recs = parse_wfdb_dir(nsrdb_path, "nsrdb")
        all_records.extend(nsr_recs)
        logger.info("Parsed %d records from nsrdb", len(nsr_recs))

    ltafdb_path = raw_dir / "ltafdb"
    if ltafdb_path.is_dir():
        ltaf_recs = parse_ltafdb_dir(ltafdb_path)
        max_ltaf = data_cfg.get("max_ltaf_segments", None)
        if max_ltaf is not None and len(ltaf_recs) > max_ltaf:
            ltaf_recs = ltaf_recs[:max_ltaf]
            logger.info("Capped ltafdb to %d segments (max_ltaf_segments)", max_ltaf)
        all_records.extend(ltaf_recs)
        logger.info("Parsed %d AF segments from ltafdb", len(ltaf_recs))

    challenge2017_path = raw_dir / "challenge2017"
    if challenge2017_path.is_dir():
        c17_recs = parse_challenge2017_dir(challenge2017_path)
        all_records.extend(c17_recs)
        logger.info("Parsed %d records from Challenge 2017", len(c17_recs))

    # NOTE: MIMIC-III is excluded from parse_all() — it is reserved as pure holdout
    # for cross-dataset evaluation (NFR-3.1). evaluate_mimic3() parses it separately.
    if mimic3_subdir.is_dir() and any(mimic3_subdir.iterdir()):
        logger.info("MIMIC-III dir found (%s) — skipped (holdout only, use evaluate --mimic3)",
                     mimic3_subdir)

    # Resample to uniform rate if target_fs is set (avoids sampling-rate bias in CNN)
    target_fs = float(data_cfg.get("target_fs", 0))
    if target_fs > 0:
        for rec in all_records:
            if abs(rec["fs"] - target_fs) >= 0.1:
                logger.info("Resampling %s from %.0f Hz to %.0f Hz", rec["subject_id"], rec["fs"], target_fs)
                rec["signal"] = _resample_to_target_fs(rec["signal"], rec["fs"], target_fs)
                rec["fs"] = target_fs

    return all_records


def collect_all_subject_ids(config_path: str = "config.yaml") -> List[str]:
    """Collect all subject IDs without loading signal data.

    Scans .hea/.label files on disk to build the same ID list that parse_all()
    would produce — but without ever calling wfdb.rdrecord(). Used for
    lightweight split creation in the streaming precompute pipeline.
    """
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))

    ids: List[str] = []

    # afdb
    afdb_path = raw_dir / "afdb"
    afdb_dir = _wfdb_dir_with_hea(afdb_path)
    if afdb_dir:
        seen = set()
        for hea in sorted(afdb_dir.glob("*.hea")):
            if hea.stem not in seen:
                seen.add(hea.stem)
                ids.append(hea.stem)
        logger.info("Collected %d subject IDs from afdb", len(seen))

    # nsrdb
    nsrdb_path = raw_dir / "nsrdb"
    nsrdb_dir = _wfdb_dir_with_hea(nsrdb_path)
    if nsrdb_dir:
        seen = set()
        for hea in sorted(nsrdb_dir.glob("*.hea")):
            if hea.stem not in seen:
                seen.add(hea.stem)
                ids.append(hea.stem)
        logger.info("Collected %d subject IDs from nsrdb", len(seen))

    # ltafdb — cap at max_ltaf_segments
    ltafdb_path = raw_dir / "ltafdb"
    if ltafdb_path.is_dir():
        max_ltaf = data_cfg.get("max_ltaf_segments", None)
        count = 0
        for label_path in sorted(ltafdb_path.glob("*.label")):
            if max_ltaf is not None and count >= max_ltaf:
                break
            if (ltafdb_path / f"{label_path.stem}.hea").exists():
                ids.append(f"ltaf_{label_path.stem}")
                count += 1
        logger.info("Collected %d subject IDs from ltafdb", count)

    # challenge2017
    c17_path = raw_dir / "challenge2017"
    if c17_path.is_dir():
        count = 0
        for label_path in sorted(c17_path.glob("*.label")):
            if (c17_path / f"{label_path.stem}.hea").exists():
                ids.append(f"c17_{label_path.stem}")
                count += 1
        logger.info("Collected %d subject IDs from challenge2017", count)

    return ids


def iter_all_records(config_path: str = "config.yaml") -> Iterator[RecordDict]:
    """Yield one RecordDict at a time from all configured databases.

    Memory-efficient: only one record's signal is in memory at a time.
    Applies target_fs resampling inline. Databases are iterated in the same
    order as parse_all(): afdb, nsrdb, ltafdb, challenge2017.
    """
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    target_fs = float(data_cfg.get("target_fs", 0))

    def _maybe_resample(rec: RecordDict) -> None:
        if target_fs > 0 and abs(rec["fs"] - target_fs) >= 0.1:
            logger.info("Resampling %s from %.0f Hz to %.0f Hz",
                        rec["subject_id"], rec["fs"], target_fs)
            rec["signal"] = _resample_to_target_fs(rec["signal"], rec["fs"], target_fs)
            rec["fs"] = target_fs

    # afdb — one record at a time
    afdb_dir = _wfdb_dir_with_hea(raw_dir / "afdb")
    if afdb_dir:
        count = 0
        seen = set()
        for hea in sorted(afdb_dir.glob("*.hea")):
            if hea.stem in seen:
                continue
            seen.add(hea.stem)
            rec = parse_wfdb_record(afdb_dir, hea.stem, "afdb")
            if rec is not None:
                _maybe_resample(rec)
                yield rec
                count += 1
        logger.info("Yielded %d records from afdb", count)

    # nsrdb — one record at a time
    nsrdb_dir = _wfdb_dir_with_hea(raw_dir / "nsrdb")
    if nsrdb_dir:
        count = 0
        seen = set()
        for hea in sorted(nsrdb_dir.glob("*.hea")):
            if hea.stem in seen:
                continue
            seen.add(hea.stem)
            rec = parse_wfdb_record(nsrdb_dir, hea.stem, "nsrdb")
            if rec is not None:
                _maybe_resample(rec)
                yield rec
                count += 1
        logger.info("Yielded %d records from nsrdb", count)

    # ltafdb — one segment at a time, cap at max_ltaf_segments
    ltafdb_path = raw_dir / "ltafdb"
    if ltafdb_path.is_dir():
        try:
            import wfdb
        except ImportError:
            raise ImportError("wfdb is required")

        max_ltaf = data_cfg.get("max_ltaf_segments", None)
        count = 0
        for label_path in sorted(ltafdb_path.glob("*.label")):
            if max_ltaf is not None and count >= max_ltaf:
                break
            record_name = label_path.stem
            hea_path = ltafdb_path / f"{record_name}.hea"
            if not hea_path.exists():
                continue
            try:
                record = wfdb.rdrecord(str(ltafdb_path / record_name))
            except Exception as e:
                logger.warning("Failed to read ltafdb segment %s: %s", record_name, e)
                continue
            signal = record.p_signal if record.p_signal is not None else record.d_signal
            if signal is None:
                continue
            signal = _ensure_1d(np.asarray(signal, dtype=np.float64))
            signal = np.nan_to_num(signal, nan=0.0)
            rec: RecordDict = {
                "subject_id": f"ltaf_{record_name}",
                "signal": signal,
                "fs": float(record.fs),
                "label": int(label_path.read_text().strip()),
            }
            _maybe_resample(rec)
            yield rec
            count += 1
        if count > 0:
            logger.info("Yielded %d AF segments from ltafdb", count)

    # challenge2017 — one record at a time
    c17_path = raw_dir / "challenge2017"
    if c17_path.is_dir():
        count = 0
        for label_path in sorted(c17_path.glob("*.label")):
            rec = parse_challenge2017_record(c17_path, label_path.stem)
            if rec is not None:
                _maybe_resample(rec)
                yield rec
                count += 1
        if count > 0:
            logger.info("Yielded %d records from challenge2017", count)
