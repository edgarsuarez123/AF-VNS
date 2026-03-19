"""
Stroke AVNS — dataset parsers.

StrokeRecordDict schema:
    subject_id  str
    signal      np.ndarray  1D float64
    fs          float
    label       int         1=stroke/event, 0=control/no-event
    session_id  str | None  e.g. "ses01_bifold" — None for population datasets
    epoch_type  str | None  "baseline"|"stim"|"recovery" — None for population data
    condition   str | None  "bifold"|"cardiac_gated"|"open_loop" — None for population data

Nothing in the AF pipeline is imported or modified here.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.dataset_parsers import (
    _select_ecg_channel,
    _resample_to_target_fs,
    load_config,
)

logger = logging.getLogger(__name__)

StrokeRecordDict = Dict[str, Any]


def parse_mimic3_stroke_dir(data_dir: str, config_path: str) -> List[StrokeRecordDict]:
    """Parse MIMIC-III stroke cohort directory.

    Each record has .hea + .dat + .label (label written by download script).
    Label 1 = stroke, 0 = control.

    Args:
        data_dir:    Path to directory containing .hea/.dat/.label files.
        config_path: Path to config YAML (reads data.target_fs, data.waveform_sec).

    Returns:
        List of StrokeRecordDict; records shorter than waveform_sec are skipped.
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required")

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    target_fs: float = float(data_cfg.get("target_fs", 0))
    waveform_sec: float = float(data_cfg.get("waveform_sec", 10))

    base = Path(data_dir)
    if not base.is_dir():
        logger.warning("mimic3_stroke_dir not found: %s", data_dir)
        return []

    records: List[StrokeRecordDict] = []
    seen: set = set()

    for hea_path in sorted(base.glob("*.hea")):
        name = hea_path.stem
        if name in seen:
            continue
        seen.add(name)

        # Read label — skip records without label file
        label_path = base / f"{name}.label"
        if not label_path.exists():
            logger.debug("No label file for %s — skipping", name)
            continue
        label = int(label_path.read_text().strip())

        try:
            record = wfdb.rdrecord(str(base / name))
        except Exception as e:
            logger.warning("Failed to read MIMIC-III stroke record %s: %s", name, e)
            continue
        if record is None:
            continue

        signal = record.p_signal if record.p_signal is not None else record.d_signal
        if signal is None:
            continue

        signal = np.asarray(signal, dtype=np.float64)

        # Select best ECG channel
        if signal.ndim == 2 and signal.shape[1] > 1:
            ch_idx = _select_ecg_channel(record.sig_name, record.units)
            signal = signal[:, ch_idx]
        else:
            signal = signal.ravel()

        # Sanitize NaN (ICU signals use NaN for missing samples)
        signal = np.nan_to_num(signal, nan=0.0)

        fs = float(record.fs)

        # Resample to target_fs if configured
        if target_fs > 0 and abs(fs - target_fs) >= 0.1:
            signal = _resample_to_target_fs(signal, fs, target_fs)
            fs = target_fs

        # Skip records too short for waveform window
        min_samples = int(waveform_sec * fs)
        if len(signal) < min_samples:
            logger.debug(
                "Skipping %s — signal too short (%d samples < %d required)",
                name, len(signal), min_samples,
            )
            continue

        records.append({
            "subject_id": name,
            "signal": signal,
            "fs": fs,
            "label": label,
            "session_id": None,
            "epoch_type": None,
            "condition": None,
        })

    logger.info("Parsed %d records from mimic3_stroke (dir: %s)", len(records), data_dir)
    return records


def parse_sharee_dir(
    data_dir: str,
    label_map: Dict[str, int],
    config_path: str,
) -> List[StrokeRecordDict]:
    """Parse SHAREE holter database.

    SHAREE has no label files — labels come from an external metadata dict.
    Leads: III, V3, V5 (8-bit, 128 Hz). Records with no entry in label_map
    receive label=0 (no cardiovascular event).

    Args:
        data_dir:   Path to shareedb directory.
        label_map:  Dict mapping record_id (str) to binary label (0 or 1).
                    17 event patients → 1, remainder → 0.
        config_path: Path to config YAML (reads data.target_fs, data.waveform_sec).

    Returns:
        List of StrokeRecordDict.
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required")

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    target_fs: float = float(data_cfg.get("target_fs", 0))
    waveform_sec: float = float(data_cfg.get("waveform_sec", 10))

    base = Path(data_dir)
    if not base.is_dir():
        logger.warning("sharee_dir not found: %s", data_dir)
        return []

    records: List[StrokeRecordDict] = []
    seen: set = set()

    for hea_path in sorted(base.glob("*.hea")):
        name = hea_path.stem
        if name in seen:
            continue
        seen.add(name)

        label = label_map.get(name, 0)

        try:
            record = wfdb.rdrecord(str(base / name))
        except Exception as e:
            logger.warning("Failed to read SHAREE record %s: %s", name, e)
            continue
        if record is None:
            continue

        signal = record.p_signal if record.p_signal is not None else record.d_signal
        if signal is None:
            continue

        signal = np.asarray(signal, dtype=np.float64)

        if signal.ndim == 2 and signal.shape[1] > 1:
            ch_idx = _select_ecg_channel(record.sig_name, record.units)
            signal = signal[:, ch_idx]
        else:
            signal = signal.ravel()

        signal = np.nan_to_num(signal, nan=0.0)

        fs = float(record.fs)

        if target_fs > 0 and abs(fs - target_fs) >= 0.1:
            signal = _resample_to_target_fs(signal, fs, target_fs)
            fs = target_fs

        min_samples = int(waveform_sec * fs)
        if len(signal) < min_samples:
            logger.debug("Skipping SHAREE %s — too short", name)
            continue

        records.append({
            "subject_id": name,
            "signal": signal,
            "fs": fs,
            "label": label,
            "session_id": None,
            "epoch_type": None,
            "condition": None,
        })

    logger.info("Parsed %d records from shareedb (dir: %s)", len(records), data_dir)
    return records


def parse_cerevasc_dir(data_dir: str, config_path: str) -> List[StrokeRecordDict]:
    """Parse CereVasc dataset (PhysioNet, requires Class 2 credentials).

    This dataset is not yet available locally. Access requires PhysioNet Class 2
    credentialing at: https://physionet.org/content/cerevasc/

    Raises:
        NotImplementedError: Always — until credentials are obtained and data is downloaded.
    """
    raise NotImplementedError(
        "parse_cerevasc_dir() is not yet implemented. "
        "The CereVasc dataset requires PhysioNet Class 2 credentials. "
        "Request access at https://physionet.org/content/cerevasc/ and then "
        "download with src/data/download_cerevasc.py once approved."
    )
