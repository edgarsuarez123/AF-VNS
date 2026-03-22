"""
Stroke AVNS — dataset parsers.

StrokeRecordDict schema:
    subject_id    str
    signal        np.ndarray  1D float64 (ECG)
    fs            float
    label         int           1=stroke/event, 0=control/no-event
    session_id    str | None    e.g. "ses01_bifold" — None for population datasets
    epoch_type    str | None    "baseline"|"stim"|"recovery" — None for population data
    condition     str | None    "bifold"|"cardiac_gated"|"open_loop" — None for population data
    resp_signal   np.ndarray | None   1D float64 respiratory reference (CVES only)
    resp_channel  str | None          "flow_rate" / "thermst" / "resp" / None

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

# Respiratory channel priority for CVES reference signal extraction
_RESP_PRIORITY = ["flow_rate", "thermst", "resp"]


def _select_resp_channel(sig_names: list):
    """Return (index, name) of best respiratory channel by priority, or (None, None)."""
    names_lower = [n.strip().lower() for n in sig_names]
    for name in _RESP_PRIORITY:
        if name in names_lower:
            return names_lower.index(name), name
    return None, None

# ---------------------------------------------------------------------------
# CVES (Cerebral Vasoregulation in Elderly with Stroke) label maps
# Derived from walking/ protocol subject IDs on PhysioNet:
#   S####S → stroke (1),  S####A → control (0)
# Numeric IDs are zero-stripped (e.g. "0044" → "44") to match stem extraction.
# Hardcoded to avoid runtime network dependency.
# ---------------------------------------------------------------------------
CVES_STROKE_IDS = {
    "30", "44", "64", "67", "68", "78", "121", "132", "154", "157",
    "160", "163", "164", "166", "169", "175", "183", "185", "187", "194",
    "197", "199", "213", "221", "225", "277", "324", "331", "363", "371", "376",
}

CVES_CONTROL_IDS = {
    "165", "172", "176", "178", "184", "200", "203", "204", "205", "207",
    "208", "210", "212", "214", "215", "218", "228", "231", "232", "239",
    "240", "242", "243", "246", "247", "248", "295", "305", "322", "332",
    "334", "336", "337", "340", "343", "351", "353", "354", "358", "364",
    "374", "378", "379", "380", "388", "389", "397", "399", "402",
}

CVES_PROTOCOLS = [
    "sit-stand",
    "head-up-tilt",
    "sit-stand-balance",
    "transcranial-doppler",
]


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
            "resp_signal": None,
            "resp_channel": None,
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
            "resp_signal": None,
            "resp_channel": None,
        })

    logger.info("Parsed %d records from shareedb (dir: %s)", len(records), data_dir)
    return records


def parse_cerevasc_dir(data_dir: str, config_path: str) -> List[StrokeRecordDict]:
    """Parse CVES (Cerebral Vasoregulation in Elderly with Stroke) dataset.

    Dataset is publicly available on PhysioNet (db: cves) — no credentials required.
    Labels are derived from walking/ protocol subject IDs: S-suffix=stroke(1), A-suffix=control(0).
    Subjects with no label entry are skipped.

    ECG channel selected via _select_ecg_channel(). TCD ECG is in uV and is scaled to mV.
    subject_id = "cves_{numeric_id}" (same for all conditions of a subject — ensures
    create_split() keeps all a subject's records in the same split, preventing leakage).

    Args:
        data_dir:    Path to cves/ directory (contains data/{sit-stand,head-up-tilt,...}).
        config_path: Path to config YAML (reads data.target_fs, data.waveform_sec).

    Returns:
        List of StrokeRecordDict, one per WFDB record file across all 4 protocols.
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
        logger.warning("cves_dir not found: %s", data_dir)
        return []

    records: List[StrokeRecordDict] = []
    seen: set = set()

    for protocol in CVES_PROTOCOLS:
        proto_dir = base / "data" / protocol
        if not proto_dir.is_dir():
            logger.debug("CVES protocol dir missing: %s", proto_dir)
            continue

        for hea_path in sorted(proto_dir.glob("*.hea")):
            stem = hea_path.stem  # e.g. "s0044-sit-stand"

            # Extract zero-stripped numeric subject ID for label lookup
            raw_num = stem.lstrip("s").split("-")[0]      # e.g. "0044"
            numeric_id = raw_num.lstrip("0") or "0"       # e.g. "44"

            if numeric_id in CVES_STROKE_IDS:
                label = 1
            elif numeric_id in CVES_CONTROL_IDS:
                label = 0
            else:
                logger.debug("CVES: no label for %s (numeric=%s) — skipping", stem, numeric_id)
                continue

            # Use subject-level ID so all conditions for same subject land in same split
            subject_id = f"cves_{raw_num}"

            key = (subject_id, stem)
            if key in seen:
                continue
            seen.add(key)

            try:
                record = wfdb.rdrecord(str(proto_dir / stem))
            except Exception as e:
                logger.warning("Failed to read CVES record %s: %s", stem, e)
                continue
            if record is None:
                continue

            signal = record.p_signal if record.p_signal is not None else record.d_signal
            if signal is None:
                continue

            signal_all = np.asarray(signal, dtype=np.float64)

            # Extract respiratory reference channel BEFORE collapsing to ECG
            resp_signal: Optional[np.ndarray] = None
            resp_channel: Optional[str] = None
            if signal_all.ndim == 2 and signal_all.shape[1] > 1 and record.sig_name:
                resp_idx, resp_channel = _select_resp_channel(record.sig_name)
                if resp_idx is not None:
                    resp_signal = signal_all[:, resp_idx].copy()
                    resp_signal = np.nan_to_num(resp_signal, nan=0.0)

            # Select best ECG channel; capture its unit for uV→mV scaling
            if signal_all.ndim == 2 and signal_all.shape[1] > 1:
                ch_idx = _select_ecg_channel(record.sig_name, record.units)
                unit = (record.units[ch_idx] or "").strip() if record.units else ""
                signal = signal_all[:, ch_idx]
            else:
                unit = (record.units[0] or "").strip() if record.units else ""
                signal = signal_all.ravel()

            # TCD ECG is recorded in uV — scale to mV for consistency
            if unit.lower() in ("uv", "\u03bcv"):
                signal = signal * 0.001

            signal = np.nan_to_num(signal, nan=0.0)

            fs = float(record.fs)

            if target_fs > 0 and abs(fs - target_fs) >= 0.1:
                signal = _resample_to_target_fs(signal, fs, target_fs)
                if resp_signal is not None:
                    resp_signal = _resample_to_target_fs(resp_signal, float(record.fs), target_fs)
                fs = target_fs

            min_samples = int(waveform_sec * fs)
            if len(signal) < min_samples:
                logger.debug("Skipping CVES %s — signal too short (%d samples)", stem, len(signal))
                continue

            records.append({
                "subject_id": subject_id,
                "signal": signal,
                "fs": fs,
                "label": label,
                "session_id": None,
                "epoch_type": None,
                "condition": None,
                "resp_signal": resp_signal,
                "resp_channel": resp_channel,
            })

    logger.info("Parsed %d records from cves (dir: %s)", len(records), data_dir)
    return records


# ---------------------------------------------------------------------------
# FANTASIA (PhysioNet fantasia)
# 40 healthy subjects (20 young f1y*, 20 elderly f1o*), watching Fantasia.
# Channels: [0] RESP (respiration belt), [1] ECG — 250 Hz, ~2 hours each.
# All label=0 (healthy controls — expands reference-labeled training pool).
# ---------------------------------------------------------------------------

def parse_fantasia_dir(
    data_dir: str,
    config_path: str,
) -> List[StrokeRecordDict]:
    """Parse FANTASIA PhysioNet records. Returns StrokeRecordDicts with
    ECG + respiration belt reference signal."""
    import wfdb

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    target_fs = int(data_cfg.get("target_fs", 250))
    waveform_sec = int(data_cfg.get("waveform_sec", 10))

    data_dir = Path(data_dir)
    hea_files = sorted(data_dir.glob("*.hea"))
    if not hea_files:
        logger.warning("No .hea files found in fantasia dir: %s", data_dir)
        return []

    records = []
    for hea in hea_files:
        stem = hea.stem
        record_path = str(hea.parent / stem)
        try:
            rec = wfdb.rdrecord(record_path)
        except Exception as e:
            logger.warning("FANTASIA: failed to read %s: %s", stem, e)
            continue

        sig_names = [n.strip().upper() for n in rec.sig_name]
        fs = float(rec.fs)

        # ECG is channel named 'ECG'; RESP is channel named 'RESP'
        ecg_idx = next((i for i, n in enumerate(sig_names) if "ECG" in n), None)
        resp_idx = next((i for i, n in enumerate(sig_names) if "RESP" in n), None)

        if ecg_idx is None:
            logger.warning("FANTASIA %s: no ECG channel in %s — skipping", stem, sig_names)
            continue

        signal_all = np.nan_to_num(rec.p_signal, nan=0.0)
        ecg = signal_all[:, ecg_idx].astype(np.float64)
        resp_signal = None
        resp_channel = None

        if resp_idx is not None:
            resp_signal = signal_all[:, resp_idx].astype(np.float64)
            resp_channel = "resp_belt"

        # Resample ECG if needed
        if fs != target_fs:
            ecg = _resample_to_target_fs(ecg, fs, target_fs)
            if resp_signal is not None:
                resp_signal = _resample_to_target_fs(resp_signal, fs, target_fs)

        min_len = waveform_sec * target_fs
        if len(ecg) < min_len:
            logger.debug("FANTASIA %s: signal too short (%d < %d) — skipping", stem, len(ecg), min_len)
            continue

        records.append({
            "subject_id": f"fantasia_{stem}",
            "signal": ecg,
            "fs": float(target_fs),
            "label": 0,  # all FANTASIA subjects are healthy controls
            "session_id": None,
            "epoch_type": None,
            "condition": None,
            "resp_signal": resp_signal,
            "resp_channel": resp_channel,
        })

    logger.info("Parsed %d records from fantasia (dir: %s)", len(records), data_dir)
    return records
