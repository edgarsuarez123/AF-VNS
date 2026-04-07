"""
Tinnitus AVNS — dataset parsers.

TinnitusRecordDict schema:
    subject_id    str
    ppg_signal    np.ndarray          1D float64 (PPG / BVP waveform)
    ppg_fs        float               PPG sampling rate (Hz)
    resp_signal   np.ndarray | None   1D float64 respiratory reference
    resp_fs       float | None        respiratory sampling rate (Hz)
    resp_channel  str | None          "impedance" / "chest_belt" / "ppg_derived"
    eda_signal    np.ndarray | None   1D float64 EDA waveform (µS)
    eda_fs        float | None        EDA sampling rate (Hz)
    label         int                 0=baseline/control, 1=stress/event
    session_id    str | None
    condition     str | None          "trifold" / "cardiac_only" / "open_loop"

Nothing in the AF or stroke pipelines is imported or modified here.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TinnitusRecordDict = Dict[str, Any]

# Required keys for schema validation
_REQUIRED_KEYS = {
    "subject_id",
    "ppg_signal",
    "ppg_fs",
    "resp_signal",
    "resp_fs",
    "resp_channel",
    "eda_signal",
    "eda_fs",
    "label",
    "session_id",
    "condition",
}


def validate_tinnitus_record(record: TinnitusRecordDict) -> bool:
    """Return True if record contains all required keys with correct types."""
    missing = _REQUIRED_KEYS - set(record.keys())
    if missing:
        logger.warning("TinnitusRecordDict missing keys: %s", missing)
        return False
    if not isinstance(record["ppg_signal"], np.ndarray):
        logger.warning("ppg_signal must be np.ndarray")
        return False
    if record["ppg_signal"].ndim != 1:
        logger.warning("ppg_signal must be 1D")
        return False
    if record["ppg_fs"] <= 0:
        logger.warning("ppg_fs must be > 0")
        return False
    return True


def _load_tinnitus_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# BIDMC PPG parser
# ---------------------------------------------------------------------------

def parse_bidmc_ppg_dir(
    data_dir: str,
    config_path: str,
) -> List[TinnitusRecordDict]:
    """Parse BIDMC PhysioNet records, extracting PPG (PLETH) + impedance respiratory.

    BIDMC native rate is 125 Hz — kept as-is (matches SBIR ≥125 Hz requirement).
    Channels present: RESP, PLETH, V, AVR, II (indices 0-4).
    PLETH = photoplethysmography (PPG); RESP = impedance pneumography.
    EDA is not available in BIDMC — eda_signal set to None.
    All records are healthy ICU controls → label=0.
    """
    import wfdb

    config = _load_tinnitus_config(config_path)
    data_cfg = config.get("data", {})
    ppg_target_fs: float = float(data_cfg.get("ppg_target_fs", 125.0))
    max_signal_sec: Optional[float] = data_cfg.get("max_signal_sec", None)

    data_dir_path = Path(data_dir)
    hea_files = sorted(data_dir_path.glob("*.hea"))
    if not hea_files:
        logger.warning("No .hea files found in BIDMC dir: %s", data_dir_path)
        return []

    records: List[TinnitusRecordDict] = []
    for hea in hea_files:
        stem = hea.stem
        record_path = str(hea.parent / stem)
        try:
            hdr = wfdb.rdheader(record_path)
            sampto = int(max_signal_sec * hdr.fs) if max_signal_sec else None
            rec = wfdb.rdrecord(record_path, sampto=sampto)
        except Exception as exc:
            logger.warning("BIDMC: failed to read %s: %s", stem, exc)
            continue

        sig_names_upper = [n.strip().upper() for n in rec.sig_name]
        native_fs = float(rec.fs)

        # PPG: PLETH channel
        ppg_idx = next((i for i, n in enumerate(sig_names_upper) if "PLETH" in n), None)
        if ppg_idx is None:
            logger.warning("BIDMC %s: no PLETH channel in %s — skipping", stem, sig_names_upper)
            continue

        # RESP: impedance pneumography
        resp_idx = next((i for i, n in enumerate(sig_names_upper) if "RESP" in n), None)

        signal_all = np.nan_to_num(rec.p_signal, nan=0.0)
        ppg = signal_all[:, ppg_idx].astype(np.float64)

        resp_signal: Optional[np.ndarray] = None
        resp_channel: Optional[str] = None
        if resp_idx is not None:
            resp_signal = signal_all[:, resp_idx].astype(np.float64)
            resp_channel = "impedance"  # no inversion needed — same polarity convention

        # Resample if native rate differs from target (BIDMC native = 125 Hz = ppg_target_fs by default)
        if native_fs != ppg_target_fs:
            from scipy.signal import resample as _scipy_resample
            n_out = int(len(ppg) * ppg_target_fs / native_fs)
            ppg = _scipy_resample(ppg, n_out)
            if resp_signal is not None:
                resp_signal = _scipy_resample(resp_signal, n_out)

        records.append({
            "subject_id": f"bidmc_{stem}",
            "ppg_signal": ppg,
            "ppg_fs": ppg_target_fs,
            "resp_signal": resp_signal,
            "resp_fs": ppg_target_fs,
            "resp_channel": resp_channel,
            "eda_signal": None,
            "eda_fs": None,
            "label": 0,
            "session_id": None,
            "condition": None,
        })

    logger.info("Parsed %d BIDMC PPG records from %s", len(records), data_dir_path)
    return records


# ---------------------------------------------------------------------------
# WESAD parser
# ---------------------------------------------------------------------------

# WESAD Empatica E4 sampling rates (wrist device)
_WESAD_BVP_FS = 64.0    # BVP / PPG (Hz)
_WESAD_EDA_FS = 4.0     # EDA / GSR (Hz)
_WESAD_TEMP_FS = 4.0    # Temperature (Hz)
_WESAD_ACC_FS = 32.0    # Accelerometer (Hz)

# WESAD RespiBAN chest device sampling rate (all channels)
_WESAD_CHEST_FS = 700.0

# Label definitions from WESAD paper:
#   0 = not defined / transient, 1 = baseline, 2 = stress,
#   3 = amusement, 4 = meditation
_WESAD_LABEL_NAMES = {0: "undefined", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def parse_wesad_dir(
    data_dir: str,
    config_path: str,
) -> List[TinnitusRecordDict]:
    """Parse WESAD dataset pickle files into TinnitusRecordDicts.

    Extracts:
        - wrist BVP (PPG) at 64 Hz
        - wrist EDA at 4 Hz
        - chest Resp at 700 Hz (downsampled to match PPG timebase)

    Each continuous label epoch becomes a separate record. Label=0 undefined
    epochs are skipped. Label mapping is configurable via config_tinnitus.yaml
    wesad.label_map (baseline→0, stress→1, amusement/meditation→0 by default).

    Subjects: S2–S17 excluding S12 (corrupted per original paper).
    """
    import pickle

    config = _load_tinnitus_config(config_path)
    wesad_cfg = config.get("wesad", {})
    label_map: dict = wesad_cfg.get("label_map", {1: 0, 2: 1, 3: 0, 4: 0})
    # YAML keys are ints but may load as strings depending on YAML parser
    label_map = {int(k): v for k, v in label_map.items()}
    subjects: list = wesad_cfg.get(
        "subjects", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    )

    data_dir_path = Path(data_dir)
    records: List[TinnitusRecordDict] = []

    for s in subjects:
        pkl_path = data_dir_path / f"S{s}" / f"S{s}.pkl"
        if not pkl_path.exists():
            logger.warning("WESAD: subject file not found: %s", pkl_path)
            continue

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
        except Exception as exc:
            logger.warning("WESAD: failed to load %s: %s", pkl_path, exc)
            continue

        # Extract signals
        try:
            bvp = np.array(data["signal"]["wrist"]["BVP"], dtype=np.float64).ravel()
            eda_wrist = np.array(data["signal"]["wrist"]["EDA"], dtype=np.float64).ravel()
            resp_chest = np.array(data["signal"]["chest"]["Resp"], dtype=np.float64).ravel()
            labels_raw = np.array(data["label"], dtype=np.int32).ravel()
        except (KeyError, TypeError) as exc:
            logger.warning("WESAD S%d: unexpected data structure: %s", s, exc)
            continue

        # Extract wrist temperature (F19 — optional, same 4 Hz as EDA)
        try:
            temp_wrist = np.array(data["signal"]["wrist"]["TEMP"], dtype=np.float64).ravel()
        except (KeyError, TypeError):
            temp_wrist = None

        # Labels are at chest FS (700 Hz) — downsample to BVP FS (64 Hz) for epoch segmentation
        # DECISION: segment using BVP-rate labels to avoid sub-sample epoch boundaries
        bvp_label_indices = np.round(
            np.arange(len(bvp)) * _WESAD_CHEST_FS / _WESAD_BVP_FS
        ).astype(np.int64)
        bvp_label_indices = np.clip(bvp_label_indices, 0, len(labels_raw) - 1)
        bvp_labels = labels_raw[bvp_label_indices]

        # Downsample chest respiration to BVP FS to align timebases
        from scipy.signal import resample as _scipy_resample
        n_bvp = len(bvp)
        resp_resampled = _scipy_resample(resp_chest, n_bvp)

        # EDA is at 4 Hz — build index array mapping EDA samples to BVP samples
        eda_label_indices = np.round(
            np.arange(len(eda_wrist)) * _WESAD_CHEST_FS / _WESAD_EDA_FS
        ).astype(np.int64)
        eda_label_indices = np.clip(eda_label_indices, 0, len(labels_raw) - 1)
        eda_labels = labels_raw[eda_label_indices]

        # Segment into continuous label epochs
        epoch_records = _segment_wesad_epochs(
            subject_id=s,
            bvp=bvp,
            bvp_labels=bvp_labels,
            eda=eda_wrist,
            eda_labels=eda_labels,
            resp=resp_resampled,
            label_map=label_map,
            temp=temp_wrist,
        )
        records.extend(epoch_records)

    logger.info("Parsed %d WESAD epoch records from %d subjects", len(records), len(subjects))
    return records


def _segment_wesad_epochs(
    subject_id: int,
    bvp: np.ndarray,
    bvp_labels: np.ndarray,
    eda: np.ndarray,
    eda_labels: np.ndarray,
    resp: np.ndarray,
    label_map: dict,
    temp: Optional[np.ndarray] = None,
) -> List[TinnitusRecordDict]:
    """Segment continuous WESAD signals into per-label-epoch records."""
    records: List[TinnitusRecordDict] = []

    # Find contiguous label runs in BVP-rate labels
    change_points = np.where(np.diff(bvp_labels) != 0)[0] + 1
    boundaries = np.concatenate([[0], change_points, [len(bvp_labels)]])

    for i in range(len(boundaries) - 1):
        start_bvp = boundaries[i]
        end_bvp = boundaries[i + 1]
        raw_label = int(bvp_labels[start_bvp])

        # Skip undefined epochs (label=0 in raw WESAD convention)
        if raw_label not in label_map or label_map[raw_label] is None:
            continue

        mapped_label = label_map[raw_label]
        label_name = _WESAD_LABEL_NAMES.get(raw_label, "unknown")

        # BVP epoch
        bvp_epoch = bvp[start_bvp:end_bvp].astype(np.float64)

        # Resp epoch — already resampled to BVP FS
        resp_epoch = resp[start_bvp:end_bvp].astype(np.float64)

        # EDA epoch — find corresponding EDA samples by label
        eda_mask = eda_labels == raw_label
        # Use contiguous run matching this BVP epoch by proportion
        eda_start = int(start_bvp * _WESAD_EDA_FS / _WESAD_BVP_FS)
        eda_end = int(end_bvp * _WESAD_EDA_FS / _WESAD_BVP_FS)
        eda_end = min(eda_end, len(eda))
        eda_epoch = eda[eda_start:eda_end].astype(np.float64)

        # Skip epochs that are too short to be useful (< 30s at BVP FS)
        min_bvp_samples = int(30 * _WESAD_BVP_FS)
        if len(bvp_epoch) < min_bvp_samples:
            continue

        # Temperature epoch (F19) — same index range as EDA (both at 4 Hz)
        temp_epoch = None
        if temp is not None and len(temp) > 0:
            temp_start = int(start_bvp * _WESAD_TEMP_FS / _WESAD_BVP_FS)
            temp_end = int(end_bvp * _WESAD_TEMP_FS / _WESAD_BVP_FS)
            temp_end = min(temp_end, len(temp))
            temp_epoch = temp[temp_start:temp_end].astype(np.float64) if temp_end > temp_start else None

        records.append({
            "subject_id": f"wesad_S{subject_id}_{label_name}_{i}",
            "ppg_signal": bvp_epoch,
            "ppg_fs": _WESAD_BVP_FS,
            "resp_signal": resp_epoch,
            "resp_fs": _WESAD_BVP_FS,
            "resp_channel": "chest_belt",
            "eda_signal": eda_epoch if len(eda_epoch) > 0 else None,
            "eda_fs": _WESAD_EDA_FS if len(eda_epoch) > 0 else None,
            "temp_signal": temp_epoch,
            "temp_fs": _WESAD_TEMP_FS if temp_epoch is not None else None,
            "label": mapped_label,
            "session_id": f"S{subject_id}",
            "condition": label_name,
        })

    return records
