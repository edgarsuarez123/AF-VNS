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
