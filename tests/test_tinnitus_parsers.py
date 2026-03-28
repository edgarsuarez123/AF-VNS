"""Tests for tinnitus pipeline schema and config."""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from src.data.tinnitus_parsers import (
    TinnitusRecordDict,
    _REQUIRED_KEYS,
    _WESAD_BVP_FS,
    _WESAD_CHEST_FS,
    _WESAD_EDA_FS,
    validate_tinnitus_record,
    parse_bidmc_ppg_dir,
    parse_wesad_dir,
)


def _make_valid_record(**overrides) -> TinnitusRecordDict:
    record: TinnitusRecordDict = {
        "subject_id": "test_001",
        "ppg_signal": np.zeros(1250, dtype=np.float64),
        "ppg_fs": 125.0,
        "resp_signal": None,
        "resp_fs": None,
        "resp_channel": None,
        "eda_signal": None,
        "eda_fs": None,
        "label": 0,
        "session_id": None,
        "condition": None,
    }
    record.update(overrides)
    return record


class TestTinnitusRecordDict:
    def test_valid_minimal_record(self):
        record = _make_valid_record()
        assert validate_tinnitus_record(record)

    def test_valid_full_record(self):
        record = _make_valid_record(
            resp_signal=np.zeros(1250, dtype=np.float64),
            resp_fs=125.0,
            resp_channel="impedance",
            eda_signal=np.zeros(40, dtype=np.float64),
            eda_fs=4.0,
            label=1,
            session_id="ses01",
            condition="trifold",
        )
        assert validate_tinnitus_record(record)

    def test_missing_key_fails(self):
        record = _make_valid_record()
        del record["ppg_signal"]
        assert not validate_tinnitus_record(record)

    def test_all_required_keys_present(self):
        record = _make_valid_record()
        assert _REQUIRED_KEYS.issubset(set(record.keys()))

    def test_ppg_signal_must_be_ndarray(self):
        record = _make_valid_record(ppg_signal=[0.0, 1.0, 2.0])
        assert not validate_tinnitus_record(record)

    def test_ppg_signal_must_be_1d(self):
        record = _make_valid_record(ppg_signal=np.zeros((10, 2), dtype=np.float64))
        assert not validate_tinnitus_record(record)

    def test_ppg_fs_must_be_positive(self):
        record = _make_valid_record(ppg_fs=0.0)
        assert not validate_tinnitus_record(record)


class TestTinnitusConfig:
    CONFIG_PATH = "config_tinnitus.yaml"

    def test_config_loads(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None

    def test_required_sections_present(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        for section in ["data", "ppg_filter", "eda", "phase_model", "phase_training", "closed_loop"]:
            assert section in cfg, f"Missing section: {section}"

    def test_ppg_target_fs(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg["data"]["ppg_target_fs"] == 125

    def test_eda_target_fs(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg["data"]["eda_target_fs"] == 4

    def test_phase_model_input_samples(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        # 2s × 125Hz = 250 samples (not 500 like ECG @ 250Hz)
        assert cfg["phase_model"]["input_samples"] == 250

    def test_phase_model_n_tasks(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg["phase_model"]["n_tasks"] == 2  # diastole + exhalation

    def test_wesad_label_map_keys(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        label_map = cfg["wesad"]["label_map"]
        # Labels 0-4 must all be present
        for k in [0, 1, 2, 3, 4]:
            assert k in label_map, f"Missing WESAD label {k}"


class TestBidmcPpgParser:
    BIDMC_DIR = "data/raw/stroke avns/bidmc"
    CONFIG_PATH = "config_tinnitus.yaml"

    def test_returns_53_records(self):
        records = parse_bidmc_ppg_dir(self.BIDMC_DIR, self.CONFIG_PATH)
        assert len(records) == 53, f"Expected 53, got {len(records)}"

    def test_ppg_signal_is_1d_float64(self):
        records = parse_bidmc_ppg_dir(self.BIDMC_DIR, self.CONFIG_PATH)
        for r in records:
            assert isinstance(r["ppg_signal"], np.ndarray)
            assert r["ppg_signal"].ndim == 1
            assert r["ppg_signal"].dtype == np.float64

    def test_ppg_fs_is_125(self):
        records = parse_bidmc_ppg_dir(self.BIDMC_DIR, self.CONFIG_PATH)
        for r in records:
            assert r["ppg_fs"] == 125.0

    def test_resp_signal_not_none(self):
        records = parse_bidmc_ppg_dir(self.BIDMC_DIR, self.CONFIG_PATH)
        for r in records:
            assert r["resp_signal"] is not None
            assert r["resp_channel"] == "impedance"

    def test_eda_is_none(self):
        records = parse_bidmc_ppg_dir(self.BIDMC_DIR, self.CONFIG_PATH)
        for r in records:
            assert r["eda_signal"] is None
            assert r["eda_fs"] is None

    def test_label_is_zero(self):
        records = parse_bidmc_ppg_dir(self.BIDMC_DIR, self.CONFIG_PATH)
        for r in records:
            assert r["label"] == 0

    def test_all_records_pass_schema_validation(self):
        records = parse_bidmc_ppg_dir(self.BIDMC_DIR, self.CONFIG_PATH)
        for r in records:
            assert validate_tinnitus_record(r), f"Record {r['subject_id']} failed validation"

    def test_ppg_signal_nonzero(self):
        records = parse_bidmc_ppg_dir(self.BIDMC_DIR, self.CONFIG_PATH)
        for r in records:
            assert len(r["ppg_signal"]) > 0


def _make_wesad_pickle(n_chest: int = 700 * 120, n_bvp: int = 64 * 120) -> dict:
    """Build a minimal WESAD-format data dict for one subject (120s duration)."""
    # Labels at chest FS: 30s undefined, 60s baseline, 30s stress
    labels = np.zeros(n_chest, dtype=np.int32)
    labels[int(700 * 30):int(700 * 90)] = 1   # baseline
    labels[int(700 * 90):] = 2                 # stress

    return {
        "signal": {
            "wrist": {
                "BVP": np.random.randn(n_bvp).reshape(-1, 1),
                "EDA": np.abs(np.random.randn(int(n_bvp * _WESAD_EDA_FS / _WESAD_BVP_FS))).reshape(-1, 1),
                "TEMP": np.random.randn(int(n_bvp * 4 / _WESAD_BVP_FS)).reshape(-1, 1),
                "ACC": np.random.randn(int(n_bvp * 32 / _WESAD_BVP_FS), 3),
            },
            "chest": {
                "Resp": np.random.randn(n_chest).reshape(-1, 1),
                "EDA": np.random.randn(n_chest).reshape(-1, 1),
                "ECG": np.random.randn(n_chest).reshape(-1, 1),
            },
        },
        "label": labels,
        "subject": 2,
    }


class TestWesadParser:
    CONFIG_PATH = "config_tinnitus.yaml"

    def _write_wesad_subject(self, tmp_dir: Path, subject_id: int) -> None:
        subj_dir = tmp_dir / f"S{subject_id}"
        subj_dir.mkdir()
        data = _make_wesad_pickle()
        with open(subj_dir / f"S{subject_id}.pkl", "wb") as f:
            pickle.dump(data, f)

    def test_parses_mock_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wesad_subject(tmp_path, 2)
            # Point config subjects list to only S2
            with patch("src.data.tinnitus_parsers._load_tinnitus_config") as mock_cfg:
                mock_cfg.return_value = {
                    "wesad": {
                        "label_map": {0: None, 1: 0, 2: 1, 3: 0, 4: 0},
                        "subjects": [2],
                    }
                }
                records = parse_wesad_dir(str(tmp_path), self.CONFIG_PATH)

        assert len(records) >= 1

    def test_all_records_pass_schema_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wesad_subject(tmp_path, 2)
            with patch("src.data.tinnitus_parsers._load_tinnitus_config") as mock_cfg:
                mock_cfg.return_value = {
                    "wesad": {
                        "label_map": {0: None, 1: 0, 2: 1, 3: 0, 4: 0},
                        "subjects": [2],
                    }
                }
                records = parse_wesad_dir(str(tmp_path), self.CONFIG_PATH)

        for r in records:
            assert validate_tinnitus_record(r), f"{r['subject_id']} failed validation"

    def test_ppg_fs_is_64(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wesad_subject(tmp_path, 2)
            with patch("src.data.tinnitus_parsers._load_tinnitus_config") as mock_cfg:
                mock_cfg.return_value = {
                    "wesad": {
                        "label_map": {0: None, 1: 0, 2: 1, 3: 0, 4: 0},
                        "subjects": [2],
                    }
                }
                records = parse_wesad_dir(str(tmp_path), self.CONFIG_PATH)

        for r in records:
            assert r["ppg_fs"] == _WESAD_BVP_FS

    def test_eda_and_resp_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wesad_subject(tmp_path, 2)
            with patch("src.data.tinnitus_parsers._load_tinnitus_config") as mock_cfg:
                mock_cfg.return_value = {
                    "wesad": {
                        "label_map": {0: None, 1: 0, 2: 1, 3: 0, 4: 0},
                        "subjects": [2],
                    }
                }
                records = parse_wesad_dir(str(tmp_path), self.CONFIG_PATH)

        for r in records:
            assert r["eda_signal"] is not None
            assert r["resp_signal"] is not None

    def test_undefined_epochs_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wesad_subject(tmp_path, 2)
            with patch("src.data.tinnitus_parsers._load_tinnitus_config") as mock_cfg:
                mock_cfg.return_value = {
                    "wesad": {
                        "label_map": {0: None, 1: 0, 2: 1, 3: 0, 4: 0},
                        "subjects": [2],
                    }
                }
                records = parse_wesad_dir(str(tmp_path), self.CONFIG_PATH)

        # No record should have condition="undefined"
        for r in records:
            assert r["condition"] != "undefined"

    def test_labels_are_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wesad_subject(tmp_path, 2)
            with patch("src.data.tinnitus_parsers._load_tinnitus_config") as mock_cfg:
                mock_cfg.return_value = {
                    "wesad": {
                        "label_map": {0: None, 1: 0, 2: 1, 3: 0, 4: 0},
                        "subjects": [2],
                    }
                }
                records = parse_wesad_dir(str(tmp_path), self.CONFIG_PATH)

        for r in records:
            assert r["label"] in (0, 1)

    def test_missing_subject_skipped_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Write S2 but request S2 and S3 (S3 missing)
            self._write_wesad_subject(tmp_path, 2)
            with patch("src.data.tinnitus_parsers._load_tinnitus_config") as mock_cfg:
                mock_cfg.return_value = {
                    "wesad": {
                        "label_map": {0: None, 1: 0, 2: 1, 3: 0, 4: 0},
                        "subjects": [2, 3],
                    }
                }
                records = parse_wesad_dir(str(tmp_path), self.CONFIG_PATH)

        # Should still return records for S2
        assert len(records) >= 1

    @pytest.mark.skipif(
        not Path("data/raw/tinnitus avns/wesad/S2/S2.pkl").exists(),
        reason="WESAD not downloaded",
    )
    def test_real_wesad_all_subjects(self):
        records = parse_wesad_dir("data/raw/tinnitus avns/wesad", self.CONFIG_PATH)
        subject_ids = {r["session_id"] for r in records}
        # Expect at least 10 subjects (some may have short epochs filtered)
        assert len(subject_ids) >= 10
        for r in records:
            assert validate_tinnitus_record(r)
