"""
F13 — Train supervised arousal state classifier on WESAD EDA features.

Loads precomputed feature cache (from tinnitus_precompute_arousal.py),
fits GradientBoostingClassifier, evaluates, and saves checkpoint.

Usage:
    python -m src.training.tinnitus_train_arousal --config config_tinnitus.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    root = Path(__file__).resolve().parents[2]
    if not Path(config_path).is_absolute() and not Path(config_path).exists():
        config_path = str(root / config_path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve(path: str, root: Path) -> str:
    return path if Path(path).is_absolute() else str(root / path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train tinnitus arousal state classifier on WESAD EDA features"
    )
    parser.add_argument("--config", default="config_tinnitus.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    cfg = _load_config(args.config)
    ac_cfg = cfg.get("arousal_classifier", {})

    cache_dir = _resolve(
        ac_cfg.get("cache_dir", "models/artifacts/cache_tinnitus_arousal"), root
    )
    checkpoint_path = _resolve(
        ac_cfg.get("checkpoint_path", "models/checkpoints/arousal_classifier.pkl"), root
    )
    model_type = ac_cfg.get("model_type", "gradient_boosting")

    # ------------------------------------------------------------------
    # Load cached features
    # ------------------------------------------------------------------
    logger.info("Loading features from %s", cache_dir)

    def _load_split(split: str):
        feats = np.load(str(Path(cache_dir) / f"{split}_features.npy"))
        labels = np.load(str(Path(cache_dir) / f"{split}_labels.npy"))
        return feats, labels

    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")

    logger.info(
        "Samples — train: %d | val: %d | test: %d",
        len(y_train), len(y_val), len(y_test),
    )
    logger.info(
        "Train label dist — in-band (0): %d | out-of-band (1): %d",
        int((y_train == 0).sum()), int((y_train == 1).sum()),
    )

    if len(y_train) == 0:
        logger.error("Empty training set. Run tinnitus_precompute_arousal first.")
        return

    # ------------------------------------------------------------------
    # Fit classifier
    # ------------------------------------------------------------------
    from src.models.arousal_classifier import ArousalClassifier

    clf = ArousalClassifier(model_type=model_type, config_path=args.config)
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def _fmt(metrics: dict) -> str:
        return (
            f"accuracy={metrics['accuracy']:.3f}  "
            f"auroc={metrics.get('auroc', float('nan')):.3f}  "
            f"f1={metrics['f1']:.3f}"
        )

    val_metrics = clf.score(X_val, y_val)
    test_metrics = clf.score(X_test, y_test)

    logger.info("Val  — %s", _fmt(val_metrics))
    logger.info("Test — %s", _fmt(test_metrics))

    # Feature importance (GBT only)
    if model_type == "gradient_boosting" and clf._model is not None:
        from src.training.tinnitus_precompute_arousal import FEATURE_NAMES
        importances = clf._model.feature_importances_
        logger.info("Feature importances:")
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
            logger.info("  %-25s %.4f", name, imp)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    clf.save(checkpoint_path)
    logger.info("Checkpoint saved to %s", checkpoint_path)

    results = {
        "val": val_metrics,
        "test": test_metrics,
        "model_type": model_type,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }
    metrics_path = str(Path(checkpoint_path).parent / "arousal_classifier_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
