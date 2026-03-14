"""
Plot training curves from training_log.csv. Saves to models/artifacts/training_curves.png.
"""

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.training.build_model import load_config

CONFIG_PATH = "config.yaml"
DEFAULT_LOG = "models/artifacts/training_log.csv"
DEFAULT_OUT = "models/artifacts/training_curves.png"


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from CSV.")
    parser.add_argument("--csv", default=None, help="Training log CSV path (default: from config dir)")
    parser.add_argument("--out", default=None, help="Output figure path")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config YAML for paths")
    args = parser.parse_args()

    config = load_config(args.config)
    scaler_path = config.get("paths", {}).get("scaler", "models/artifacts/scaler.pkl")
    artifacts_dir = str(Path(scaler_path).parent)
    csv_path = args.csv or str(Path(artifacts_dir) / "training_log.csv")
    out_path = args.out or str(Path(artifacts_dir) / "training_curves.png")

    if not Path(csv_path).is_file():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    import csv as csv_module
    epochs = []
    train_loss = []
    val_loss = []
    val_auroc = []
    with open(csv_path) as f:
        r = csv_module.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_auroc.append(float(row["val_auroc"]))

    if not epochs:
        print("No rows in CSV")
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax1.plot(epochs, train_loss, label="Train loss")
    ax1.plot(epochs, val_loss, label="Val loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_auroc, color="green", label="Val AUROC")
    ax2.set_ylabel("Val AUROC")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training curves")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
