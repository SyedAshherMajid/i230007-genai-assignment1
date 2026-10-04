"""Regenerate compact learning-curve figures from recorded experiment histories."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report/figures"


def records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    runs = (
        ("Task 1: universal", ROOT / "artifacts/task1/history.jsonl", "score", "train_loss"),
        ("Task 2: classifier", ROOT / "artifacts/task2/classifier/history.jsonl", "macro_f1", "train_loss"),
        ("Task 2: salt expert", ROOT / "artifacts/task2/expert_1/history.jsonl", "score", "train_loss"),
        ("Task 4: sketch GAN", ROOT / "artifacts/task4/history.jsonl", "score", "g_l1"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), constrained_layout=True)
    for axis, (title, path, validation_key, training_key) in zip(axes.flat, runs):
        history = records(path)
        epochs = [row["epoch"] + 1 for row in history]
        validation = [row[validation_key] for row in history]
        training = [row[training_key] for row in history]
        axis.plot(epochs, validation, color="#2854a5", linewidth=1.8, label="Validation")
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Macro-F1 ↑" if validation_key == "macro_f1" else "Selection score ↓",
                        color="#2854a5")
        axis.tick_params(axis="y", labelcolor="#2854a5")
        axis.grid(alpha=0.22)
        other = axis.twinx()
        other.plot(epochs, training, color="#c86a22", linewidth=1.35, alpha=0.85,
                   label="Training loss" if training_key == "train_loss" else "Generator L1")
        other.set_ylabel("Training loss" if training_key == "train_loss" else "Generator L1",
                         color="#c86a22")
        other.tick_params(axis="y", labelcolor="#c86a22")
    FIGURES.mkdir(parents=True, exist_ok=True)
    destination = FIGURES / "training_curves.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
