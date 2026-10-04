"""Evaluate pet restoration models on the saved deterministic manifests."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from genai_a1.models import ConvAutoencoder, CorruptionClassifier, SoftMixture
from genai_a1.pets import PetCasesDataset
from genai_a1.training import reconstruction_metrics, sha256

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("clean", "noise", "blur", "occlusion")
SEVERITIES = ("none", "low", "medium", "high")
METRIC_NAMES = ("mae", "ssim", "psnr")


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def configured_autoencoder(checkpoint: dict) -> ConvAutoencoder:
    config = checkpoint.get("config", {})
    return ConvAutoencoder(int(config.get("base", 16)), int(config.get("bottleneck", 48)),
                           float(config.get("dropout", 0.1)))


def configured_classifier(checkpoint: dict) -> CorruptionClassifier:
    config = checkpoint.get("config", {})
    return CorruptionClassifier(int(config.get("base", 16)), float(config.get("dropout", 0.1)))


def parent_directory(config: dict) -> Path:
    parents = Path(config.get("parents", ROOT / "artifacts/task2"))
    return parents if parents.is_absolute() else ROOT / parents


def load_task_models(tasks: list[str], device: torch.device) -> tuple[dict, dict]:
    models: dict[str, torch.nn.Module] = {}
    checkpoints: dict[str, Path] = {}
    if "task1" in tasks:
        path = ROOT / "artifacts/task1/best.pt"
        checkpoint = load_checkpoint(path)
        model = configured_autoencoder(checkpoint)
        model.load_state_dict(checkpoint["model"])
        models["task1"] = model
        checkpoints["task1"] = path
    if "task2" in tasks:
        classifier_path = ROOT / "artifacts/task2/classifier/best.pt"
        classifier_state = load_checkpoint(classifier_path)
        classifier = configured_classifier(classifier_state)
        classifier.load_state_dict(classifier_state["model"])
        models["classifier"] = classifier
        checkpoints["classifier"] = classifier_path
        experts = []
        for label in (1, 2, 3):
            path = ROOT / f"artifacts/task2/expert_{label}/best.pt"
            state = load_checkpoint(path)
            expert = configured_autoencoder(state)
            expert.load_state_dict(state["model"])
            experts.append(expert)
            models[f"expert_{label}"] = expert
            checkpoints[f"expert_{label}"] = path
        models["hard_experts"] = torch.nn.ModuleList(experts)
    if "task3" in tasks:
        path = ROOT / "artifacts/task3/best.pt"
        checkpoint = load_checkpoint(path)
        config = checkpoint.get("config", {})
        parents = parent_directory(config)
        classifier_state = load_checkpoint(parents / "classifier/best.pt")
        gate = configured_classifier(classifier_state)
        experts = []
        for label in (1, 2, 3):
            expert_state = load_checkpoint(parents / f"expert_{label}/best.pt")
            experts.append(configured_autoencoder(expert_state))
        model = SoftMixture(gate, experts, float(config.get("temperature", 1.0)))
        model.load_state_dict(checkpoint["model"])
        models["task3"] = model
        checkpoints["task3"] = path
    for model in models.values():
        model.to(device).eval()
    return models, checkpoints


def normalized_severity(case: dict) -> str:
    return "none" if case["label"] == "clean" else case.get("severity") or "none"


def image_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def grouped_means(values: dict) -> dict:
    result = {}
    for label in LABELS:
        for severity in SEVERITIES:
            metrics = values.get((label, severity))
            if not metrics:
                continue
            result[f"{label}/{severity}"] = {
                "count": len(metrics["mae"]),
                **{name: (None if np.isposinf(items).any() else float(np.mean(items)))
                   for name, items in metrics.items()},
            }
    return result


def render_grid(examples: list[dict], path: Path, title: str) -> None:
    if not examples:
        return
    figure, axes = plt.subplots(len(examples), 4, figsize=(12, max(3.0, 2.6 * len(examples))),
                                squeeze=False)
    for row_index, example in enumerate(examples):
        axes[row_index, 0].imshow(example["target"])
        axes[row_index, 1].imshow(example["input"])
        axes[row_index, 2].imshow(example["output"])
        error = np.abs(example["target"] - example["output"]).mean(axis=2)
        axes[row_index, 3].imshow(error, cmap="magma", vmin=0, vmax=1)
        for column, heading in enumerate(("Clean target", "Corrupted input", "Reconstruction", "Absolute error")):
            axes[row_index, column].set_title(heading if row_index == 0 else "", fontsize=9)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(
            f"{example['id']}\n{example['label']} / {example['severity']}\nMAE {example['mae']:.3f}",
            fontsize=8, rotation=0, labelpad=52, va="center")
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_totals, 1)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set(xticks=np.arange(4), yticks=np.arange(4), xticklabels=LABELS,
             yticklabels=LABELS, xlabel="Predicted condition", ylabel="True condition",
             title="Pet corruption classifier (row-normalized)")
    for row in range(4):
        for column in range(4):
            axis.text(column, row, f"{normalized[row, column]:.2f}\n({matrix[row, column]})",
                      ha="center", va="center", color="black", fontsize=8)
    figure.colorbar(image, ax=axis, label="Row fraction")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_routing_heatmap(weight_values: dict, path: Path) -> None:
    rows = [(label, severity) for label in LABELS for severity in SEVERITIES
            if (label, severity) in weight_values]
    if not rows:
        return
    matrix = np.stack([np.mean(weight_values[row], axis=0) for row in rows])
    figure, axis = plt.subplots(figsize=(7, max(3, 0.42 * len(rows))))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axis.set(xticks=np.arange(4), xticklabels=LABELS,
             yticks=np.arange(len(rows)), yticklabels=[f"{label} / {severity}" for label, severity in rows],
             title="Mean soft routing weights by condition and severity")
    for row in range(len(rows)):
        for column in range(4):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center",
                      color="white" if matrix[row, column] < 0.55 else "black", fontsize=8)
    figure.colorbar(image, ax=axis, label="Mean branch weight")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def select_examples(dataset: PetCasesDataset) -> tuple[set[tuple[str, str, str]], list[tuple[str, str]]]:
    rng = np.random.default_rng(42)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in dataset.cases:
        groups[(case["label"], normalized_severity(case))].append(case)
    requested = [("clean", "none")]
    requested.extend((label, severity) for label in LABELS[1:] for severity in SEVERITIES[1:])
    selected: list[dict] = []
    used_cases: set[tuple[str, str, str]] = set()
    for group in requested:
        candidates = groups[group]
        case = candidates[int(rng.integers(0, len(candidates)))]
        selected.append(case)
        used_cases.add((case["id"], case["label"], normalized_severity(case)))
    for group in (("noise", "high"), ("occlusion", "high")):
        candidates = [case for case in groups[group]
                      if (case["id"], case["label"], normalized_severity(case)) not in used_cases]
        case = candidates[int(rng.integers(0, len(candidates)))]
        selected.append(case)
        used_cases.add((case["id"], case["label"], normalized_severity(case)))
    keys = {(case["id"], case["label"], normalized_severity(case)) for case in selected}
    labels = [(case["label"], normalized_severity(case)) for case in selected]
    return keys, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=("task1", "task2", "task3"), default=["task1"])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit-ids", type=int, help="Limit to the first N split IDs for smoke checks")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluation/pets")
    args = parser.parse_args()

    started = time.perf_counter()
    device = torch.device(args.device)
    dataset = PetCasesDataset(ROOT / "data/processed/pets", args.split, args.limit_ids)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    models, checkpoint_paths = load_task_models(args.tasks, device)
    sample_keys, sample_labels = select_examples(dataset) if "task1" in args.tasks else (set(), [])
    sample_outputs: dict[tuple[str, str, str], dict] = {}
    failures: list[dict] = []
    values: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    classifier_targets: list[int] = []
    classifier_predictions: list[int] = []
    classifier_confusion = np.zeros((4, 4), dtype=np.int64)
    weight_values: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    records: list[dict] = []

    for batch in loader:
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        labels = batch["label"].to(device)
        true_labels = labels.detach().cpu().numpy()
        batch_size = image.shape[0]
        outputs: dict[str, torch.Tensor] = {}
        classifier_probabilities = None
        predicted_labels = None
        hard_oracle = None
        hard_predicted = None
        soft_weights = None

        with torch.inference_mode():
            if "task1" in args.tasks:
                outputs["task1"] = models["task1"](image)
            if "task2" in args.tasks:
                logits = models["classifier"](image)
                classifier_probabilities = logits.softmax(1)
                predicted_labels = logits.argmax(1)
                classifier_targets.extend(true_labels.tolist())
                classifier_predictions.extend(predicted_labels.cpu().tolist())
                classifier_confusion += confusion_matrix(true_labels, predicted_labels.cpu().numpy(), labels=np.arange(4))
                branches = torch.stack([image] + [expert(image) for expert in models["hard_experts"]], dim=1)
                oracle_index = labels[:, None, None, None, None].expand(-1, 1, *branches.shape[2:])
                predicted_index = predicted_labels[:, None, None, None, None].expand(-1, 1, *branches.shape[2:])
                hard_oracle = branches.gather(1, oracle_index).squeeze(1)
                hard_predicted = branches.gather(1, predicted_index).squeeze(1)
            if "task3" in args.tasks:
                outputs["task3"], soft_weights = models["task3"](image)

        task_metrics: dict[str, dict[str, np.ndarray]] = {}
        if "task1" in args.tasks:
            task_metrics["task1"] = reconstruction_metrics(outputs["task1"], target)
        if "task2" in args.tasks:
            task_metrics["hard_oracle"] = reconstruction_metrics(hard_oracle, target)
            task_metrics["hard_predicted"] = reconstruction_metrics(hard_predicted, target)
        if "task3" in args.tasks:
            task_metrics["task3"] = reconstruction_metrics(outputs["task3"], target)

        for index in range(batch_size):
            identifier = batch["id"][index]
            label = LABELS[int(true_labels[index])]
            severity = batch["severity"][index]
            row = {"id": identifier, "split": args.split, "label": label, "severity": severity}
            for task_name, metrics in task_metrics.items():
                for metric_name, metric_values in metrics.items():
                    value = float(metric_values[index])
                    row[f"{task_name}_{metric_name}"] = value if np.isfinite(value) else None
                    values[task_name][(label, severity)][metric_name].append(value)
            if classifier_probabilities is not None:
                probabilities = classifier_probabilities[index].cpu().tolist()
                prediction = int(predicted_labels[index].item())
                row["predicted_label"] = LABELS[prediction]
                row["classifier_probabilities"] = json.dumps(probabilities)
            if soft_weights is not None:
                weights = soft_weights[index].cpu().numpy()
                weight_values[(label, severity)].append(weights)
                row["soft_weights"] = json.dumps(weights.tolist())
                for branch, weight in zip(LABELS, weights):
                    row[f"soft_weight_{branch}"] = float(weight)
            records.append(row)

            key = (identifier, label, severity)
            if key in sample_keys:
                sample_outputs[key] = {
                    "id": identifier, "label": label, "severity": severity,
                    "mae": float(task_metrics["task1"]["mae"][index]),
                    "target": image_array(target[index]), "input": image_array(image[index]),
                    "output": image_array(outputs["task1"][index]),
                }
            if label in LABELS[1:]:
                mae = float(task_metrics["task1"]["mae"][index]) if "task1" in args.tasks else -1.0
                if "task1" in args.tasks:
                    failures.append({
                        "id": identifier, "label": label, "severity": severity, "mae": mae,
                        "target": image_array(target[index]), "input": image_array(image[index]),
                        "output": image_array(outputs["task1"][index]),
                    })
                    failures.sort(key=lambda item: item["mae"], reverse=True)
                    del failures[4:]

    summary: dict = {
        "split": args.split,
        "manifest_cases": len(dataset),
        "tasks": {},
        "checkpoints": {name: {"path": str(path), "sha256": sha256(path)}
                        for name, path in checkpoint_paths.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    for task_name, groups in values.items():
        summary["tasks"][task_name] = {"per_condition_severity": grouped_means(groups)}
    if classifier_targets:
        precision, recall, f1, support = precision_recall_fscore_support(
            classifier_targets, classifier_predictions, labels=np.arange(4), zero_division=0)
        normalized = classifier_confusion / np.maximum(classifier_confusion.sum(axis=1, keepdims=True), 1)
        summary["tasks"]["classifier"] = {
            "accuracy": float(np.mean(np.asarray(classifier_targets) == classifier_predictions)),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "confusion_matrix": classifier_confusion.tolist(),
            "confusion_matrix_normalized": normalized.tolist(),
            "per_class": {label: {"precision": float(precision[index]), "recall": float(recall[index]),
                                  "f1": float(f1[index]), "support": int(support[index])}
                          for index, label in enumerate(LABELS)},
        }
    if weight_values:
        summary["tasks"]["task3"]["mean_routing_weights"] = {
            f"{label}/{severity}": np.mean(weights, axis=0).tolist()
            for (label, severity), weights in weight_values.items()
        }
        render_routing_heatmap(weight_values, args.output / "soft_routing_heatmap.png")

    args.output.mkdir(parents=True, exist_ok=True)
    if classifier_targets:
        render_confusion_matrix(classifier_confusion, args.output / "classifier_confusion.png")
    with (args.output / "pet_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = list(dict.fromkeys(key for row in records for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    if "task1" in args.tasks:
        examples = [sample_outputs[(case_id, label, severity)]
                    for case_id, label, severity in sample_keys if (case_id, label, severity) in sample_outputs]
        examples.sort(key=lambda item: (LABELS.index(item["label"]), SEVERITIES.index(item["severity"]), item["id"]))
        render_grid(examples, args.output / "task1_examples.png", "Task 1 deterministic held-out examples")
        render_grid(failures[:4], args.output / "task1_failures.png", "Task 1 highest-error corruption cases")
        summary["tasks"]["task1"]["representative_examples"] = len(examples)
        summary["tasks"]["task1"]["failure_examples"] = len(failures)
        summary["tasks"]["task1"]["example_conditions"] = [f"{label}/{severity}" for label, severity in sample_labels]
    (args.output / "pet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(records), "tasks": list(summary["tasks"]),
                      "elapsed_seconds": summary["elapsed_seconds"], "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()