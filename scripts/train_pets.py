"""Train and validate Task 1, Task 2, or Task 3 on shared pet data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

try:
    import mlflow
except ImportError:
    from genai_a1.offline_tracking import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genai_a1.models import ConvAutoencoder, CorruptionClassifier, SoftMixture
from genai_a1.pets import BalancedBatchSampler, PetCasesDataset, PetTrainDataset
from genai_a1.training import (append_metrics, reconstruction_loss, reconstruction_metrics,
                                save_checkpoint, seed_everything, sha256)

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["universal", "classifier", "specialist", "soft"])
    p.add_argument("--expert", type=int, choices=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--bottleneck", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda-ce", type=float, default=0.3)
    p.add_argument("--lambda-balance", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--limit-val-ids", type=int, default=128)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--parents", type=Path, help="Task 2 checkpoint directory for soft mixture")
    return p


def model_from_args(args: argparse.Namespace) -> torch.nn.Module:
    if args.task in ("universal", "specialist"):
        return ConvAutoencoder(args.base, args.bottleneck)
    if args.task == "classifier":
        return CorruptionClassifier(args.base, args.dropout)
    if args.parents is None:
        raise ValueError("Soft mixture requires --parents containing classifier and three expert checkpoints")
    gate = CorruptionClassifier(args.base, args.dropout)
    gate.load_state_dict(torch.load(args.parents / "classifier/best.pt", map_location="cpu", weights_only=False)["model"])
    experts = []
    for label in (1, 2, 3):
        expert = ConvAutoencoder(args.base, args.bottleneck)
        expert.load_state_dict(torch.load(args.parents / f"expert_{label}/best.pt", map_location="cpu", weights_only=False)["model"])
        experts.append(expert)
    return SoftMixture(gate, experts)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, args: argparse.Namespace,
             device: torch.device) -> dict:
    model.eval()
    by_label: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    correct = defaultdict(list)
    weights = defaultdict(list)
    for batch in loader:
        image = batch["input"].to(device)
        target = batch["target"].to(device)
        true = batch["label"].to(device)
        if args.task == "classifier":
            predicted = model(image).argmax(1)
            for label, is_correct in zip(true.tolist(), (predicted == true).tolist()):
                correct[label].append(float(is_correct))
            continue
        if args.task == "soft":
            output, routes = model(image)
            for label, route in zip(true.tolist(), routes.cpu().tolist()):
                weights[label].append(route)
        elif args.task == "specialist":
            output = torch.where((true == args.expert)[:, None, None, None], model(image), image)
        else:
            output = model(image)
        measured = reconstruction_metrics(output, target)
        for index, label in enumerate(true.tolist()):
            for name, values in measured.items():
                by_label[label][name].append(float(values[index]))
    if args.task == "classifier":
        scores = {f"accuracy_{k}": float(np.mean(values)) for k, values in correct.items()}
        scores["score"] = 1.0 - float(np.mean(list(scores.values())))
        return scores
    summary: dict = {}
    for label, values in by_label.items():
        for name, items in values.items():
            summary[f"{name}_{label}"] = float(np.mean(items))
    # Fixed selection metric; score does not change when train loss alpha changes.
    summary["score"] = float(np.mean([0.8 * summary[f"mae_{i}"] + 0.2 * (1 - summary[f"ssim_{i}"])
                                      for i in range(4)]))
    if args.task == "soft":
        for label, values in weights.items():
            summary[f"weights_{label}"] = np.mean(values, axis=0).tolist()
    return summary


def main() -> None:
    args = parser().parse_args()
    if args.task == "specialist" and args.expert is None:
        raise ValueError("--expert is required")
    if args.task in ("classifier", "soft") and args.batch % 4:
        raise ValueError("Classifier and soft mixture require batch divisible by four")
    seed_everything(42)
    data_root = ROOT / "data/processed/pets"
    run_dir = args.run_dir or ROOT / "artifacts" / args.task
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = PetTrainDataset(data_root, args.expert if args.task == "specialist" else None)
    if args.task in ("classifier", "soft"):
        sampler = BalancedBatchSampler(len(train.rows), args.batch)
        train_loader = DataLoader(train, batch_sampler=sampler, num_workers=0)
    else:
        sampler = None
        train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0,
                                  drop_last=True)
    validation = PetCasesDataset(data_root, "validation", args.limit_val_ids)
    val_loader = DataLoader(validation, batch_size=args.batch, num_workers=0)
    model = model_from_args(args).to(device)
    if args.task == "soft":
        optimizer = torch.optim.Adam([{"params": model.gate.parameters(), "lr": args.lr},
                                      {"params": model.experts.parameters(), "lr": args.lr * 0.25}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_score = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"device": str(device), "torch": torch.__version__,
                   "train_split_sha256": sha256(data_root / "train.jsonl"),
                   "validation_cases_sha256": sha256(data_root / "validation_cases.jsonl")})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("GenAI Assignment 1")
    stale_epochs = 0
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params({k: v for k, v in config.items() if v is not None and len(str(v)) < 250})
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.perf_counter()
            train.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            if args.task == "soft":
                frozen = epoch < args.warmup
                for expert in model.experts:
                    for parameter in expert.parameters():
                        parameter.requires_grad_(not frozen)
            losses = []
            for step, batch in enumerate(train_loader):
                image = batch["input"].to(device)
                target = batch["target"].to(device)
                label = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                if args.task == "classifier":
                    logits = model(image)
                    loss = F.cross_entropy(logits, label)
                elif args.task == "soft":
                    output, routes = model(image)
                    reconstruction, _ = reconstruction_loss(output, target, args.alpha)
                    gate_ce = F.cross_entropy(model.gate(image), label)
                    balance = (routes.mean(0) - 0.25).square().mean()
                    loss = reconstruction + args.lambda_ce * gate_ce + args.lambda_balance * balance
                else:
                    output = model(image)
                    loss, _ = reconstruction_loss(output, target, args.alpha)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                if args.max_steps is not None and step + 1 >= args.max_steps:
                    break
            metrics = validate(model, val_loader, args, device)
            metrics.update({"train_loss": float(np.mean(losses)), "epoch": epoch,
                            "elapsed_seconds": time.perf_counter() - epoch_start})
            append_metrics(run_dir / "history.jsonl", metrics)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=epoch)
            improved = metrics["score"] < best_score
            if improved:
                best_score = metrics["score"]
                stale_epochs = 0
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, config, best_score)
            else:
                stale_epochs += 1
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, config, best_score)
            print(json.dumps({"run": run_dir.name, "epoch": epoch, "score": metrics["score"],
                              "train_loss": metrics["train_loss"], "seconds": metrics["elapsed_seconds"]}), flush=True)
            if stale_epochs >= args.patience:
                print(f"Early stop after {stale_epochs} unimproved epochs")
                break
        mlflow.log_artifact(str(run_dir / "config.json"))
        mlflow.log_artifact(str(run_dir / "history.jsonl"))


if __name__ == "__main__":
    main()
