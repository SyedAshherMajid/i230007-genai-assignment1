"""Train a style-conditioned pix2pix-style generator and PatchGAN discriminator."""
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

from genai_a1.fs2k import FS2KDataset
from genai_a1.models import SketchGenerator, StylePatchDiscriminator
from genai_a1.training import append_metrics, reconstruction_metrics, seed_everything, sha256

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--g-lr", type=float, default=2e-4)
    p.add_argument("--d-lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--limit-val", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/task4")
    p.add_argument("--resume", type=Path)
    return p


@torch.no_grad()
def validate(generator: SketchGenerator, loader: DataLoader, device: torch.device) -> dict:
    generator.eval()
    by_style = defaultdict(lambda: defaultdict(list))
    for batch in loader:
        photo = batch["photo"].to(device)
        sketch = batch["sketch"].to(device)
        style = batch["style"].to(device)
        output = generator(photo, style)
        metrics = reconstruction_metrics((output + 1) / 2, (sketch + 1) / 2)
        for i, label in enumerate(style.tolist()):
            for name, values in metrics.items():
                by_style[label][name].append(float(values[i]))
    summary = {f"{name}_{style}": float(np.mean(values))
               for style, measures in by_style.items() for name, values in measures.items()}
    summary["score"] = float(np.mean([0.8 * summary[f"mae_{i}"] + 0.2 * (1 - summary[f"ssim_{i}"])
                                      for i in (0, 1, 2)]))
    return summary


def save(path: Path, generator, discriminator, g_optimizer, d_optimizer, epoch, config, best_score):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(), "d_optimizer": d_optimizer.state_dict(),
                "epoch": epoch, "config": config, "best_score": best_score}, temp)
    temp.replace(path)


def main() -> None:
    args = parser().parse_args()
    seed_everything(42)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "data/processed/fs2k"
    train = FS2KDataset(data_root, "train")
    val = FS2KDataset(data_root, "validation")
    if args.limit_val:
        # Fixed deterministic panel with every style represented.
        chosen = set()
        per_style = max(1, args.limit_val // 3)
        for label in (0, 1, 2):
            indices = [i for i, row in enumerate(val.rows) if row["style"] == label]
            chosen.update(indices[int(i)] for i in np.linspace(0, len(indices) - 1,
                                                                min(per_style, len(indices))))
        val.rows = [row for i, row in enumerate(val.rows) if i in chosen]
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val, batch_size=args.batch, num_workers=0)
    generator = SketchGenerator(args.base, args.embedding_dim, args.dropout).to(device)
    discriminator = StylePatchDiscriminator(args.base, args.embedding_dim).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    start_epoch = 0
    best_score = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"device": str(device), "torch": torch.__version__,
                   "train_split_sha256": sha256(data_root / "train.jsonl"),
                   "validation_split_sha256": sha256(data_root / "validation.jsonl")})
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2))
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("GenAI Assignment 1")
    stale = 0
    with mlflow.start_run(run_name=args.run_dir.name):
        mlflow.log_params({k: v for k, v in config.items() if v is not None and len(str(v)) < 250})
        for epoch in range(start_epoch, args.epochs):
            train.set_epoch(epoch)
            start = time.perf_counter()
            generator.train()
            discriminator.train()
            totals = defaultdict(list)
            for step, batch in enumerate(train_loader):
                photo = batch["photo"].to(device)
                target = batch["sketch"].to(device)
                style = batch["style"].to(device)
                generated = generator(photo, style)

                d_optimizer.zero_grad(set_to_none=True)
                real_logits = discriminator(photo, target, style)
                fake_logits = discriminator(photo, generated.detach(), style)
                d_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                d_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                d_loss = (d_real + d_fake) * 0.5
                d_loss.backward()
                d_optimizer.step()

                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                g_optimizer.zero_grad(set_to_none=True)
                fake_logits = discriminator(photo, generated, style)
                g_adversarial = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                g_l1 = F.l1_loss(generated, target)
                g_loss = g_adversarial + args.lambda_l1 * g_l1
                g_loss.backward()
                g_optimizer.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                for key, value in (("d_real", d_real), ("d_fake", d_fake),
                                   ("g_adversarial", g_adversarial), ("g_l1", g_l1)):
                    totals[key].append(float(value.detach()))
                if args.max_steps and step + 1 >= args.max_steps:
                    break
            measured = validate(generator, val_loader, device)
            measured.update({key: float(np.mean(values)) for key, values in totals.items()})
            measured.update({"epoch": epoch, "elapsed_seconds": time.perf_counter() - start})
            append_metrics(args.run_dir / "history.jsonl", measured)
            mlflow.log_metrics(measured, step=epoch)
            if measured["score"] < best_score:
                best_score = measured["score"]
                stale = 0
                save(args.run_dir / "best.pt", generator, discriminator, g_optimizer, d_optimizer,
                     epoch, config, best_score)
            else:
                stale += 1
            save(args.run_dir / "last.pt", generator, discriminator, g_optimizer, d_optimizer,
                 epoch, config, best_score)
            print(json.dumps({"epoch": epoch, "score": measured["score"],
                              "seconds": measured["elapsed_seconds"]}), flush=True)
            if epoch >= 49 and stale >= args.patience:
                break
        mlflow.log_artifact(str(args.run_dir / "config.json"))
        mlflow.log_artifact(str(args.run_dir / "history.jsonl"))


if __name__ == "__main__":
    main()
