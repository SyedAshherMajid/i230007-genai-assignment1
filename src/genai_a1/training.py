"""Shared losses, metrics, run bookkeeping, and checkpoint serialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruction_loss(output: torch.Tensor, target: torch.Tensor, alpha: float = 0.8) -> tuple[torch.Tensor, dict]:
    mae = F.l1_loss(output, target)
    similarity = ssim(output.float(), target.float(), data_range=1.0, size_average=True)
    loss = alpha * mae + (1 - alpha) * (1 - similarity)
    return loss, {"mae": float(mae.detach()), "ssim": float(similarity.detach())}


@torch.no_grad()
def reconstruction_metrics(output: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    output = output.clamp(0, 1).float()
    target = target.float()
    mae = (output - target).abs().flatten(1).mean(1)
    mse = (output - target).square().flatten(1).mean(1)
    psnr = torch.where(mse > 0, -10 * torch.log10(mse.clamp_min(1e-12)), torch.inf)
    similarities = ssim(output, target, data_range=1.0, size_average=False)
    return {"mae": mae.cpu().numpy(), "psnr": psnr.cpu().numpy(),
            "ssim": similarities.cpu().numpy()}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, config: dict, best_score: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "epoch": epoch, "config": config, "best_score": best_score,
                  "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                  "python_rng": random.getstate(), "saved_at": time.time()}
    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


def append_metrics(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
