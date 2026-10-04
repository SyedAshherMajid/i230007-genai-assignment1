"""Pet image loaders shared by the universal, specialist, and routing tasks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .corruptions import LABELS, apply_corruption, recipe


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def image_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div_(255)


class PetTrainDataset(Dataset):
    def __init__(self, root: str | Path, specialist: int | None = None, seed: int = 42):
        self.root = Path(root)
        self.rows = read_jsonl(self.root / "train.jsonl")
        self.images = np.load(self.root / "train_images.npy", mmap_mode="r")
        self.specialist = specialist
        self.seed = seed
        self.epoch = 0
        if specialist is not None and specialist not in (1, 2, 3):
            raise ValueError("Specialist label must be 1, 2, or 3")

    def __len__(self) -> int:
        return len(self.rows) * (1 if self.specialist is not None else 4)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        label = self.specialist if self.specialist is not None else index % 4
        image_index = index if self.specialist is not None else index // 4
        identifier = self.rows[image_index]["id"]
        clean = self.images[image_index]
        # Unique per sample and epoch; deterministic independently of worker scheduling.
        seed = self.seed + self.epoch * len(self) + index
        settings = recipe(LABELS[label], seed)
        damaged = apply_corruption(clean, settings)
        return {"input": image_tensor(damaged), "target": image_tensor(clean),
                "label": torch.tensor(label, dtype=torch.long), "id": identifier}


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, image_count: int, batch_size: int, seed: int = 42):
        if batch_size % 4:
            raise ValueError("Balanced batches need batch_size divisible by 4")
        self.image_count = image_count
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (self.image_count * 4) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(len(self)):
            image_indices = rng.integers(0, self.image_count, size=self.batch_size)
            labels = np.tile(np.arange(4), self.batch_size // 4)
            rng.shuffle(labels)
            yield (image_indices * 4 + labels).tolist()


class PetCasesDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "validation", limit_ids: int | None = None):
        if split not in ("validation", "test"):
            raise ValueError(split)
        self.root = Path(root)
        self.cases = read_jsonl(self.root / f"{split}_cases.jsonl")
        self.ids = {row["id"]: i for i, row in enumerate(read_jsonl(self.root / f"{split}.jsonl"))}
        self.images = np.load(self.root / f"{split}_images.npy", mmap_mode="r")
        if limit_ids is not None:
            selected = sorted({case["id"] for case in self.cases})[:limit_ids]
            self.cases = [case for case in self.cases if case["id"] in set(selected)]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict:
        case = self.cases[index]
        clean = self.images[self.ids[case["id"]]]
        damaged = apply_corruption(clean, case)
        return {"input": image_tensor(damaged), "target": image_tensor(clean),
                "label": torch.tensor(LABELS.index(case["label"]), dtype=torch.long),
                "id": case["id"], "severity": case["severity"]}

