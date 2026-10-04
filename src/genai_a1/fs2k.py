"""Paired photo/sketch dataset with shared image transforms."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .pets import read_jsonl, image_tensor


class FS2KDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        if split not in ("train", "validation", "test"):
            raise ValueError(split)
        self.root = Path(root)
        self.rows = read_jsonl(self.root / f"{split}.jsonl")
        self.split = split
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(self.root / row["photo"]) as image:
            photo = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(self.root / row["sketch"]) as image:
            sketch = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if self.split == "train" and random.Random(42 + self.epoch * len(self) + index).random() < 0.5:
            photo = np.ascontiguousarray(photo[:, ::-1])
            sketch = np.ascontiguousarray(sketch[:, ::-1])
        return {"photo": image_tensor(photo).mul_(2).sub_(1),
                "sketch": image_tensor(sketch).mul_(2).sub_(1),
                "style": torch.tensor(row["style"], dtype=torch.long), "id": row["id"]}
