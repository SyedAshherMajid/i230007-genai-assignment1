"""Held-out FS2K evaluation and style/failure image grids."""
from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from genai_a1.fs2k import FS2KDataset
from genai_a1.models import SketchGenerator
from genai_a1.training import reconstruction_metrics, sha256

ROOT = Path(__file__).resolve().parents[1]


def to_pil(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().cpu().clamp(-1, 1).numpy().transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def make_panel(items: list[tuple[Image.Image, Image.Image, Image.Image, str]], output: Path) -> None:
    canvas = Image.new("RGB", (128 * 3, len(items) * 152), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (photo, target, generated, label) in enumerate(items):
        top = row * 152
        for col, image in enumerate((photo, target, generated)):
            canvas.paste(image, (col * 128, top + 20))
        draw.text((3, top + 3), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    checkpoint_path = ROOT / "artifacts/task4/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = SketchGenerator(int(config["base"]), int(config["embedding_dim"]),
                                float(config["dropout"])).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    dataset = FS2KDataset(ROOT / "data/processed/fs2k", "test")
    loader = DataLoader(dataset, batch_size=16, num_workers=0)
    rows = []
    samples = defaultdict(list)
    failures = []
    first_photo = None
    with torch.no_grad():
        for batch in loader:
            photo = batch["photo"].to(device)
            sketch = batch["sketch"].to(device)
            style = batch["style"].to(device)
            output = generator(photo, style)
            metrics = reconstruction_metrics((output + 1) / 2, (sketch + 1) / 2)
            for i, identifier in enumerate(batch["id"]):
                row = {"id": identifier, "style": int(style[i]),
                       **{key: float(value[i]) for key, value in metrics.items()}}
                rows.append(row)
                if len(samples[row["style"]]) < 3 or len(failures) < 4 or row["ssim"] < max(x[0]["ssim"] for x in failures):
                    entry = (row, to_pil(photo[i]), to_pil(sketch[i]), to_pil(output[i]))
                    if len(samples[row["style"]]) < 3:
                        samples[row["style"]].append(entry)
                    failures.append(entry)
                    failures.sort(key=lambda item: item[0]["ssim"])
                    failures = failures[:4]
                if first_photo is None:
                    first_photo = photo[i:i+1].clone()
    out = ROOT / "artifacts/evaluation"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fs2k_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "style", "mae", "ssim", "psnr"])
        writer.writeheader(); writer.writerows(rows)
    summary = {"checkpoint_sha256": sha256(checkpoint_path), "n": len(rows),
               "per_style": {}, "overall": {}, "style_macro": {}}
    for style in (0, 1, 2):
        subset = [row for row in rows if row["style"] == style]
        summary["per_style"][str(style)] = {"n": len(subset), **{metric: float(np.mean([x[metric] for x in subset]))
                                  for metric in ("mae", "ssim", "psnr")}}
    for metric in ("mae", "ssim", "psnr"):
        summary["overall"][metric] = float(np.mean([row[metric] for row in rows]))
        summary["style_macro"][metric] = float(np.mean([summary["per_style"][str(i)][metric] for i in (0, 1, 2)]))
    (out / "fs2k_summary.json").write_text(json.dumps(summary, indent=2))
    chosen = []
    for style in (0, 1, 2):
        for record, photo, target, generated in samples[style]:
            chosen.append((photo, target, generated, f"Style {style+1} | {record['id']} | SSIM {record['ssim']:.3f}"))
    make_panel(chosen, ROOT / "report/figures/task4_samples.png")
    make_panel([(photo, target, generated, f"Failure | {record['id']} | SSIM {record['ssim']:.3f}")
                for record, photo, target, generated in failures],
               ROOT / "report/figures/task4_failures.png")
    if first_photo is not None:
        with torch.no_grad():
            variants = [generator(first_photo, torch.tensor([s], device=device)) for s in (0, 1, 2)]
        triplets = [(to_pil(first_photo[0]), to_pil(variants[s][0]), to_pil(variants[s][0]),
                     f"Same photo | Style {s+1}") for s in (0, 1, 2)]
        make_panel(triplets, ROOT / "report/figures/task4_style_variants.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
