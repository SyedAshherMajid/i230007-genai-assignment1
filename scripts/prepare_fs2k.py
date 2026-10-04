"""Prepare annotation-paired FS2K photos/sketches and a stratified validation split."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/fs2k"
OUT = ROOT / "data/processed/fs2k"


def locate_root() -> Path:
    candidates = [path.parent for path in RAW.rglob("anno_train.json")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one FS2K root, found {candidates}")
    return candidates[0]


def pair_path(base: Path, kind: str, name: str) -> Path:
    stem = name if kind == "photo" else name.replace("photo", "sketch").replace("image", "sketch")
    for extension in (".jpg", ".png"):
        candidate = base / kind / (stem + extension)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {kind} image matching {name}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    archive = RAW / "FS2K.zip"
    marker = RAW / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                destination = (RAW / member.filename).resolve()
                if not destination.is_relative_to(RAW.resolve()):
                    raise ValueError(f"Unsafe ZIP entry: {member.filename}")
            source.extractall(RAW)
        marker.write_text("FS2K authors' public Google Drive archive")
    base = locate_root()
    official_train = json.loads((base / "anno_train.json").read_text())
    official_test = json.loads((base / "anno_test.json").read_text())
    if (len(official_train), len(official_test)) != (1058, 1046):
        raise ValueError("Unexpected FS2K official split size")
    train_rows, validation_rows = train_test_split(official_train, test_size=0.15,
                                                    random_state=42,
                                                    stratify=[row["style"] for row in official_train])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "photo").mkdir(exist_ok=True)
    (OUT / "sketch").mkdir(exist_ok=True)
    all_splits = {"train": train_rows, "validation": validation_rows, "test": official_test}
    for split, rows in all_splits.items():
        manifest = []
        for row in tqdm(rows, desc=f"Prepare FS2K {split}"):
            name = row["image_name"]
            style = int(row["style"])
            if style not in (0, 1, 2):
                raise ValueError(f"Invalid style {style}: {name}")
            identifier = name.replace("/", "_")
            photo_file = OUT / "photo" / (identifier + ".png")
            sketch_file = OUT / "sketch" / (identifier + ".png")
            for kind, target in (("photo", photo_file), ("sketch", sketch_file)):
                if not target.exists():
                    with Image.open(pair_path(base, kind, name)) as source:
                        source.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC).save(target, optimize=True)
            manifest.append({"id": identifier, "style": style,
                             "photo": str(photo_file.relative_to(OUT)).replace("\\", "/"),
                             "sketch": str(sketch_file.relative_to(OUT)).replace("\\", "/")})
        write_jsonl(OUT / f"{split}.jsonl", manifest)
    counts = {key: len(rows) for key, rows in all_splits.items()}
    assert counts == {"train": 899, "validation": 159, "test": 1046}
    (OUT / "provenance.json").write_text(json.dumps({"source": "FS2K authors' archive",
                                                    "seed": 42, "counts": counts,
                                                    "style_mapping": {"0": "Style 1", "1": "Style 2", "2": "Style 3"}}, indent=2))
    print(counts)


if __name__ == "__main__":
    main()
