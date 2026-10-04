"""Stage the non-test pet split as a private Kaggle input dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/pets"
OUTPUT = ROOT / "cloud/pets_upload"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    filenames = (
        "train.jsonl", "train_images.npy", "validation.jsonl",
        "validation_images.npy", "validation_cases.jsonl", "provenance.json",
    )
    hashes = {}
    for filename in filenames:
        source = SOURCE / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = OUTPUT / filename
        if not destination.exists() or sha256(source) != sha256(destination):
            shutil.copy2(source, destination)
        hashes[filename] = sha256(destination)
    (OUTPUT / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    metadata = {
        "title": "GenAI A1 Private Pet Training Inputs",
        "id": "syedashher/genai-a1-pet-training-inputs",
        "licenses": [{"name": "other"}],
        "description": "Private assignment training and validation arrays only; the official test split is intentionally excluded.",
    }
    (OUTPUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "files": hashes}, indent=2))


if __name__ == "__main__":
    main()