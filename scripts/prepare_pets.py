"""Verify official Oxford archives; create shared splits and deterministic evaluations."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from genai_a1.corruptions import LABELS, SEVERITIES, recipe

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/pets"
OUT = ROOT / "data/processed/pets"
ARCHIVES = {
    "images.tar.gz": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
    "annotations.tar.gz": "95a8c909bbe2e81eed6a22bccdf3f68f",
}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_safe(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            destination = (RAW / member.name).resolve()
            if not destination.is_relative_to(RAW.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe tar entry: {member.name}")
        source.extractall(RAW)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    use_official_images = (RAW / "images.tar.gz").exists() and (RAW / "images.tar.gz").stat().st_size == 791918971
    for filename, expected in ARCHIVES.items():
        if filename == "images.tar.gz" and not use_official_images:
            continue
        archive = RAW / filename
        actual = md5_file(archive)
        if actual != expected:
            raise ValueError(f"{filename}: MD5 {actual}, expected {expected}")
        marker = RAW / (filename + ".extracted")
        if not marker.exists():
            extract_safe(archive)
            marker.write_text(actual)
    annotations = RAW / "annotations"
    images = RAW / "images"

    def read_split(name: str) -> list[tuple[str, int]]:
        rows = []
        for line in (annotations / name).read_text().splitlines():
            columns = line.split()
            if columns and not columns[0].startswith("#"):
                rows.append((columns[0], int(columns[1])))
        return rows

    development = read_split("trainval.txt")
    test = read_split("test.txt")
    assert len(development) == 3680 and len(test) == 3669
    all_ids = [item[0] for item in development + test]
    all_id_set = set(all_ids)
    development_ids = {item[0] for item in development}
    assert len(all_ids) == len(set(all_ids))
    train, val = train_test_split(development, test_size=0.2, random_state=42,
                                  stratify=[breed for _, breed in development])
    assert len(train) == 2944 and len(val) == 736
    OUT.mkdir(parents=True, exist_ok=True)
    resized = OUT / "images"
    resized.mkdir(exist_ok=True)
    if use_official_images:
        for identifier in tqdm(all_ids, desc="Decode and resize pets"):
            target = resized / f"{identifier}.png"
            if not target.exists():
                with Image.open(images / f"{identifier}.jpg") as source:
                    image = source.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
                    image.save(target, optimize=True)
        image_source = "Oxford-IIIT Pet official images archive"
    else:
        import pyarrow.parquet as parquet
        observed = set()
        for split, parquet_path in (("trainval", RAW / "hf_train.parquet"), ("test", RAW / "hf_test.parquet")):
            for batch in parquet.ParquetFile(parquet_path).iter_batches(batch_size=64, columns=["image_id", "image"]):
                ids = batch.column(0).to_pylist()
                images_batch = batch.column(1).to_pylist()
                for identifier, content in zip(ids, images_batch):
                    if identifier in observed or identifier not in all_id_set:
                        raise ValueError(f"Unexpected or duplicate HF image ID: {identifier}")
                    observed.add(identifier)
                    if split == "trainval" and identifier not in development_ids:
                        raise ValueError(f"Development/test split mismatch: {identifier}")
                    target = resized / f"{identifier}.png"
                    if not target.exists():
                        with Image.open(BytesIO(content["bytes"])) as source:
                            source.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC).save(target, optimize=True)
            print(f"Read {split} images from {parquet_path.name}", flush=True)
        if observed != all_id_set:
            raise ValueError(f"HF and Oxford IDs differ: {len(observed)} vs {len(all_ids)}")
        image_source = "timm/oxford-iiit-pet Hugging Face mirror, IDs checked against official split annotations"
    split_info = {"train": train, "validation": val, "test": test}
    for split, rows in split_info.items():
        write_jsonl(OUT / f"{split}.jsonl", [{"id": identifier, "breed": breed} for identifier, breed in rows])
        array_path = OUT / f"{split}_images.npy"
        if not array_path.exists():
            array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.uint8,
                                               shape=(len(rows), 128, 128, 3))
            for index, (identifier, _) in enumerate(tqdm(rows, desc=f"Cache {split} tensors")):
                with Image.open(resized / f"{identifier}.png") as source:
                    array[index] = np.asarray(source.convert("RGB"), dtype=np.uint8)
            array.flush()
    for split, rows in (("validation", val), ("test", test)):
        cases = []
        for index, (identifier, _) in enumerate(rows):
            for label in LABELS:
                for severity in (SEVERITIES if label != "clean" else (None,)):
                    seed = 42_000_000 + index * 100 + LABELS.index(label) * 10 + (SEVERITIES.index(severity) if severity else 0)
                    cases.append({"id": identifier, "split": split, **recipe(label, seed, severity)})
        write_jsonl(OUT / f"{split}_cases.jsonl", cases)
        assert len(cases) == len(rows) * 10
    info = {"source": image_source, "seed": 42,
            "resize": "128x128 RGB bicubic", "counts": {name: len(rows) for name, rows in split_info.items()},
            "archive_md5": ARCHIVES}
    (OUT / "provenance.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
