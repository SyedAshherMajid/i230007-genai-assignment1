"""Train the blur specialist on the private Kaggle train/validation pet data."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import torch


def main() -> None:
    torch.set_num_threads(2)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(json.dumps({"torch": torch.__version__, "device": device}), flush=True)
    source = Path("/kaggle/input")
    code = next(source.rglob("source_bundle.txt"))
    pet_manifest = next(source.rglob("train_images.npy")).parent
    root = Path("/kaggle/temp/genai_a1")
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "source.zip"
    archive_path.write_bytes(base64.b64decode(code.read_text()))
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)
    data = root / "data/processed/pets"
    data.mkdir(parents=True, exist_ok=True)
    for path in pet_manifest.iterdir():
        if path.is_file():
            shutil.copy2(path, data / path.name)
    expected = json.loads((pet_manifest / "sha256.json").read_text())
    for filename, value in expected.items():
        if hashlib.sha256((data / filename).read_bytes()).hexdigest() != value:
            raise ValueError(f"Pet input hash mismatch: {filename}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'vendor'}:{root / 'src'}"
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    output = Path("/kaggle/working/expert_2")
    subprocess.run([sys.executable, str(root / "scripts/train_pets.py"), "specialist",
                    "--expert", "2", "--epochs", "35", "--batch", "4",
                    "--lr", "0.0005567734622861566", "--base", "24",
                    "--bottleneck", "64", "--alpha", "0.5557975442608167",
                    "--max-steps", "200", "--limit-val-ids", "736",
                    "--patience", "8", "--run-dir", str(output)],
                   cwd=root, env=env, check=True)
    print(json.dumps({"completed": True, "device": device,
                      "history_rows": len((output / "history.jsonl").read_text().splitlines())}), flush=True)


if __name__ == "__main__":
    main()
