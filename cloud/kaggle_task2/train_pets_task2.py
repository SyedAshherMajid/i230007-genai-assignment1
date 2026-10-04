"""CPU-only Kaggle runner for bounded pet specialist hyperparameter screening."""
from __future__ import annotations

import base64
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
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(json.dumps({"torch": torch.__version__, "device": device_name}), flush=True)

    kaggle_input = Path("/kaggle/input")
    code_bundle = next(kaggle_input.rglob("source_bundle.txt"))
    pet_manifest = next(kaggle_input.rglob("train_images.npy")).parent
    base = Path("/kaggle/temp/genai_a1")
    base.mkdir(parents=True, exist_ok=True)
    source_archive = base / "source.zip"
    source_archive.write_bytes(base64.b64decode(code_bundle.read_text()))
    with zipfile.ZipFile(source_archive) as archive:
        archive.extractall(base)
    data_root = base / "data/processed/pets"
    data_root.mkdir(parents=True, exist_ok=True)
    for source in pet_manifest.iterdir():
        if source.is_file():
            shutil.copy2(source, data_root / source.name)

    source_hashes = json.loads((pet_manifest / "sha256.json").read_text())
    for filename, expected in source_hashes.items():
        import hashlib
        digest = hashlib.sha256((data_root / filename).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Pet input hash mismatch: {filename}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{base / 'vendor'}:{base / 'src'}"
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    python = sys.executable

    def run(command: list[str]) -> None:
        print("RUN", json.dumps(command), flush=True)
        subprocess.run([python, str(base / command[0]), *command[1:]],
                       cwd=base, env=env, check=True)

    studies = base / "artifacts/studies"
    shared = {"lr": 7.855830402514553e-05, "batch": 16, "base": 24,
              "bottleneck": 64, "alpha": 0.6824279936868144}
    run(["scripts/tune.py", "universal", "--trials", "2", "--epochs", "4", "--max-steps", "75",
         "--enqueue-trial", json.dumps({**shared, "dropout": 0.0}),
         "--enqueue-trial", json.dumps({**shared, "dropout": 0.2})])
    universal_destination = Path("/kaggle/working/universal_followup")
    shutil.copytree(studies / "universal", universal_destination, dirs_exist_ok=True)
    tracking_db = base / "mlflow.db"
    if tracking_db.exists():
        shutil.copy2(tracking_db, "/kaggle/working/mlflow.db")
    tracking_artifacts = base / "mlruns"
    if tracking_artifacts.exists():
        shutil.make_archive("/kaggle/working/mlruns", "zip", tracking_artifacts)
    print(json.dumps({"completed": True, "device": device_name,
                      "universal_dropout_followup": json.loads((universal_destination / "best.json").read_text()),
                      "universal_study_root": str(universal_destination)}, indent=2), flush=True)


if __name__ == "__main__":
    main()