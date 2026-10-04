"""Package the FS2K archive and source for a private Kaggle GPU notebook."""
from __future__ import annotations

import json
import base64
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "cloud/upload"
KERNEL = ROOT / "cloud/kaggle_kernel"
CODE_UPLOAD = ROOT / "cloud/code_upload"


def main() -> None:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    CODE_UPLOAD.mkdir(parents=True, exist_ok=True)
    KERNEL.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "data/raw/fs2k/FS2K.zip"
    if not archive.exists():
        raise FileNotFoundError(archive)
    destination = UPLOAD / "FS2K.zip"
    if not destination.exists():
        import os
        os.link(archive, destination)
    source_zip = UPLOAD / "assignment_source.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for location in (ROOT / "src", ROOT / "scripts"):
            for file in location.rglob("*.py"):
                zipped.write(file, file.relative_to(ROOT))
        vendor = ROOT / ".venv/Lib/site-packages/pytorch_msssim"
        for file in vendor.glob("*.py"):
            zipped.write(file, Path("vendor/pytorch_msssim") / file.name)
        license_file = ROOT / ".venv/Lib/site-packages/pytorch_msssim-1.0.0.dist-info/LICENSE"
        zipped.write(license_file, "vendor/pytorch_msssim/LICENSE")
    (CODE_UPLOAD / "source_bundle.txt").write_text(base64.b64encode(source_zip.read_bytes()).decode("ascii"))
    (CODE_UPLOAD / "dataset-metadata.json").write_text(json.dumps({
        "title": "GenAI A1 Private Training Code",
        "id": "syedashher/genai-a1-private-training-code",
        "licenses": [{"name": "other"}],
        "description": "Private academic experiment source bundle by Syed Ashher Majid."
    }, indent=2))
    (UPLOAD / "dataset-metadata.json").write_text(json.dumps({
        "title": "GenAI A1 FS2K Private Training Inputs",
        "id": "syedashher/genai-a1-fs2k-private-inputs",
        "licenses": [{"name": "other"}],
        "description": "Private course experiment inputs. FS2K original archive from Deng-Ping Fan and collaborators; source code by Syed Ashher Majid. Not for public redistribution."
    }, indent=2))
    notebook = {
        "cells": [{"cell_type": "code", "id": "train-fs2k", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": [line + "\n" for line in '''import os, pathlib, subprocess, sys, zipfile, shutil, base64, io
base = pathlib.Path('/kaggle/temp/a1')
base.mkdir(parents=True, exist_ok=True)
source = pathlib.Path('/kaggle/input')
code = next(source.rglob('source_bundle.txt'))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(code.read_text()))) as z: z.extractall(base)
raw = base / 'data/raw/fs2k'
raw.mkdir(parents=True, exist_ok=True)
official_root = next(source.rglob('anno_train.json')).parent
shutil.copytree(official_root, raw / 'FS2K', dirs_exist_ok=True)
(raw / '.extracted').write_text('Kaggle private copy of FS2K authors archive')
env = os.environ.copy()
env['PYTHONPATH'] = str(base / 'vendor') + ':' + str(base / 'src')
subprocess.check_call([sys.executable, str(base / 'scripts/prepare_fs2k.py')], cwd=base, env=env)
out = pathlib.Path('/kaggle/working/task4_baseline')
subprocess.check_call([sys.executable, str(base / 'scripts/train_fs2k.py'), '--epochs', '35', '--batch', '8', '--base', '16', '--limit-val', '90', '--run-dir', str(out)], cwd=base, env=env)
if (base / 'mlflow.db').exists(): shutil.copy2(base / 'mlflow.db', '/kaggle/working/task4_mlflow.db')
if (base / 'mlruns').exists(): shutil.make_archive('/kaggle/working/task4_mlflow_artifacts', 'zip', base / 'mlruns')
if (base / 'artifacts/offline_tracking').exists(): shutil.make_archive('/kaggle/working/task4_offline_logs', 'zip', base / 'artifacts/offline_tracking')'''.splitlines()]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    (KERNEL / "train_fs2k.ipynb").write_text(json.dumps(notebook, indent=2))
    (KERNEL / "kernel-metadata.json").write_text(json.dumps({
        "id": "syedashher/genai-a1-fs2k-baseline",
        "title": "GenAI A1 FS2K Baseline",
        "code_file": "train_fs2k.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["syedashher/genai-a1-fs2k-private-inputs", "syedashher/genai-a1-private-training-code"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []
    }, indent=2))
    print("Private Kaggle dataset and notebook package prepared")


if __name__ == "__main__":
    main()
