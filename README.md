# Generative AI Assignment 1

Syed Ashher Majid (23i-0007), FAST NUCES Islamabad, CS(A). Course: Generative AI. Instructor: Dr Akhtar Jamil.

This repository implements four image-to-image tasks: universal pet restoration, classifier-guided specialist restoration, a soft mixture of experts, and style-conditioned face-to-sketch generation. See [WORK_STATUS.md](WORK_STATUS.md) for the current handoff and [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for the detailed requirements and design decisions.

## Environment

The training environment used Python 3.10 and PyTorch 2.7.1 with CUDA 11.8. On Windows, create an isolated environment and install the pinned requirements:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu118
$env:PYTHONPATH = 'src'
$env:MLFLOW_DISABLE_AGENT_HINT = '1'
```

Use a CPU-only PyTorch build if CUDA is unavailable. The FastAPI container installs only `requirements-backend.txt`; the frontend production image is built by Docker Compose.

## Data

Keep raw and processed data under `data/`; these files are intentionally excluded from Git. For pets, place the official annotations and images under `data/raw/pets/` (or the verified Hugging Face parquet mirror files `hf_train.parquet` and `hf_test.parquet` alongside the official annotation files), then run:

```powershell
python scripts/prepare_pets.py
```

The script verifies official IDs, makes the seed-42 2,944/736/3,669 split, caches 128-pixel RGB images, and writes deterministic validation/test corruption manifests. It excludes the official test split from training.

Download and prepare FS2K with:

```powershell
python scripts/download_fs2k.py
python scripts/prepare_fs2k.py
```

The FS2K split is 899/159/1,046 train/validation/test pairs. The test split is reserved for final evaluation.

## Training And Evaluation

Run one local GPU job at a time. Optuna databases, checkpoints, logs, and metrics are written under ignored `artifacts/` and `mlflow.db` paths.

1. Task 1: search and train the universal autoencoder.

```powershell
python scripts/tune.py universal --trials 3 --epochs 4 --max-steps 75
python scripts/train_pets.py universal --epochs 35 --batch 16 --lr 0.00010333490157966143 --base 24 --bottleneck 48 --alpha 0.5260206371941119 --dropout 0.1 --limit-val-ids 736 --patience 8 --run-dir artifacts/task1
```

2. Task 2: tune the classifier and shared specialist configuration, train the classifier, then train experts 1 (salt-and-pepper), 2 (blur), and 3 (occlusion) separately. Use the selected settings recorded in each study's `best.json`.

```powershell
python scripts/tune.py classifier --trials 4 --epochs 6 --max-steps 120
python scripts/train_pets.py classifier --epochs 30 --batch 16 --lr 0.00021791779554628546 --base 24 --dropout 0 --weight-decay 0 --limit-val-ids 736 --patience 6 --run-dir artifacts/task2/classifier
python scripts/tune.py specialists --trials 3 --epochs 4 --max-steps 60
```

For each expert, run `scripts/train_pets.py specialist --expert {1,2,3}` with the selected `--lr`, `--batch`, `--base`, `--bottleneck`, and `--alpha`, and a separate `--run-dir artifacts/task2/expert_{n}`. Do not share expert weights.

3. Task 3: search and train from the completed Task 2 checkpoints. The gate and experts load their architectures from those parent checkpoints; gate warm-up precedes joint fine-tuning.

```powershell
python scripts/tune.py soft --trials 3 --epochs 6 --max-steps 40 --parents artifacts/task2
python scripts/train_pets.py soft --epochs 25 --batch 4 --parents artifacts/task2 --run-dir artifacts/task3
```

4. Task 4: tune and train the style-conditioned GAN, then evaluate the official FS2K test pairs.

```powershell
python scripts/tune.py gan --trials 4 --epochs 8 --max-steps 100
python scripts/train_fs2k.py --epochs 100 --batch 4 --g-lr 0.00010648845527550903 --d-lr 0.00014965254242886746 --base 32 --embedding-dim 16 --dropout 0.5 --lambda-l1 100 --patience 20 --run-dir artifacts/task4
python scripts/evaluate_fs2k.py
```

After all pet checkpoints are final, evaluate the same deterministic official test cases across tasks:

```powershell
python scripts/evaluate_pets.py --tasks task1 task2 task3 --split test
```

The evaluator writes case-level CSV, summary JSON, per-condition/severity metrics, Task 1 example/failure panels, classifier confusion data, hard-routing oracle/predicted results, and Task 3 routing weights/heatmap under `artifacts/evaluation/`.

## Models And Application

After all seven trained checkpoints exist, export them and check PyTorch/ONNX parity:

```powershell
python scripts/export_models.py
```

This writes the required inference graphs to `models/`. The untrained smoke exports are not substitutes for these final trained models. Start the frontend and backend containers together:

```powershell
docker compose up --build
```

Open `http://localhost:8080`. The API health endpoint is proxied at `http://localhost:8080/api/health`; the application reports missing models until all expected ONNX files are present. Stop containers with `Ctrl+C`.

For local development, start the API with `uvicorn backend.main:app --reload --port 8000` and the frontend from `frontend/` with `npm ci` followed by `npm run dev`. The Vite development server uses the configured API proxy.

## Tests And Report

```powershell
python -m pytest tests -q
python -m pip check
python scripts/evaluate_pets.py --help
python scripts/export_models.py --help
& tools\tectonic.exe -X compile report\main.tex --outdir report\build
```

The IEEE LaTeX sources and generated result tables are under `report/`. Final pet results, application screenshots, trained ONNX files, and the demonstration link are added as their verification is completed. Raw datasets, virtual environments, checkpoints, caches, credentials, and mutable experiment stores must remain out of Git.

