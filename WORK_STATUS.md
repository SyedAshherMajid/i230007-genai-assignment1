# Project handoff and live work status

Updated: 4 October 2026. This file is a restart guide for any agent continuing the assignment. Read [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for the full requirements, phase definitions, hardware inventory, and intended deliverables. Read `GenAI_Assignment#1.pdf` for the authoritative assignment. The execution plan's opening status paragraph describes the state when it was written and is now stale; use this file for actual progress. Do not place secrets in Git, logs, or this file.

## Goal and constraints

Finish all four model tasks, tuning, evaluation, tracking, React/Tailwind and FastAPI/ONNX application, Docker Compose, IEEE report, public repository, and demo. The user has authorized autonomous decisions and implementation, including public repository creation and use of their existing Kaggle account. Do not ask routine permission. Use one Kaggle account and one local GPU job at a time. Keep the requested personal deadline out of all files and messages.

## Phase-by-phase state

| Plan phase / deliverable | State and evidence | Next action |
|---|---|---|
| Requirements and repository | PDF read; plan written; Git `main` initialized; public GitHub repository `SyedAshherMajid/i230007-genai-assignment1` created; initial local commit `6218196`. | Commit current work locally. Git push currently fails 403 because stored GitHub credential lacks repository contents write permission; preserve commits, later solve authorization without disclosing tokens. |
| Environment and data | `.venv` Python 3.10 has CUDA PyTorch 2.7.1/cu118, torchvision, Optuna, ONNX, ORT, FastAPI and tracking dependencies. `.venv-tools` Python 3.14 has Kaggle CLI. Official Oxford Pet annotations and FS2K archive downloaded. Pet images came from verified HF mirror; all IDs checked against official split. Processed pet split is 2,944/736/3,669; FS2K is 899/159/1,046 train/val/test. Data is under ignored `data/`. | Run `pip check` and inspect data provenance; no need to redownload. |
| Shared modeling and training | `src/genai_a1/` has corruption, data, models, training, and offline tracking modules. `scripts/` has pet/FS2K prep, train, tune, export, and FS2K evaluation. Four corruption tests passed. ONNX smoke export and parity passed for seven untrained graphs, errors near 1e-6. | Finish pet training/tuning/evaluation; check final exports with trained checkpoints. |
| Task 4, FS2K | Four bounded local Optuna attempts completed (three successful, one pruned/failed); best config in `artifacts/studies/gan/best.json`. Full local run exited normally after 68 epochs; `best.pt`, `last.pt`, history, MLflow DB are under ignored `artifacts/task4`/`mlflow.db`. Held-out 1,046-case evaluation is in `artifacts/evaluation/fs2k_summary.json` and case CSV; figures in `report/figures/`. Overall MAE 0.1153, SSIM 0.4509, PSNR 14.83. | Include results and figure discussion in report; compare cloud result if it succeeds. |
| Parallel Kaggle Task 4 | Private input/code datasets exist. Baseline kernel versions 1–4 failed due path mistakes then Kaggle runtime DNS preventing pip. Revised notebook vendors SSIM and uses offline tracking, so no pip is needed. Version 5 pushed and was RUNNING at last check. | Poll `kaggle kernels status`; download logs/output after completion. Do not assume earlier failed runs produced training results. |
| Tasks 1–3, pets | Clean/corrupted deterministic train/validation/test data prepared. Architectures and training script exist. No completed pet training, tuning, or metrics yet. | Tune small budgets, train universal restoration, hard router plus three specialists, and soft mixture initialized from task 2; evaluate on fixed held-out cases. Task 3 depends on task 2 weights. |
| UI/API/container | Stitch project `projects/16665216233764559974` produced four design screens in `design/`. React/Tailwind frontend builds with `npm run build`. FastAPI TestClient returned 200 for all four smoke-model workflows and 400 for invalid image. `docker compose config --quiet` passed; Docker build was started. | Check Docker build result; integrate trained ONNX models and repeat API/Compose smoke tests. |
| Report and demo | IEEE `report/main.tex` drafted; measured FS2K table and sample/failure figures have been inserted in `report/generated/results.tex`. Tectonic downloaded and first compile started. Pet tables/discussion are still missing. YouTube demo intentionally last. | Finish LaTeX compile, then fill pet results, interpretation, limitations, and demo link. |

## Active and recent processes

Process/session IDs are only hints: they may disappear when an agent or terminal restarts. Check files and process list first. Local FS2K training exited 0 after 68 epochs (prior session `91163`); pet preparation finished with `data/processed/pets/provenance.json` and all three split arrays. **Universal pet Optuna tuning is active** in session `67380`: `scripts/tune.py universal --trials 3 --epochs 4 --max-steps 75`, first trial already completed. Kaggle kernel version 5 was RUNNING at last check. Docker Compose build is in session `75939`. Tectonic report compile is in session `73752`; binary `tools/tectonic.exe` exists. Tests and `pip check` passed. Inspect current output/status before restarting anything.

Useful read-only checks from PowerShell in the project root:

```powershell
Get-Process python,pythonw,docker -ErrorAction SilentlyContinue | Select-Object Name,Id,CPU,Path
Get-Content artifacts/task4/history.jsonl -Tail 3
Get-Content data/processed/pets/provenance.json
git status --short
docker compose ps
```

## Exact continuation order

1. Poll active universal tuning session or inspect `artifacts/studies/universal/` to see whether it finished. Do not launch another local GPU job concurrently. `pip check` and four corruption tests already passed.
2. Poll Kaggle version 5: set the CLI token only in process memory with `$env:KAGGLE_API_TOKEN=[Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')`; use `& .venv-tools\Scripts\kaggle.exe kernels status syedashher/genai-a1-fs2k-baseline`; obtain logs/output with `kernels output ... -p cloud/output -o`. Never print the token.
3. Use `artifacts/evaluation/fs2k_summary.json`, `fs2k_cases.csv`, and `report/figures/task4_*.png` for Task 4 report discussion; compare cloud result if available. The evaluator currently uses a fixed checkpoint path and ignores command arguments.
4. After universal tuning, run the full Task 1 model, then bounded Task 2 classifier/specialist tuning and training, then Task 3 soft mixture initialized from Task 2 (`scripts/train_pets.py --help`). Reuse the 3 GB GTX 1060 and stop unproductive trials early. Evaluate deterministic validation/test cases; do not tune on test.
5. Export final trained models to ONNX via `scripts/export_models.py --help`; validate PyTorch/ONNX parity, then FastAPI requests and Docker Compose startup. Check existing Docker build output or rerun build if no build is active.
6. Finish the IEEE report from real results. `tools/tectonic.exe` exists; poll current compilation or rerun `& tools\tectonic.exe -X compile report\main.tex --outdir report\build` after creating `report/build/`. Add pet results and replace the remaining draft discussion/conclusion. Record a 5–7 minute UI/demo, upload to user's YouTube channel at the very end, add link to README/report as needed.
7. Commit all code/docs (never datasets, checkpoints, credentials, caches). Retry push only after GitHub contents-write authorization is available. The public repo exists, but a successful `git push` has **not** occurred.

## Commands and paths

Use `$env:PYTHONPATH='src'` for direct training/evaluation scripts. The project interpreter is `.venv\Scripts\python.exe`, the Kaggle CLI `.venv-tools\Scripts\kaggle.exe`, and Docker Compose uses `compose.yaml`. Keep downloaded data and large outputs on D:. `data/`, `artifacts/`, `models/`, tracking DBs, cloud upload/output bundles, and tools binaries are ignored by Git. The trained model paths, metrics, and report tables must be recorded in committed documentation before the handoff is considered final.

The last known Task 4 local training command was:

```powershell
$env:PYTHONPATH='src'
$env:MLFLOW_DISABLE_AGENT_HINT='1'
& .venv\Scripts\python.exe scripts\train_fs2k.py --epochs 100 --batch 4 --g-lr 0.00010648845527550903 --d-lr 0.00014965254242886746 --base 32 --embedding-dim 16 --dropout 0.5 --lambda-l1 100 --patience 20 --run-dir artifacts\task4
```

Do not rerun this from scratch if the checkpoint and completed history are sound. Check whether `--resume` exists in script help before resuming any interrupted run.

## Known blockers and decisions

- GitHub remote is `https://github.com/SyedAshherMajid/i230007-genai-assignment1`. Initial API authentication could create the repo, but Git blob upload failed with HTTP 403 `Resource not accessible by personal access token`; existing SSH key was unauthorized. Local commits remain valid. Do not store or print credentials.
- Kaggle Internet toggle did not make PyPI resolvable in the earlier run. Vendored `pytorch_msssim` plus an offline logger remove that dependency. One verified Kaggle account is sufficient; do not use multiple accounts to evade free-tier limits.
- The official Pet image tar download was slow and stopped; the complete HF parquet mirror is local and checked against the official annotation IDs. The official annotation MD5 was verified. Do not present the partial tar as a completed download.
- The plan's historical environment and account notes may contradict the current state. This file records implementation progress; the PDF controls assignment requirements.
