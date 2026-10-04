# Project handoff and live work status

## Current Snapshot (5 October 2026)

**This snapshot is authoritative and supersedes any inconsistent phase rows, process notes, or numbered steps below. Do not stop active training jobs.**

### Active work

- Local GTX 1060 job: salt-and-pepper Task 2 specialist, **original process still active** (Windows process IDs 10548 wrapper / 7856 worker, started 00:08 local time). Latest completed epoch at this update is 23, score 0.06336. A PowerShell process listing briefly appeared empty, leading to a duplicate resume process; the newer duplicate (IDs 24256/24568, session `10679`) was immediately stopped before it wrote a new epoch. History has no duplicate epochs. **Do not resume or start another local GPU job while PID 7856 is alive.** Check `Get-CimInstance Win32_Process` with `CommandLine` as well as checkpoint/history if listing output looks blank.
- Kaggle kernel `syedashher/genai-a1-pet-tasks-1-and-2`, version 6, is RUNNING at last check. It now trains blur expert 2 from `cloud/kaggle_task2/train_blur_expert.py` with the selected shared hyperparameters, 35 epochs, 200 steps per epoch, full 736-ID validation, and patience 8. GPU was requested; previous Kaggle runs received CPU only, so check actual runtime log before claiming GPU usage. Download checkpoint and logs only after completion. One Kaggle account is used.
- The version-5 Kaggle matched Task 1 dropout comparison completed and downloaded to ignored `cloud/kaggle_task2/output/universal_followup/`: dropout 0.0 score 0.14382; dropout 0.2 score 0.14877 after four matched epochs. This supports keeping the existing Task 1 final checkpoint; it is not a full-length comparison.

### Task status

- **Task 1: training/evaluation complete.** Universal model trained for 35 epochs; best full-validation objective 0.07750 at epoch 33. Final test ran over 36,690 deterministic cases. Summary/CSV and the 12-example/four-highest-error panels are in `artifacts/evaluation/pets_task1/`. Test findings are in `report/generated/results.tex`: high occlusion is hardest (MAE 0.0599, SSIM 0.6414); clean case MAE is 0.0355. Best checkpoint SHA-256: `ef7278947d625c434bd5be5958b7c3cf16b50d3fa84129de99e8f1624c7a22d8`. The controlled dropout follow-up is complete; see active-work note above.
- **Task 2: partially complete.** Classifier Optuna search: 3 COMPLETE, 1 PRUNED; selected lr 0.0002179178, batch 16, base 24, dropout 0, weight decay 0. Classifier early-stopped at epoch 28; best checkpoint epoch 22, full-validation accuracy 0.99579 and macro-F1 0.99313 at `artifacts/task2/classifier/best.pt`. Shared specialist search: 3 trials complete; best objective 0.09516 with lr 0.0005567735, batch 4, base 24, bottleneck 64, alpha 0.555798. Expert 1 is active as above; experts 2 and 3, final hard-router test metrics, and routing failure analysis remain.
- **Task 3: implementation present, training not started.** Soft mixture, temperature, summed balance regularizer, warm-up/fine-tuning path, architecture-aware Task 2 parent loading, and parent-aware exporter are implemented and smoke-checked. It must wait for all three final Task 2 expert checkpoints. Then run a bounded `soft` Optuna study on validation, train from those exact parents, and report per-condition/severity weights plus the routing heatmap.
- **Task 4: local training/evaluation complete.** Local FS2K GAN trained 68 epochs and evaluated on all 1,046 official held-out pairs: MAE 0.1153, SSIM 0.4509, PSNR 14.83. Results, figures, and study records are under `artifacts/evaluation/`, `report/figures/`, and `artifacts/studies/gan/`. Kaggle version 6 also completed CPU-only with a 90-ID validation subset; its epoch-31 `best.pt` is valid, but downloading `last.pt` ended in an incomplete transfer. Treat local results as primary.

### Report, application, and repository

- `report/generated/results.tex` now contains Task 1 per-severity results and figures, Task 2 classifier validation metrics/confusion figure, and measured FS2K metrics/figures. `report/main.tex` contains measured Task 1 architecture/compression details and evidence-based initial discussion. `report/build/main.pdf` compiles; remaining messages are existing fontconfig/underfull-box warnings. Still needed: Task 2 hard-routing and Task 3 results, app screenshots, final limitations/conclusion, and YouTube link.
- README setup/run instructions are populated. Stitch designs, React/Tailwind shell, FastAPI routes, Compose configuration, and Docker image builds/smoke tests exist. Trained Task 1 universal, Task 2 classifier, and Task 4 sketch ONNX graphs are now in ignored `models/`, with max parity errors below 4e-6. FastAPI TestClient returned 200 for real trained Task 1 and Task 4 requests. The backend health API and UI now report readiness by workspace and refresh every 20 seconds. React production build passes. Docker was restarted and rebuilding in session `95610` at last check. Export the remaining four graphs after their checkpoints exist, then retest complete API/Compose workflows.
- `pytest tests -q` last passed (7 tests); changed Python files had no editor diagnostics. `pip check` and `docker compose config --quiet` passed earlier.
- Git HEAD was local commit `5d2d6ca` before this handoff update; changes remain uncommitted. Preserve the pre-existing `scripts/prepare_fs2k.py` uppercase-extension fix and all generated report changes. The user has now explicitly asked to try pushing to the public GitHub repository. Previous push failed HTTP 403 due missing contents-write permission; do not block model/report work on it. Never put credentials, datasets, or checkpoints in Git.

### Next sequence

1. Monitor the original expert 1 process 7856 through early-stop/completion; preserve `best.pt`, `last.pt`, history, and config. The duplicate session `10679` was stopped and must not be resumed. Inspect process command lines and checkpoint before any later resume.
2. Poll Kaggle v6 blur expert; after it completes, download `expert_2/best.pt`, history, config, and log, verify checkpoint/model and full-validation score. If Kaggle CPU output is materially weaker than a local full fit, schedule a local fine-tune after expert 1/3. Do not run concurrent local GPU jobs.
3. Train occlusion expert 3 on the local GTX 1060 as soon as expert 1 exits, using selected shared settings. The Task 1 dropout follow-up is done and does not require retraining Task 1.
4. Tune/train Task 3 only after Task 2 classifier and all expert checkpoints exist. Keep the exact parent lineage, gate warm-up, and lower joint learning rate.
5. Run `python scripts/evaluate_pets.py --tasks task1 task2 task3 --split test` only after model selection is frozen. Add hard oracle/predicted and Task 3 severity/weight results to the report.
6. Export and parity-check trained models, verify all API workflows and Docker Compose health, finish the report, then record/upload the 5–7 minute demo.
7. Commit progress locally if appropriate; do not push to GitHub unless the user later asks.

Legacy handoff notes below were last updated on 4 October 2026; use the Current Snapshot above for live status and continuation order. `GenAI_Assignment#1.pdf` remains authoritative. Do not place secrets in Git, logs, or this file.

## Latest verified state

- The local `main` branch is based on handoff commit `5d2d6ca`. Before any edits, the worktree had one pre-existing modification: `scripts/prepare_fs2k.py`, adding case-insensitive archive extension variants. Preserve it; do not include it in unrelated commits.
- Task 1 universal training completed 35 epochs. Its best full-validation score is 0.07750 at epoch 33. Official test evaluation completed on all 36,690 deterministic cases in 102.6 seconds. Results, case CSV, and the 12-example/four-highest-error panels are under `artifacts/evaluation/pets_task1/`; the selected checkpoint SHA-256 is `ef7278947d625c434bd5be5958b7c3cf16b50d3fa84129de99e8f1624c7a22d8`. The report now includes the per-severity table and both figures and compiles successfully.
- Kaggle kernel `syedashher/genai-a1-fs2k-baseline` reports `COMPLETE` after version 6. Its 35-row history and 500-byte config downloaded. The 14,742,069-byte `best.pt` loads successfully at epoch 31; the checkpoint records a CPU-only PyTorch 2.11 run and a 90-ID validation subset. The first output download ended with an incomplete-read error while fetching `last.pt`, which is currently zero bytes. Do not treat the cloud run as a replacement for the local FS2K run or compare its subset score as a like-for-like final test.
- `scripts/prepare_fs2k.py` is a user change and fixes uppercase `.JPG`/`.PNG` extension resolution needed on Kaggle/Linux. Its source package is already on kernel version 6. Keep the change and verify it remains intact.
- The PDF was read directly (9 pages). Critical evaluation requirements include deterministic severity manifests with untouched official test data, oracle and predicted hard routing, Task 3 initialized from Task 2 and gate warm-up before joint tuning, Optuna across all four tasks, trained-model ONNX parity, all four workflows in Docker Compose, an IEEE LaTeX report, and a 5–7 minute demo.
- Private Kaggle train/validation pet inputs (official test split excluded) and source bundle are uploaded. Kaggle runs CPU-only despite a T4 request; version 5 is running a matched Task 1 dropout comparison (0.0 vs 0.2) with corrected vendored `PYTHONPATH` handling. Earlier versions exposed missing accelerator allocation and a dropped vendored-SSIM path; do not request another GPU or use another account.
- Task 2 classifier search completed with three COMPLETE trials and one PRUNED. Best objective was 0.08054 (learning rate 0.0002179, batch 16, base 24, dropout 0, weight decay 0). The selected classifier stopped at epoch 28; its best checkpoint is epoch 22, with validation accuracy 0.99579 and macro-F1 0.99313, at `artifacts/task2/classifier/best.pt`.
- The shared specialist Optuna search completed three trials in session `cb85db30-0637-4f40-9081-bd3e5fb877f0`. Best objective was 0.09516 with lr 0.0005568, batch 4, base 24, bottleneck 64, alpha 0.5558. Salt-and-pepper expert training is now active in session `082cbbda-80bf-47a7-a02d-b7ad29890d86`, using the full validation split.
- Added `scripts/evaluate_pets.py` for per-condition/per-severity metrics, Task 1's 12-example and four highest-error grids, classifier reports/confusion image, hard oracle/predicted routing, Task 3 routing-weight summaries/heatmap, case CSV, summary JSON, and checkpoint hashes. A CPU validation smoke test on two IDs completed in 6.5 seconds and produced 12 examples/four failures. Full evaluation waits for final checkpoints.
- Corrected clean-case severity handling, made autoencoder dropout, classifier weight decay, and soft-gate temperature effective, added architecture-aware Task 3 parent loading, and matched the PDF's summed balance penalty. The classifier trainer now records accuracy, macro/per-class precision/recall/F1, and raw/normalized confusion matrices. Seven focused tests pass; touched Python files have no editor diagnostics.
- Fixed tuner subprocess environment inheritance so Kaggle's vendored dependencies stay available, added captured child-log tails to failed Optuna trials, and made ONNX soft-mixture construction respect each Task 2 parent architecture.
- **Current sequence supersedes the older numbered continuation block below:** finish salt, blur, and occlusion expert training sequentially; tune/train Task 3 from those final Task 2 checkpoints; then run combined held-out evaluation, trained ONNX export/parity, API/Compose checks, and final report/demo work.

## Goal and constraints

Finish all four model tasks, tuning, evaluation, tracking, React/Tailwind and FastAPI/ONNX application, Docker Compose, IEEE report, public repository, and demo. The user has authorized autonomous decisions and implementation, including public repository creation and use of their existing Kaggle account. Do not ask routine permission. Use one Kaggle account and one local GPU job at a time. Keep the requested personal deadline out of all files and messages.

## Phase-by-phase state

| Plan phase / deliverable | State and evidence | Next action |
|---|---|---|
| Requirements and repository | PDF read; plan written; Git `main` initialized; public GitHub repository `SyedAshherMajid/i230007-genai-assignment1` created; initial local commit `6218196`. | Commit current work locally. Git push currently fails 403 because stored GitHub credential lacks repository contents write permission; preserve commits, later solve authorization without disclosing tokens. |
| Environment and data | `.venv` Python 3.10 has CUDA PyTorch 2.7.1/cu118, torchvision, Optuna, ONNX, ORT, FastAPI and tracking dependencies. `.venv-tools` Python 3.14 has Kaggle CLI. Official Oxford Pet annotations and FS2K archive downloaded. Pet images came from verified HF mirror; all IDs checked against official split. Processed pet split is 2,944/736/3,669; FS2K is 899/159/1,046 train/val/test. Data is under ignored `data/`. | Run `pip check` and inspect data provenance; no need to redownload. |
| Shared modeling and training | `src/genai_a1/` has corruption, data, models, training, and offline tracking modules. `scripts/` has pet/FS2K prep, train, tune, export, and FS2K evaluation. Four corruption tests passed. ONNX smoke export and parity passed for seven untrained graphs, errors near 1e-6. | Finish pet training/tuning/evaluation; check final exports with trained checkpoints. |
| Task 4, FS2K | Four bounded local Optuna attempts completed (three successful, one pruned/failed); best config in `artifacts/studies/gan/best.json`. Full local run exited normally after 68 epochs; `best.pt`, `last.pt`, history, MLflow DB are under ignored `artifacts/task4`/`mlflow.db`. Held-out 1,046-case evaluation is in `artifacts/evaluation/fs2k_summary.json` and case CSV; figures in `report/figures/`. Overall MAE 0.1153, SSIM 0.4509, PSNR 14.83. | Include results and figure discussion in report; compare cloud result if it succeeds. |
| Parallel Kaggle Task 4 | Kernel version 6 reports `COMPLETE`. The 35-epoch history/config and valid best checkpoint (epoch 31) are downloaded. It ran on CPU with only 90 validation IDs; `last.pt` download failed and is zero bytes. | Retry only the missing `last.pt` transfer if it is useful. Treat as supplementary; local Task 4 is the primary result. |
| Tasks 1–3, pets | Task 1 is trained/evaluated on the official test cases; Task 2 classifier is trained (validation macro-F1 0.99313); shared specialist search is complete and expert 1 is training. Experts 2–3 and Task 3 remain. | Finish all experts, initialize/train Task 3, then evaluate all systems on identical held-out cases. |
| Kaggle pet training | Private train/validation-only input and refreshed source datasets are ready. Kernel v1 requested a T4 but exited at startup because CUDA was unavailable (`torch 2.11.0+cpu`). | Continue on the local GPU; only retry Kaggle if GPU allocation is independently confirmed. |
| UI/API/container | Stitch project `projects/16665216233764559974` produced four design screens in `design/`. React/Tailwind frontend builds with `npm run build`. FastAPI TestClient returned 200 for all four smoke-model workflows and 400 for invalid image. `docker compose config --quiet` passed; Docker build was started. | Check Docker build result; integrate trained ONNX models and repeat API/Compose smoke tests. |
| Report and demo | IEEE report now contains Task 1 severity metrics and examples/failures, Task 2 classifier validation metrics/confusion figure, and measured FS2K results. Tectonic compiles with existing underfull-box/fontconfig warnings. Hard-routing and Task 3 results, application screenshots, discussion, limitations, and demo link remain. | Add expert/MoE results, complete analysis, then record/upload the demo last. |

## Active and recent processes

Process/session IDs are only hints: they may disappear when an agent or terminal restarts. The only local GPU job is salt-and-pepper expert training in session `082cbbda-80bf-47a7-a02d-b7ad29890d86`. Kaggle kernel v4 is running CPU-only Task 1 dropout screening. Task 1 and the Task 2 classifier/search are complete; experts 2–3 and Task 3 are pending. Local FS2K training exited normally after 68 epochs; pet preparation completed with provenance and all splits. Kaggle FS2K kernel version 6 is complete. `tools/tectonic.exe` exists. Seven focused corruption/model tests pass, and initial untrained ONNX parity smoke checks passed earlier.

Useful read-only checks from PowerShell in the project root:

```powershell
Get-Process python,pythonw,docker -ErrorAction SilentlyContinue | Select-Object Name,Id,CPU,Path
Get-Content artifacts/task4/history.jsonl -Tail 3
Get-Content data/processed/pets/provenance.json
git status --short
docker compose ps
```

## Exact continuation order

1. Task 1 is complete; see `artifacts/evaluation/pets_task1/pet_summary.json` and `report/generated/results.tex` for official test metrics.
2. Finish sequential expert training with the selected shared settings, then run Task 2 oracle/predicted routing evaluation on the fixed official test manifest.
2. Optionally retry just Kaggle's missing `last.pt` with the existing account and process-memory token setup: `$env:KAGGLE_API_TOKEN=[Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')`; `$env:PYTHONIOENCODING='utf-8'`; then `& .venv-tools\Scripts\kaggle.exe kernels output syedashher/genai-a1-fs2k-baseline -p cloud\output --file-pattern 'last\.pt' -o`. Never print the token. The kernel itself is complete; do not push another version without a concrete need.
3. Use `artifacts/evaluation/fs2k_summary.json`, `fs2k_cases.csv`, and `report/figures/task4_*.png` for Task 4 report discussion. Local Task 4 is the primary result; the Kaggle run is CPU-only and uses 90 validation IDs. The local evaluator uses a fixed checkpoint path and ignores command arguments.
4. After Task 1 evaluation, run bounded Task 2 classifier/specialist tuning and training, then Task 3 soft mixture initialized from Task 2 (`scripts/train_pets.py --help`). Reuse the 3 GB GTX 1060 and keep one local GPU job active. Evaluate on deterministic validation/test manifests; never tune on the official test set.
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
