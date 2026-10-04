"""Reproducible bounded Optuna searches; each trial trains and validates models."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import optuna

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["universal", "classifier", "specialists", "soft", "gan"])
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--parents", type=Path, default=ROOT / "artifacts/task2")
    parser.add_argument("--enqueue-trial", action="append", default=[],
                        help="JSON object of fixed parameters for a controlled trial")
    args = parser.parse_args()
    study_dir = ROOT / "artifacts/studies" / args.task
    study_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=args.task, direction="minimize",
        storage=f"sqlite:///{(study_dir / 'study.db').as_posix()}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3))
    for parameters in args.enqueue_trial:
        study.enqueue_trial(json.loads(parameters))

    def run_trial(trial: optuna.Trial, command: list[str], run_name: str) -> float:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if value)
        process = subprocess.Popen([sys.executable] + command, cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        score = float("inf")
        log_file = study_dir / f"{run_name}.log"
        with log_file.open("w", encoding="utf-8") as log:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                if line.startswith("{"):
                    try:
                        record = json.loads(line)
                        if "score" in record:
                            score = min(score, float(record["score"]))
                            trial.report(score, step=int(record["epoch"]))
                            if trial.should_prune():
                                process.terminate()
                                process.wait(timeout=30)
                                raise optuna.TrialPruned()
                    except (json.JSONDecodeError, KeyError):
                        pass
        return_code = process.wait()
        if return_code != 0 or not score < float("inf"):
            log_tail = log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"Trial failed (exit {return_code}); inspect {log_file}\n{log_tail}")
        return score

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 5e-5, 6e-4, log=True)
        batch = trial.suggest_categorical("batch", [8, 16, 32] if args.task == "classifier" else [4, 8, 16])
        if args.task == "gan":
            d_lr = trial.suggest_float("d_lr", 5e-5, 3e-4, log=True)
            base = trial.suggest_categorical("base", [16, 32])
            embedding = trial.suggest_categorical("embedding_dim", [4, 8, 16])
            dropout = trial.suggest_categorical("dropout", [0.0, 0.2, 0.5])
            lambda_l1 = trial.suggest_categorical("lambda_l1", [50.0, 100.0, 150.0])
            command = ["scripts/train_fs2k.py", "--epochs", str(args.epochs), "--max-steps", str(args.max_steps),
                       "--batch", str(batch), "--g-lr", str(lr), "--d-lr", str(d_lr),
                       "--base", str(base), "--embedding-dim", str(embedding), "--dropout", str(dropout),
                       "--lambda-l1", str(lambda_l1), "--limit-val", "90", "--patience", "20",
                       "--run-dir", str(study_dir / f"trial_{trial.number}")]
            return run_trial(trial, command, f"trial_{trial.number}")
        params = ["--epochs", str(args.epochs), "--max-steps", str(args.max_steps),
                  "--batch", str(batch), "--lr", str(lr), "--limit-val-ids", "128", "--patience", "8"]
        if args.task in ("universal", "specialists"):
            base = trial.suggest_categorical("base", [8, 16, 24])
            bottleneck = trial.suggest_categorical("bottleneck", [24, 48, 64])
            alpha = trial.suggest_float("alpha", 0.5, 0.9)
            params += ["--base", str(base), "--bottleneck", str(bottleneck), "--alpha", str(alpha)]
            if args.task == "universal":
                dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
                params += ["--dropout", str(dropout)]
        if args.task == "classifier":
            base = trial.suggest_categorical("base", [8, 16, 24])
            dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.3])
            weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])
            params += ["--base", str(base), "--dropout", str(dropout),
                       "--weight-decay", str(weight_decay)]
        if args.task == "soft":
            alpha = trial.suggest_float("alpha", 0.5, 0.9)
            temperature = trial.suggest_categorical("temperature", [0.5, 1.0, 2.0])
            ce = trial.suggest_float("lambda_ce", 0.1, 0.7)
            balance = trial.suggest_float("lambda_balance", 0.001, 0.1, log=True)
            warmup = trial.suggest_int("warmup", 2, 5)
            params += ["--alpha", str(alpha), "--temperature", str(temperature),
                       "--lambda-ce", str(ce), "--lambda-balance", str(balance),
                       "--warmup", str(warmup), "--parents", str(args.parents)]
        if args.task == "specialists":
            results = []
            for expert in (1, 2, 3):
                command = ["scripts/train_pets.py", "specialist", "--expert", str(expert),
                           "--run-dir", str(study_dir / f"trial_{trial.number}_expert_{expert}")] + params
                results.append(run_trial(trial, command, f"trial_{trial.number}_expert_{expert}"))
            return sum(results) / len(results)
        command = ["scripts/train_pets.py", args.task, "--run-dir", str(study_dir / f"trial_{trial.number}")] + params
        return run_trial(trial, command, f"trial_{trial.number}")

    study.optimize(objective, n_trials=args.trials)
    (study_dir / "best.json").write_text(json.dumps({"value": study.best_value,
                                                      "params": study.best_params}, indent=2))
    print(json.dumps({"best_value": study.best_value, "best_params": study.best_params}))


if __name__ == "__main__":
    main()
