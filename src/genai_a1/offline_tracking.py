"""Small local event store for cloud runtimes without package-index access.

Cloud records are imported into the required local MLflow experiment afterward.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time


class OfflineTracker:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[2] / "artifacts/offline_tracking"
        self.experiment = "experiment"
        self.run_dir = None

    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    @contextmanager
    def start_run(self, run_name: str):
        self.run_dir = self.root / f"{run_name}_{int(time.time())}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "experiment.txt").write_text(self.experiment)
        try:
            yield self
        finally:
            self.run_dir = None

    def log_params(self, params: dict) -> None:
        (self.run_dir / "params.json").write_text(json.dumps(params, indent=2))

    def log_metrics(self, metrics: dict, step: int) -> None:
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({"step": step, "metrics": metrics}) + "\n")

    def log_artifact(self, source: str) -> None:
        folder = self.run_dir / "files"
        folder.mkdir(exist_ok=True)
        shutil.copy2(source, folder / Path(source).name)


mlflow = OfflineTracker()
