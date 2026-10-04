"""Export trained PyTorch models to ONNX and compare ONNX Runtime outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from genai_a1.models import ConvAutoencoder, CorruptionClassifier, SketchGenerator, SoftMixture
from genai_a1.training import sha256

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def soft_model(config: dict, smoke: bool) -> SoftMixture:
    base = int(config.get("base", 16))
    bottleneck = int(config.get("bottleneck", 48))
    dropout = float(config.get("dropout", 0.1))
    temperature = float(config.get("temperature", 1.0))
    if smoke:
        return SoftMixture(CorruptionClassifier(base, dropout),
                           [ConvAutoencoder(base, bottleneck, dropout) for _ in range(3)], temperature)
    parent_dir = Path(config.get("parents", ROOT / "artifacts/task2"))
    if not parent_dir.is_absolute():
        parent_dir = ROOT / parent_dir
    classifier_state = checkpoint(parent_dir / "classifier/best.pt")
    classifier_config = classifier_state.get("config", {})
    gate = CorruptionClassifier(int(classifier_config.get("base", base)),
                                float(classifier_config.get("dropout", dropout)))
    gate.load_state_dict(classifier_state["model"])
    experts = []
    for label in (1, 2, 3):
        expert_state = checkpoint(parent_dir / f"expert_{label}/best.pt")
        expert_config = expert_state.get("config", {})
        expert = ConvAutoencoder(int(expert_config.get("base", base)),
                                 int(expert_config.get("bottleneck", bottleneck)),
                                 float(expert_config.get("dropout", dropout)))
        expert.load_state_dict(expert_state["model"])
        experts.append(expert)
    return SoftMixture(gate, experts, temperature)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Export untrained architecture smoke checks")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--only", nargs="+", choices=("universal", "classifier", "salt_expert",
                                                     "blur_expert", "occlusion_expert", "soft_moe",
                                                     "sketch_generator"),
                        help="Export selected trained models while other tasks are still training")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    sources = [
        ("universal", ROOT / "artifacts/task1/best.pt", "ae"),
        ("classifier", ROOT / "artifacts/task2/classifier/best.pt", "classifier"),
        ("salt_expert", ROOT / "artifacts/task2/expert_1/best.pt", "ae"),
        ("blur_expert", ROOT / "artifacts/task2/expert_2/best.pt", "ae"),
        ("occlusion_expert", ROOT / "artifacts/task2/expert_3/best.pt", "ae"),
        ("soft_moe", ROOT / "artifacts/task3/best.pt", "soft"),
        ("sketch_generator", ROOT / "artifacts/task4/best.pt", "sketch"),
    ]
    reports = []
    previous_report = args.output / "parity.json"
    if args.only and previous_report.exists():
        reports = [record for record in json.loads(previous_report.read_text())
                   if record["model"] not in args.only]
    for name, source, kind in sources:
        if args.only and name not in args.only:
            continue
        state = {} if args.smoke else checkpoint(source)
        config = state.get("config", {})
        base = int(config.get("base", 16))
        bottleneck = int(config.get("bottleneck", 48))
        dropout = float(config.get("dropout", 0.1))
        if kind == "ae":
            model = ConvAutoencoder(base, bottleneck, dropout)
        elif kind == "classifier":
            model = CorruptionClassifier(base, dropout)
        elif kind == "soft":
            model = soft_model(config, args.smoke)
        else:
            model = SketchGenerator(base, int(config.get("embedding_dim", 8)),
                                    float(config.get("dropout", 0.2)))
        if state:
            model.load_state_dict(state["generator"] if kind == "sketch" else state["model"])
        model.eval()
        image = torch.rand(1, 3, 128, 128)
        inputs = (image, torch.tensor([2], dtype=torch.int64)) if kind == "sketch" else (image,)
        input_names = ["image", "style_id"] if kind == "sketch" else ["image"]
        output_names = ["restored", "weights"] if kind == "soft" else ["output"]
        out = args.output / f"{name}.onnx"
        torch.onnx.export(model, inputs, out, opset_version=17, dynamo=False,
                          input_names=input_names, output_names=output_names,
                          do_constant_folding=True)
        onnx.checker.check_model(onnx.load(out))
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            native = model(*inputs)
        native_outputs = native if isinstance(native, tuple) else (native,)
        runtime = session.run(None, {name: value.numpy() for name, value in zip(input_names, inputs)})
        errors = [float(np.max(np.abs(reference.numpy() - actual)))
                  for reference, actual in zip(native_outputs, runtime)]
        if max(errors) > 0.002:
            raise AssertionError(f"{name}: ONNX parity error {errors}")
        reports.append({"model": name, "source_checkpoint": None if args.smoke else str(source),
                        "source_sha256": None if args.smoke else sha256(source),
                        "onnx_sha256": sha256(out), "max_absolute_errors": errors})
        print(f"{name}: max errors {errors}", flush=True)
    (args.output / "parity.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
