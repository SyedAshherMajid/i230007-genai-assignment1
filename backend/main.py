"""Single-image inference for all four assignment workspaces."""
from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path
import time

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import numpy as np
import onnxruntime as ort

from genai_a1.corruptions import LABELS, apply_corruption, recipe

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
app = FastAPI(title="Restore & Sketch Lab", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:8080"],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
sessions: dict[str, ort.InferenceSession] = {}
NEEDED = ("universal", "classifier", "salt_expert", "blur_expert", "occlusion_expert",
          "soft_moe", "sketch_generator")


def session(name: str) -> ort.InferenceSession:
    if name not in NEEDED:
        raise HTTPException(400, "Unknown model")
    if name not in sessions:
        path = MODEL_DIR / f"{name}.onnx"
        if not path.is_file():
            raise HTTPException(503, f"Model {name} is not available yet")
        sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sessions[name]


def png_data_url(image: np.ndarray) -> str:
    with BytesIO() as stream:
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(stream, format="PNG")
        return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def to_batch(image: np.ndarray, sketch: bool = False) -> np.ndarray:
    batch = image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return batch * 2 - 1 if sketch else batch


def to_image(batch: np.ndarray, sketch: bool = False) -> np.ndarray:
    image = batch[0].transpose(1, 2, 0)
    if sketch:
        image = (image + 1) / 2
    return (np.clip(image, 0, 1) * 255).round().astype(np.uint8)


@app.get("/health")
def health() -> dict:
    available = {name: (MODEL_DIR / f"{name}.onnx").is_file() for name in NEEDED}
    workspaces = {
        "universal": available["universal"],
        "hard": all(available[name] for name in ("classifier", "salt_expert", "blur_expert", "occlusion_expert")),
        "soft": available["soft_moe"],
        "sketch": available["sketch_generator"],
    }
    return {"status": "ready" if all(workspaces.values()) else "models_pending",
            "models": available, "workspaces": workspaces}


@app.post("/infer/{task}")
async def infer(task: str, file: UploadFile = File(...),
                corruption: str = Query("none"), severity: str = Query("medium"),
                seed: int = Query(42), style: int = Query(1, ge=1, le=3)) -> dict:
    if task not in ("universal", "hard", "soft", "sketch"):
        raise HTTPException(404, "Unknown workspace")
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 10 MB")
    try:
        with Image.open(BytesIO(data)) as source:
            source.load()
            image = np.asarray(source.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Upload a valid image")
    if task == "sketch" and corruption != "none":
        raise HTTPException(400, "Corruption controls are for restoration workspaces")
    if corruption != "none":
        if corruption not in LABELS:
            raise HTTPException(400, "Unknown corruption type")
        try:
            image = apply_corruption(image, recipe(corruption, seed, severity))
        except ValueError as error:
            raise HTTPException(400, str(error))
    started = time.perf_counter()
    result: dict = {"task": task, "input": png_data_url(image), "resolution": [128, 128]}
    if task == "sketch":
        output = session("sketch_generator").run(None, {"image": to_batch(image, sketch=True),
                                                        "style_id": np.array([style - 1], dtype=np.int64)})[0]
        result["style"] = style
        result["output"] = png_data_url(to_image(output, sketch=True))
    else:
        batch = to_batch(image)
        if task == "universal":
            output = session("universal").run(None, {"image": batch})[0]
        elif task == "hard":
            logits = session("classifier").run(None, {"image": batch})[0][0]
            shift = np.exp(logits - logits.max())
            probabilities = (shift / shift.sum()).tolist()
            selected = int(np.argmax(logits))
            result.update({"probabilities": probabilities, "predicted_label": LABELS[selected],
                           "selected_branch": "identity" if selected == 0 else LABELS[selected] + " expert"})
            output = batch if selected == 0 else session(("", "salt_expert", "blur_expert", "occlusion_expert")[selected]).run(None, {"image": batch})[0]
        else:
            output, weights = session("soft_moe").run(None, {"image": batch})
            result["weights"] = weights[0].tolist()
        result["output"] = png_data_url(to_image(output))
    result["inference_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result
