"""Stage inspected Python 3.10 wheels using hard links, preserving the pip cache."""
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
inventory = json.loads((ROOT / "planning" / "cached_packages.json").read_text())
out = ROOT / "wheelhouse"
out.mkdir(exist_ok=True)
preferred = {"torch", "torchvision", "numpy", "pillow", "scikit_learn", "scipy"}
for item in inventory["wheels"]:
    if item["name"].lower().replace("-", "_") not in preferred:
        continue
    if "cp310-cp310-win_amd64" not in item["tags"]:
        continue
    if item["name"].lower() == "torch" and item["version"] != "2.7.1+cu118":
        continue
    if item["name"].lower() == "torchvision" and item["version"] != "0.22.1+cu118":
        continue
    source = Path(item["path"])
    with zipfile.ZipFile(source) as wheel:
        if wheel.testzip() is not None:
            raise ValueError(f"Damaged wheel: {source}")
        wheel_meta = next(x for x in wheel.namelist() if x.endswith(".dist-info/WHEEL"))
        metadata = next(x for x in wheel.namelist() if x.endswith(".dist-info/METADATA"))
        if item["name"].replace("-", "_").lower() not in metadata.lower():
            raise ValueError(f"Package metadata mismatch: {source}")
        if not wheel.read(wheel_meta):
            raise ValueError(f"Empty WHEEL metadata: {source}")
    filename = f"{item['name'].replace('-', '_')}-{item['version']}-cp310-cp310-win_amd64.whl"
    target = out / filename
    if not target.exists():
        os.link(source, target)
    print(filename)
