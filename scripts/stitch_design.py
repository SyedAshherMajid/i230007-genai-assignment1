"""Create original Stitch design evidence for the four-workspace application."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.request
import winreg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "design"
OUT.mkdir(exist_ok=True)


def call(name: str, args: dict) -> dict:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        api_key = winreg.QueryValueEx(key, "STITCH_API_KEY")[0]
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": args}}
    request = urllib.request.Request("https://stitch.googleapis.com/mcp",
        data=json.dumps(payload).encode(),
        headers={"X-Goog-Api-Key": api_key, "Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"})
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.load(response)
    if "error" in data or data.get("result", {}).get("isError"):
        raise RuntimeError(json.dumps(data.get("error", data.get("result")))[:600])
    return data["result"]


PROMPT = """Design a polished, responsive DESKTOP web application dashboard named Restore & Sketch Lab for an academic computer vision project. Four clear workspaces in a left sidebar or top navigation: 1 Universal Restoration, 2 Hard-Routed Restoration, 3 Soft Mixture of Experts, 4 Face-to-Sketch. The active workspace is Universal Restoration. Main content: generous drag-and-drop image upload area, a selector for using the image as already damaged or applying a chosen synthetic corruption (clean, salt-and-pepper, Gaussian blur, black rectangle occlusion), compact settings controls, a prominent Run Restoration button. Below: a before/after image comparison with separate cards showing the actual processed input and restored output, inference time, and Download PNG. Include a sample-image affordance and concise model-resolution label 128 x 128. Professional research-lab aesthetic, clear hierarchy, warm off-white background with charcoal text and indigo/teal accents, Inter font, accessible contrast, restrained shadows, plenty of whitespace. The other tabs should visibly indicate that they support classifier probabilities and selected expert, four soft mixture weights, and photo-to-sketch styles 1/2/3 with upload/webcam respectively. Include graceful empty and model-unavailable states. Do not add login, chat, or payment flows."""


def main() -> None:
    project_file = OUT / "stitch_project.json"
    if project_file.exists():
        project = json.loads(project_file.read_text())
    else:
        project = call("create_project", {"title": "Restore & Sketch Lab - Generative AI Assignment 1"})
        project_file.write_text(json.dumps(project, indent=2))
    print("Stitch project created", flush=True)
    (OUT / "dashboard_prompt.txt").write_text(PROMPT, encoding="utf-8")
    if not (OUT / "dashboard_response.json").exists():
        project_name = project.get("structuredContent", {}).get("name")
        if not project_name:
            text = " ".join(block.get("text", "") for block in project.get("content", []) if block.get("type") == "text")
            import re
            match = re.search(r"projects/(\d+)", text)
            if not match:
                raise RuntimeError("Could not identify Stitch project ID")
            project_name = match.group(0)
        project_id = project_name.split("/")[-1]
        result = call("generate_screen_from_text", {"projectId": project_id,
                  "deviceType": "DESKTOP", "prompt": PROMPT})
        (OUT / "dashboard_response.json").write_text(json.dumps(result, indent=2))
        print("Stitch dashboard screen generated", flush=True)


if __name__ == "__main__":
    main()
