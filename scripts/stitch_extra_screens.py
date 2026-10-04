"""Generate companion Stitch screens with the dashboard's design system."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.request

from stitch_design import call, OUT

dashboard = json.loads((OUT / "dashboard_response.json").read_text())
screen = dashboard["structuredContent"]["outputComponents"][0]["design"]["screens"][0]
project_id = dashboard["structuredContent"]["projectId"]
design_system = screen["designSystem"]["name"]

PROMPTS = {
    "hard_routing": "Use the same visual design system as Restore & Sketch Lab. Design a desktop workspace for Hard-Routed Restoration. Sidebar with four workspaces. Main area has an image upload card, optional corruption settings, and a Run button. Results have true input and output images, four classifier probability bars labeled Clean, Noise, Blur, Occlusion, a clearly highlighted predicted class and selected specialist (or identity bypass for clean), inference time and PNG download. Show an honest empty state with no fake probabilities or fabricated quality metrics. Warm off-white, indigo and teal, Inter, accessible and polished.",
    "soft_mixture": "Use the same visual design system as Restore & Sketch Lab. Design a desktop workspace for Soft Mixture-of-Experts Restoration. Sidebar with four workspaces. Main area has image upload, optional synthetic corruption controls, and a Run button. Results show actual processed input and restored output. Include four prominent contribution bars labeled Identity, Noise Expert, Blur Expert, Occlusion Expert with percentages that sum to 100 only when a real result exists. In empty state show labels without fake numerical results. Include inference time and PNG download. Warm off-white, indigo and teal, Inter, accessible and polished.",
    "face_sketch": "Use the same visual design system as Restore & Sketch Lab. Design a desktop workspace for Face-to-Sketch Generator. Sidebar with four workspaces. Main area lets the user upload a portrait or capture it with webcam permission, choose Style 1, 2, or 3 with clear visual selection, and press Generate Sketch. Results display original photo and generated sketch side by side plus Download PNG and elapsed inference time. Include helpful empty state, camera-denied fallback to upload, and model-unavailable status. Never show fake outputs or invented quality scores. Warm off-white, indigo and teal, Inter, accessible and polished."
}


def main() -> None:
    for slug, prompt in PROMPTS.items():
        (OUT / f"{slug}_prompt.txt").write_text(prompt, encoding="utf-8")
        output = OUT / f"{slug}_response.json"
        if output.exists():
            continue
        result = call("generate_screen_from_text", {"projectId": project_id, "prompt": prompt,
                      "deviceType": "DESKTOP", "designSystem": design_system})
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        generated = result["structuredContent"]["outputComponents"][0]["design"]["screens"][0]
        for key, suffix in (("screenshot", "png"), ("htmlCode", "html")):
            if key in generated:
                with urllib.request.urlopen(generated[key]["downloadUrl"], timeout=60) as response:
                    (OUT / f"{slug}.{suffix}").write_bytes(response.read())
        print(f"Stitch {slug} screen saved", flush=True)


if __name__ == "__main__":
    main()
