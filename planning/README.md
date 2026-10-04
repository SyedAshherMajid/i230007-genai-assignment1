# Evidence supporting the execution plan

These files record the local inspection performed while preparing [EXECUTION_PLAN.md](../EXECUTION_PLAN.md) on 4 October 2026. They are planning evidence, not assignment results.

| File | What it records |
|---|---|
| `environment_inventory.json` | Relevant C:/D: directory scan and installed Python package metadata |
| `cached_packages.json` | Identified wheel metadata inside existing pip caches, including exact paths and Python/platform tags |
| `npm_cache_inventory.json` | Relevant package tarball/metadata entries in the existing npm cache |
| `hardware_and_tools.json` | Hardware, disk space, tool versions, WSL and recording-tool observations |
| `runtime_import_checks.json` | Actual imports in selected existing Python environments |
| `gpu_probe_results.json` | Actual GPU forward/backward operations and small synthetic timings |
| `audit_environment.py` | Windows read-only inventory script; writes its report beside itself |
| `gpu_probe.py` | Small synthetic GPU probe; writes its results beside itself |

The scan did not inspect all protected folders, archives with unrelated names, or Docker image contents. Package metadata, cached archives, successful imports, and a working training operation are different levels of evidence; the plan distinguishes them.

No software was installed, upgraded, or removed. No assignment datasets were downloaded. No user accounts or remote repositories were created. The short synthetic probe used random images, not the assignment datasets, and does not measure reconstruction quality or full training speed. WSL Ubuntu was started to inspect its environment.

Raw audit files include machine-specific paths. Keep them local or sanitize them before publishing the assignment repository. They do not replace the reproducibility metadata that must be generated during actual training.

To repeat the Windows scan later, from the project directory:

```powershell
python planning/audit_environment.py
```

To repeat the synthetic GPU check using the environment verified during this audit:

```powershell
& 'D:\Assesment\.venv310\Scripts\python.exe' planning/gpu_probe.py
```

Repeating either command updates its corresponding local JSON report. Other evidence files are snapshots of the additional targeted checks performed during planning.
