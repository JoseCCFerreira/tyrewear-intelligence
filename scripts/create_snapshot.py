from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT_DIR / "snapshots"
JSON_PATH = SNAPSHOT_DIR / "project_snapshot.json"
MD_PATH = SNAPSHOT_DIR / "project_snapshot.md"


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT_DIR, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    tracked_files = [
        "index.html",
        "style.css",
        "technical_support.html",
        "README.md",
        "requirements.txt",
        "app.py",
        "scripts/generate_tyre_data.py",
        "scripts/analyze_tyre_data.py",
        "scripts/run_full_pipeline.py",
        "scripts/test_deep_learning.py",
        "scripts/create_snapshot.py",
        "data/raw/tyrewear_europe_raw.csv",
        "data/processed/tyrewear_europe_clean.csv",
        "data/processed/tyrewear_europe_monthly_resampled.csv",
        "data/processed/tyrewear_europe_annual_country.csv",
        "data/outputs/hypothesis_tests.csv",
        "data/outputs/analysis_summary.json",
        "data/outputs/tyre_clusters.csv",
        "data/outputs/cluster_summary.csv",
    ]
    deep_learning_path = SNAPSHOT_DIR / "deep_learning_test_results.json"
    deep_learning_results = {}
    if deep_learning_path.exists():
        deep_learning_results = json.loads(deep_learning_path.read_text(encoding="utf-8"))

    snapshot = {
        "project": "TyreWear Intelligence",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repository": {
            "git_status": run(["git", "status", "--short"]),
            "latest_commit": run(["git", "log", "-1", "--oneline"]),
            "remote": run(["git", "remote", "-v"]),
        },
        "files": [
            {"path": path, "size_bytes": file_size(ROOT_DIR / path)}
            for path in tracked_files
        ],
        "deep_learning": deep_learning_results,
    }

    JSON_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    dl_tests = deep_learning_results.get("tests", [])
    dl_lines = "\n".join(
        f"- {item.get('framework')}: {item.get('status')} | MSE {item.get('final_mse')} | version {item.get('version')}"
        for item in dl_tests
    ) or "- No deep learning test results found."
    md = f"""# TyreWear Intelligence Snapshot

Generated at: `{snapshot['created_at']}`

## Repository

- Latest commit: `{snapshot['repository']['latest_commit'] or 'not committed yet'}`
- Remote: `{snapshot['repository']['remote'] or 'no remote configured'}`

## Files

| File | Size |
|---|---:|
"""
    for item in snapshot["files"]:
        md += f"| `{item['path']}` | {item['size_bytes']} bytes |\n"
    md += f"""

## Deep Learning Smoke Tests

{dl_lines}
"""
    MD_PATH.write_text(md, encoding="utf-8")
    print(JSON_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
