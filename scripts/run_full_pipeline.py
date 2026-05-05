from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def run(script: str) -> None:
    print(f"Running {script}...")
    subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / script)], check=True)


def main() -> None:
    run("generate_tyre_data.py")
    run("analyze_tyre_data.py")
    run("test_deep_learning.py")
    run("create_snapshot.py")
    print("Full TyreWear Intelligence pipeline completed.")


if __name__ == "__main__":
    main()
