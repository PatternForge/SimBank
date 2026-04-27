import subprocess
import sys
from pathlib import Path


def run(state):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    GOV_SCRIPT = REPO_ROOT / "Governance" / "run_v4.py"

    if not GOV_SCRIPT.exists():
        raise FileNotFoundError(f"Governance script not found: {GOV_SCRIPT}")

    subprocess.run(
        [
            sys.executable,
            str(GOV_SCRIPT),
            "--run-id",
            state.run_id,
            "--caller",
            "platform",
        ],
        check=True
    )

    state.mark("drift", "success")
