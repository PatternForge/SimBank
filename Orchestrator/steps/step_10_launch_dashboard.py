import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv


def run(state):

    REPO_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(REPO_ROOT / ".env")

    if os.getenv("SIMBANK_CI", "false").lower() == "true":
        state.mark("dashboard_launch", "skipped", "CI mode")
        return

    if os.getenv("SIMBANK_SKIP_DASHBOARD", "false").lower() == "true":
        state.mark("dashboard_launch", "skipped", "User disabled")
        return

    dashboard_app = REPO_ROOT / "Dashboard" / "app.py"

    if not dashboard_app.exists():
        state.mark("dashboard_launch", "error", f"Dashboard app not found: {dashboard_app}")
        return

    try:
        subprocess.Popen(
            [
                "streamlit",
                "run",
                str(dashboard_app)
            ],
            cwd=str(REPO_ROOT / "Dashboard")
        )

        state.mark("dashboard_launch", "success", "Dashboard launched")

    except Exception as e:
        state.mark("dashboard_launch", "error", str(e))
