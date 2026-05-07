from pathlib import Path
from datetime import datetime
import shutil

def sync_latest_sources(logger):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    orchestrator_sources = REPO_ROOT / "Orchestrator" / "sources"
    target_sources = REPO_ROOT / "SimBank" / "sources"

    timestamped_folders = [
        d for d in orchestrator_sources.iterdir()
        if d.is_dir() and "_" in d.name
    ]

    def parse_ts(folder):
        return datetime.strptime(folder.name, "%Y-%m-%d_%H-%M-%S")

    latest_folder = max(timestamped_folders, key=parse_ts)
    target_sources.mkdir(parents=True, exist_ok=True)

    for file in latest_folder.glob("*.csv"):
        shutil.copy2(file, target_sources / file.name)

    logger(f"synced sources from {latest_folder.name}")

def run(state):
    logger = state.logger.info if hasattr(state, "logger") else print
    sync_latest_sources(logger)
    state.mark("sync_latest_sources", "success")
