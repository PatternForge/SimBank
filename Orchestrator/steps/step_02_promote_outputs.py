from pathlib import Path
import shutil


def run(state):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    output = REPO_ROOT / "SimBank" / "Output"

    if not output.exists():
        raise FileNotFoundError(f"Output folder not found: {output}")

    runs = [d for d in output.iterdir() if d.is_dir() and "_" in d.name]
    if not runs:
        raise RuntimeError("No run folders found in Output")

    latest_output = max(runs, key=lambda d: d.name)

    for f in latest_output.iterdir():
        if f.is_file():
            shutil.copy2(f, output / f.name)

    orchestrator_sources = REPO_ROOT / "Orchestrator" / "sources"
    target_sources = REPO_ROOT / "SimBank" / "sources"

    timestamped_folders = [
        d for d in orchestrator_sources.iterdir()
        if d.is_dir() and "_" in d.name
    ]

    if not timestamped_folders:
        raise RuntimeError("No timestamped source folders found in Orchestrator/sources")

    latest_sources = sorted(timestamped_folders)[-1]

    target_sources.mkdir(parents=True, exist_ok=True)

    for csv in latest_sources.glob("*.csv"):
        shutil.copy2(csv, target_sources / csv.name)

    state.mark("promote_outputs", "success", latest_output.name)
