from pathlib import Path
import shutil


def run(state):
    REPO_ROOT = Path(__file__).resolve().parents[2]

    output = REPO_ROOT / "SimBank" / "Output"

    if not output.exists():
        raise FileNotFoundError(f"Output folder not found: {output}")

    runs = [d for d in output.iterdir() if d.is_dir()]
    if not runs:
        raise RuntimeError("No run folders found in Output")

    latest = max(runs, key=lambda d: d.name)

    for f in latest.iterdir():
        if f.is_file():
            shutil.copy2(f, output / f.name)

    state.mark("promote_outputs", "success", latest.name)
