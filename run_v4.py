"""
SimBank V4 - Full Workflow Orchestrator
"""

import subprocess
import sys
from pathlib import Path


def run_step(cmd: str, label: str):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60 + "\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def extract_run_id(pattern: str, base_dir: Path) -> str:
    files = [f for f in base_dir.glob(pattern) if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = files[0]
    parts = latest.stem.split("_")
    return f"{parts[-2]}_{parts[-1]}"


def wait_for_enter(label: str):
    print(f"\n{label}")
    input("Press ENTER to approve and continue...")


def main():
    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "SimBank" / "Output"
    output_dir2 = repo_root / "Output"


    print("\n╔══════════════════════════════════════╗")
    print("║   SimBank V4 - Full Workflow        ║")
    print("╚══════════════════════════════════════╝\n")

    # Step 1: Code Drift
    run_step("python code_drift_detector.py", "Step 1: Code Drift Detection")
    code_run_id = extract_run_id("code_drift_results_*.json", output_dir)
    print(f"✓ Code Run ID: {code_run_id}")

    run_step(f"python code_approval_handler.py --run-id {code_run_id}", "Step 2: Code Approval Gate")
    wait_for_enter("Code review posted to Slack.")
    run_step(f"python code_approval_handler.py --run-id {code_run_id} --approve", "Applying Code Approval")

    # Step 3: Data Drift
    run_step("python drift_detector.py", "Step 3: Data Drift Detection")
    data_run_id = extract_run_id("drift_results_*.json", output_dir)
    print(f"✓ Data Run ID: {data_run_id}")

    run_step(f"python slack_approval_handler.py --run-id {data_run_id}", "Step 4: Data Approval Gate")
    wait_for_enter("Data review posted to Slack.")
    run_step(f"python slack_approval_handler.py --run-id {data_run_id} --approve", "Applying Data Approval")

    # Step 5: Docs Drift
    run_step("python docs_drift_detector.py", "Step 5: Docs Drift Detection")
    pointer_file = None

    if (output_dir2 / "latest_docs_run_id.txt").exists():
        pointer_file = output_dir2 / "latest_docs_run_id.txt"
    elif (output_dir / "latest_docs_run_id.txt").exists():
        pointer_file = output_dir / "latest_docs_run_id.txt"
    else:
        raise FileNotFoundError("latest_docs_run_id.txt not found in either output directory")

    docs_run_id = pointer_file.read_text().strip()
    print(f"✓ Docs Run ID: {docs_run_id}")

    run_step(f"python docs_approval_handler.py --run-id {docs_run_id}", "Step 6: Docs Approval & Publish")
    wait_for_enter("Docs review posted to Slack.")
    run_step(f"python docs_approval_handler.py --run-id {docs_run_id} --approve", "Publishing Docs & Writing Baseline")

    print("\n" + "=" * 60)
    print("✅ V4 Workflow Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
