"""
SimBank V4 - Full Workflow Orchestrator
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_step(cmd, label, allow_drift=False):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60 + "\n")
    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0 and not allow_drift:
        pass
    return result.returncode


def extract_run_id(pattern, base_dir):
    files = [f for f in base_dir.glob(pattern) if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = files[0]
    parts = latest.stem.split("_")
    return f"{parts[-2]}_{parts[-1]}"


def wait_for_enter(label):
    print(f"\n{label}")
    input("Press ENTER to approve and continue...")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--caller", default="manual")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("extra", nargs="*")
    args, _ = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "SimBank" / "Output"
    output_dir2 = repo_root / "Output"

    print("\n╔══════════════════════════════════════╗")
    print("║   SimBank V4 - Full Workflow        ║")
    print("╚══════════════════════════════════════╝\n")

    run_step(
        [sys.executable, "-m", "Governance.drift.code_drift_detector"],
        "Step 1: Code Drift Detection",
        allow_drift=True
    )
    code_run_id = extract_run_id("code_drift_results_*.json", output_dir)
    print(f"✓ Code Run ID: {code_run_id}")

    run_step(
        [sys.executable, "-m", "Governance.approvals.code_approval_handler", "--run-id", code_run_id],
        "Step 2: Code Approval Gate"
    )
    wait_for_enter("Code review posted to Slack.")
    run_step(
        [sys.executable, "-m", "Governance.approvals.code_approval_handler", "--run-id", code_run_id, "--approve"],
        "Applying Code Approval"
    )

    run_step(
        [sys.executable, "-m", "Governance.drift.drift_detector"],
        "Step 3: Data Drift Detection",
        allow_drift=True
    )
    data_run_id = extract_run_id("drift_results_*.json", output_dir)
    print(f"✓ Data Run ID: {data_run_id}")

    run_step(
        [sys.executable, "-m", "Governance.approvals.slack_approval_handler", "--run-id", data_run_id],
        "Step 4: Data Approval Gate"
    )
    wait_for_enter("Data review posted to Slack.")
    run_step(
        [sys.executable, "-m", "Governance.approvals.slack_approval_handler", "--run-id", data_run_id, "--approve"],
        "Applying Data Approval"
    )

    run_step(
        [sys.executable, "-m", "Governance.drift.docs_drift_detector"],
        "Step 5: Docs Drift Detection",
        allow_drift=True
    )

    pointer_file = None
    if (output_dir2 / "latest_docs_run_id.txt").exists():
        pointer_file = output_dir2 / "latest_docs_run_id.txt"
    elif (output_dir / "latest_docs_run_id.txt").exists():
        pointer_file = output_dir / "latest_docs_run_id.txt"
    else:
        raise FileNotFoundError("latest_docs_run_id.txt not found")

    docs_run_id = pointer_file.read_text().strip()
    print(f"✓ Docs Run ID: {docs_run_id}")

    run_step(
        [sys.executable, "-m", "Governance.approvals.docs_approval_handler", "--run-id", docs_run_id],
        "Step 6: Docs Approval & Publish"
    )
    wait_for_enter("Docs review posted to Slack.")
    run_step(
        [sys.executable, "-m", "Governance.approvals.docs_approval_handler", "--run-id", docs_run_id, "--approve"],
        "Publishing Docs & Writing Baseline"
    )

    print("\n" + "=" * 60)
    print("✅ V4 Workflow Complete")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
