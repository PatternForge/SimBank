"""
SimBank V4 - Data Approval Handler (ENTER-based)
"""

import json
import os
from datetime import datetime
from pathlib import Path
import snowflake.connector
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Resolve repo root (Governance/approvals/slack_approval_handler.py → repo root)
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = REPO_ROOT / "SimBank" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": "COMPUTE_WH",
    "database": "SIMBANK",
    "schema": "GOVERNANCE",
    "role": "ACCOUNTADMIN",
}


def load_results(run_id: str):
    path = OUTPUT_DIR / f"drift_results_{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Drift results not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def post_to_slack(payload):
    if not SLACK_WEBHOOK_URL:
        print("⚠ SLACK_WEBHOOK_URL not set — skipping Slack post")
        return
    requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
    print("✓ Data message posted to Slack")


def write_attestation(run_id: str, manifest_hash: str):
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO SIMBANK.GOVERNANCE.DATA_ATTESTATIONS
        (RUN_ID, MANIFEST_HASH, STATUS, APPROVED_AT, APPROVED_BY)
        VALUES (%s, %s, 'APPROVED', CURRENT_TIMESTAMP(), %s)
        """,
        (run_id, manifest_hash, "ross"),
    )

    conn.commit()
    cur.close()
    conn.close()
    print("✓ Data attestation written to Snowflake")


def build_message(results):
    run_id = results["run_id"]
    manifest_hash = results["manifest_hash"][:12]
    return {
        "text": (
            f"📊 Data Drift Check\n"
            f"Run ID: `{run_id}`\n"
            f"Manifest: `{manifest_hash}...`\n"
            f"Press ENTER locally to approve."
        )
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()

    results = load_results(args.run_id)

    if args.approve:
        write_attestation(args.run_id, results["manifest_hash"])
        return

    payload = build_message(results)
    post_to_slack(payload)
    print("Waiting for ENTER approval...")
    return


if __name__ == "__main__":
    main()
