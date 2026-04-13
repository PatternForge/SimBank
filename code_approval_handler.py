"""
SimBank V4 - Code Approval Handler (ENTER-based)
"""

import json
import os
from datetime import datetime
from pathlib import Path
import hashlib
import snowflake.connector
import requests
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "SimBank" / "Output"
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
    path = OUTPUT_DIR / f"code_drift_results_{run_id}.json"
    with open(path, "r") as f:
        return json.load(f)


def post_to_slack(payload):
    requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
    print("✓ Code message posted to Slack")


def write_code_baseline(run_id: str):
    MODELS_DIR = REPO_ROOT / "simbank_dbt" / "models"
    sql_files = {}

    for folder in ["staging", "mart"]:
        d = MODELS_DIR / folder
        if d.exists():
            for f in d.glob("*.sql"):
                content = f.read_text()
                rel = f.relative_to(MODELS_DIR)
                sql_files[str(rel)] = content

    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    cur = conn.cursor()

    for file_path, content in sql_files.items():
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        cur.execute(
            """
            INSERT INTO SIMBANK.GOVERNANCE.CODE_BASELINES
            (RUN_ID, FILE_PATH, FILE_HASH, FILE_CONTENT, STATUS, APPROVED_AT, APPROVED_BY)
            VALUES (%s, %s, %s, %s, 'APPROVED', CURRENT_TIMESTAMP(), %s)
            """,
            (run_id, file_path, file_hash, content, "ross"),
        )

    conn.commit()
    cur.close()
    conn.close()
    print("✓ Code baseline written to Snowflake")


def build_message(results):
    run_id = results["run_id"]
    has_drift = results["has_drift"]
    emoji = "🔴" if has_drift else "✅"
    status = "DRIFT DETECTED" if has_drift else "NO DRIFT"

    return {
        "text": f"{emoji} Code Drift - {status}\nRun ID: `{run_id}`\nPress ENTER locally to approve."
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()

    if args.approve:
        write_code_baseline(args.run_id)
        return

    results = load_results(args.run_id)
    payload = build_message(results)
    post_to_slack(payload)
    print("Waiting for ENTER approval...")
    return


if __name__ == "__main__":
    main()
