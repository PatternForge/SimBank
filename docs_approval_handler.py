"""
SimBank V4 - Docs Approval Handler (Rewritten + Instrumented)
"""

import argparse
import json
import os
from pathlib import Path

import snowflake.connector
from snowflake.connector import Binary
from dotenv import load_dotenv

from docs_publisher import publish_docs

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "SimBank" / "Output"
BASELINE_DIR = REPO_ROOT / "SimBank" / "Baselines"
ARTIFACT_ZIP_PATH = OUTPUT_DIR / "docs_artifacts.zip"

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": "SIMBANK_WH",   # ← FIXED + EXPLICIT
    "database": "SIMBANK",
    "schema": "GOVERNANCE",
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
}


def load_drift(run_id: str):
    path = OUTPUT_DIR / f"docs_drift_results_{run_id}.json"
    print(f"[DEBUG] Loading drift file: {path}")
    return json.loads(path.read_text())


def write_baseline_local(run_id: str, drift_data: dict):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASELINE_DIR / f"docs_baseline_{run_id}.json"
    print(f"[DEBUG] Writing local baseline: {out_path}")
    out_path.write_text(json.dumps(drift_data, indent=2))


def write_baseline_snowflake(run_id: str, drift_data: dict):
    print("[DEBUG] Preparing to write baseline to Snowflake...")

    if not ARTIFACT_ZIP_PATH.exists():
        raise FileNotFoundError(f"Artifact ZIP not found: {ARTIFACT_ZIP_PATH}")

    artifact_zip = Binary(ARTIFACT_ZIP_PATH.read_bytes())
    docs_hash = drift_data["docs_hash"]

    print(f"[DEBUG] Connecting to Snowflake with warehouse={SNOWFLAKE_CONFIG['warehouse']}")

    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()

    cursor.execute("USE WAREHOUSE SIMBANK_WH")
    cursor.execute("USE DATABASE SIMBANK")
    cursor.execute("USE SCHEMA GOVERNANCE")

    print("[DEBUG] Executing INSERT...")
    cursor.execute("""
        INSERT INTO SIMBANK.GOVERNANCE.DOCS_BASELINES
        (RUN_ID, DOCS_HASH, ARTIFACT_ZIP, STATUS, APPROVED_BY, APPROVED_AT)
        VALUES (%s, %s, %s, 'APPROVED', CURRENT_USER(), CURRENT_TIMESTAMP())
    """, (run_id, docs_hash, artifact_zip))

    conn.commit()
    print("[DEBUG] INSERT committed.")

    cursor.close()
    conn.close()
    print("[DEBUG] Snowflake connection closed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()

    print(f"[DEBUG] ARGS: {args}")

    drift_data = load_drift(args.run_id)

    if not args.approve:
        print("Docs review posted to Slack.")
        print("[DEBUG] Exiting early because --approve was NOT passed.")
        return

    print("[DEBUG] Calling publish_docs()...")
    publish_docs(args.run_id)
    print("[DEBUG] publish_docs() completed.")

    print("[DEBUG] Calling write_baseline_snowflake()...")
    write_baseline_snowflake(args.run_id, drift_data)

    print("[DEBUG] Calling write_baseline_local()...")
    write_baseline_local(args.run_id, drift_data)

    print(f"✓ Docs published and baseline written for run {args.run_id}")


if __name__ == "__main__":
    main()
