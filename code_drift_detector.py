"""
SimBank V4 - Code Drift Detector
Compares current dbt SQL files against last approved baseline in Snowflake.
Detects: file changes, new files, deleted files.
"""

import hashlib
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

# Paths (repo root)
MODELS_DIR = Path("simbank_dbt/models")
STAGING_DIR = MODELS_DIR / "staging"
MART_DIR = MODELS_DIR / "mart"
OUTPUT_DIR = Path("SimBank/Output")
OUTPUT_DIR.mkdir(exist_ok=True)

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database": "SIMBANK",
    "schema": "GOVERNANCE",
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
}


def collect_sql_files() -> Dict[str, str]:
    sql_files = {}
    for directory in [STAGING_DIR, MART_DIR]:
        if not directory.exists():
            continue
        for sql_file in directory.glob("*.sql"):
            with open(sql_file, "r", encoding="utf-8") as f:
                content = f.read()
            rel_path = sql_file.relative_to(MODELS_DIR)
            sql_files[str(rel_path)] = content
    return sql_files


def compute_file_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def fetch_baseline_from_snowflake() -> Dict[str, Dict]:
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT RUN_ID
            FROM SIMBANK.GOVERNANCE.CODE_BASELINES
            WHERE STATUS = 'APPROVED'
            ORDER BY APPROVED_AT DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            return {}

        latest_run_id = row[0]

        cursor.execute(
            """
            SELECT FILE_PATH, FILE_HASH, FILE_CONTENT, APPROVED_AT
            FROM SIMBANK.GOVERNANCE.CODE_BASELINES
            WHERE RUN_ID = %s
            """,
            (latest_run_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        baseline = {}
        for file_path, file_hash, file_content, approved_at in rows:
            baseline[file_path] = {
                "hash": file_hash,
                "content": file_content,
                "approved_at": approved_at,
                "run_id": latest_run_id,
            }
        return baseline

    except Exception as e:
        print(f"Warning: Could not fetch baseline from Snowflake: {e}")
        return {}


def detect_code_drift(current_files: Dict[str, str], baseline: Dict[str, Dict]) -> Dict:
    drift = {
        "has_drift": False,
        "changes": {"modified": [], "added": [], "deleted": []},
        "current_files": {},
        "baseline_files": baseline,
    }

    for file_path, content in current_files.items():
        current_hash = compute_file_hash(content)
        drift["current_files"][file_path] = {"hash": current_hash, "content": content}

        if file_path in baseline:
            baseline_hash = baseline[file_path]["hash"]
            if current_hash != baseline_hash:
                drift["has_drift"] = True
                drift["changes"]["modified"].append(
                    {
                        "file": file_path,
                        "baseline_hash": baseline_hash,
                        "current_hash": current_hash,
                    }
                )
        else:
            drift["has_drift"] = True
            drift["changes"]["added"].append(
                {"file": file_path, "hash": current_hash}
            )

    for file_path in baseline:
        if file_path not in current_files:
            drift["has_drift"] = True
            drift["changes"]["deleted"].append(
                {"file": file_path, "baseline_hash": baseline[file_path]["hash"]}
            )

    return drift


def generate_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def main():
    print("\n" + "=" * 60)
    print("SimBank V4 - Code Drift Detector")
    print("=" * 60)

    run_id = generate_run_id()
    print(f"\nRun ID: {run_id}\n")

    print("Collecting current SQL files...")
    current_files = collect_sql_files()
    print(f"Found {len(current_files)} SQL files\n")

    if not current_files:
        print("Error: No SQL files found in simbank_dbt/models/staging or mart")
        sys.exit(1)

    print("Fetching last approved baseline from Snowflake...")
    baseline = fetch_baseline_from_snowflake()

    bootstrap_mode = not bool(baseline)
    if baseline:
        print(f"Found baseline with {len(baseline)} files\n")
    else:
        print("No baseline found — entering BOOTSTRAP MODE (first run)\n")

    print("Detecting code drift...\n")
    drift = detect_code_drift(current_files, baseline)

    if drift["has_drift"]:
        print("🔴 CODE DRIFT DETECTED\n")

        if drift["changes"]["modified"]:
            print(f"Modified files ({len(drift['changes']['modified'])}):")
            for change in drift["changes"]["modified"]:
                print(f"  • {change['file']}")
                print(f"    Baseline: {change['baseline_hash'][:12]}...")
                print(f"    Current:  {change['current_hash'][:12]}...\n")

        if drift["changes"]["added"]:
            print(f"Added files ({len(drift['changes']['added'])}):")
            for change in drift["changes"]["added"]:
                print(f"  • {change['file']} ({change['hash'][:12]}...)\n")

        if drift["changes"]["deleted"]:
            print(f"Deleted files ({len(drift['changes']['deleted'])}):")
            for change in drift["changes"]["deleted"]:
                print(
                    f"  • {change['file']} (was {change['baseline_hash'][:12]}...)\n"
                )
    else:
        print("✅ NO CODE DRIFT DETECTED\n")

    if bootstrap_mode:
        print("BOOTSTRAP MODE: Allowing drift and continuing workflow.\n")

    results = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "has_drift": drift["has_drift"],
        "changes": drift["changes"],
        "current_file_count": len(current_files),
        "baseline_file_count": len(baseline),
        "bootstrap_mode": bootstrap_mode,
    }

    results_path = OUTPUT_DIR / f"code_drift_results_{run_id}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to: {results_path}\n")

    return 0 if bootstrap_mode else (1 if drift["has_drift"] else 0)


if __name__ == "__main__":
    sys.exit(main())
