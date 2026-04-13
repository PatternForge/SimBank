"""
SimBank V4 - Drift Detector
Compares current Snowflake RAW tables against baseline CSVs.
Computes hashes and field manifests.
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

SOURCES_DIR = Path("SimBank/sources")
OUTPUT_DIR = Path("SimBank/Output")
OUTPUT_DIR.mkdir(exist_ok=True)

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database": "SIMBANK",
    "schema": "RAW",
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
}

RAW_TABLES = [
    "RETAIL_DEPOSITS",
    "RETAIL_LOANS",
    "BUSINESS_DEPOSITS",
    "BUSINESS_LOANS",
]

logger = None


def setup_logger():
    global logger
    import logging

    logger = logging.getLogger("drift_detector")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_baseline_csv(table_name: str) -> pd.DataFrame:
    csv_path = SOURCES_DIR / f"{table_name.lower()}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {csv_path}")
    logger.info(f"Loading baseline: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def fetch_snowflake_data(table_name: str) -> pd.DataFrame:
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()

        logger.info(f"Fetching {table_name} from Snowflake...")
        cursor.execute(f"SELECT * FROM SIMBANK.RAW.{table_name}")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]

        cursor.close()
        conn.close()

        df = pd.DataFrame(rows, columns=cols)
        logger.info(f"  Fetched {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch {table_name}: {str(e)}")
        raise


def compute_row_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return hashlib.sha256("".encode()).hexdigest()
    df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    content = df_sorted.to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()


def get_field_manifest(df: pd.DataFrame) -> Dict:
    manifest = {
        "field_count": len(df.columns),
        "fields": [],
        "row_count": len(df),
    }
    for col in df.columns:
        manifest["fields"].append(
            {
                "name": col,
                "type": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
            }
        )
    return manifest


def detect_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, table_name: str) -> Dict:
    baseline_df.columns = [c.upper() for c in baseline_df.columns]
    current_df.columns = [c.upper() for c in current_df.columns]

    drift = {"table": table_name, "has_drift": False, "issues": []}

    baseline_rows = len(baseline_df)
    current_rows = len(current_df)
    variance = abs(current_rows - baseline_rows) / max(baseline_rows, 1)

    drift["baseline_rows"] = baseline_rows
    drift["current_rows"] = current_rows
    drift["row_variance"] = variance

    if variance > 0.20:
        drift["has_drift"] = True
        drift["issues"].append(
            f"Row count variance {variance:.1%} exceeds BLOCK threshold"
        )
    elif variance > 0.05:
        drift["issues"].append(f"Row count variance {variance:.1%} (ALERT)")

    baseline_fields = set(baseline_df.columns)
    current_fields = set(current_df.columns)

    added = current_fields - baseline_fields
    removed = baseline_fields - current_fields

    if added:
        drift["has_drift"] = True
        drift["issues"].append(f"Fields added: {', '.join(sorted(added))}")
    if removed:
        drift["has_drift"] = True
        drift["issues"].append(f"Fields removed: {', '.join(sorted(removed))}")

    for col in baseline_fields & current_fields:
        baseline_nulls = baseline_df[col].isna().sum()
        current_nulls = current_df[col].isna().sum()
        if current_nulls > 0 and baseline_nulls == 0:
            drift["issues"].append(f"Field {col}: nulls introduced ({current_nulls})")

    return drift


def generate_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def main():
    setup_logger()

    logger.info("=" * 60)
    logger.info("SimBank V4 - Drift Detector")
    logger.info("=" * 60)

    run_id = generate_run_id()
    logger.info(f"\nRun ID: {run_id}\n")

    logger.info("Checking baseline CSVs...")
    for table in RAW_TABLES:
        csv_path = SOURCES_DIR / f"{table.lower()}.csv"
        if not csv_path.exists():
            logger.error(f"Missing baseline: {csv_path}")
            sys.exit(1)
    logger.info("✓ All baseline CSVs found\n")

    logger.info("Loading baseline snapshots...")
    baselines = {t: load_baseline_csv(t) for t in RAW_TABLES}
    logger.info("")

    logger.info("Fetching current state from Snowflake...")
    current = {t: fetch_snowflake_data(t) for t in RAW_TABLES}
    logger.info("")

    logger.info("Running drift detection...\n")
    all_drift_results = []
    has_blocking_drift = False

    for table in RAW_TABLES:
        drift = detect_drift(baselines[table], current[table], table)
        all_drift_results.append(drift)

        status = "❌ DRIFT DETECTED" if drift["has_drift"] else "✓ STABLE"
        logger.info(f"{table}: {status}")
        logger.info(
            f"  Rows: {drift['baseline_rows']} → {drift['current_rows']} ({drift['row_variance']:.1%})"
        )
        for issue in drift["issues"]:
            logger.warning(f"  ⚠ {issue}")
        if drift["has_drift"]:
            has_blocking_drift = True

    logger.info("")
    logger.info("Computing hashes and field manifests...\n")

    hashes = {}
    manifests = {}
    for table in RAW_TABLES:
        hashes[table] = compute_row_hash(current[table])
        manifests[table] = get_field_manifest(current[table])
        logger.info(f"{table}: {hashes[table][:12]}...")

    logger.info("")

    combined_manifest = {
        "tables": manifests,
        "hashes": hashes,
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
    }
    combined_hash = hashlib.sha256(
        json.dumps(combined_manifest, sort_keys=True).encode()
    ).hexdigest()

    logger.info(f"Combined manifest hash: {combined_hash[:12]}...\n")

    results = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "manifest_hash": combined_hash,
        "has_drift": has_blocking_drift,
        "drift_results": all_drift_results,
        "manifests": manifests,
        "hashes": hashes,
    }

    results_path = OUTPUT_DIR / f"drift_results_{run_id}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results written to: {results_path}\n")
    logger.info("=" * 60)
    if has_blocking_drift:
        logger.warning("🔴 BLOCKING DRIFT DETECTED - Manual review required")
        logger.info("=" * 60)
        return 1
    else:
        logger.info("✅ NO DRIFT DETECTED - Ready for approval gate")
        logger.info("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
