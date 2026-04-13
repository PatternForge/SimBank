"""
SimBank V4 - Docs Drift Detector (Final Version)
- Only detect docs drift when code drift has occurred
- Only include YAML-described columns (Option A)
- Ignore all dbt/Snowflake noise
"""

import hashlib
import json
import os
import sys
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent

DBT_PROJECT_DIR = REPO_ROOT / "simbank_dbt"
DBT_TARGET = DBT_PROJECT_DIR / "target"

OUTPUT_DIR = REPO_ROOT / "SimBank" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# New docs pointer location
OUTPUT_DIR2 = REPO_ROOT / "Output"
OUTPUT_DIR2.mkdir(parents=True, exist_ok=True)

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": "SIMBANK_WH",
    "database": "SIMBANK",
    "schema": "GOVERNANCE",
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
}

ARTIFACT_FILES = [
    "catalog.json",
    "semantic_manifest.json",
]


def code_is_unchanged() -> bool:
    latest = OUTPUT_DIR / "code_drift_results_latest.json"
    if not latest.exists():
        return False

    try:
        data = json.loads(latest.read_text())
        return data.get("has_drift") is False
    except Exception:
        return False


def generate_docs():
    print("Running: dbt docs generate...")
    result = subprocess.run(
        ["dbt", "docs", "generate"],
        cwd=str(DBT_PROJECT_DIR),
        shell=True,
    )
    if result.returncode != 0:
        print("✗ Failed to generate dbt docs")
        sys.exit(1)
    print("✓ dbt docs generated")


def zip_artifacts() -> bytes:
    zip_path = OUTPUT_DIR / "docs_artifacts.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in sorted(ARTIFACT_FILES):
            fpath = DBT_TARGET / fname
            if fpath.exists():
                info = zipfile.ZipInfo(fname)
                info.date_time = (2020, 1, 1, 0, 0, 0)
                with open(fpath, "rb") as f:
                    z.writestr(info, f.read())

    return zip_path.read_bytes()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_catalog(catalog: dict) -> dict:
    nodes = catalog.get("nodes", {})
    sources = catalog.get("sources", {})

    def clean_node(node):
        described_columns = {
            col_name: {
                "name": col_name,
                "description": col_data.get("description"),
            }
            for col_name, col_data in node.get("columns", {}).items()
            if col_data.get("description") not in (None, "", " ")
        }

        return {
            "name": node.get("name"),
            "description": node.get("description"),
            "columns": dict(sorted(described_columns.items())),
        }

    cleaned = {
        "nodes": {
            name: clean_node(node)
            for name, node in sorted(nodes.items())
        },
        "sources": {
            name: clean_node(src)
            for name, src in sorted(sources.items())
        },
    }

    return cleaned


def normalize_semantic_manifest(manifest: dict) -> dict:
    return {
        "metrics": manifest.get("metrics", {}),
        "entities": manifest.get("entities", {}),
        "dimensions": manifest.get("dimensions", {}),
        "measures": manifest.get("measures", {}),
    }


def deterministic_hash(obj) -> str:
    normalized = json.dumps(obj, sort_keys=True, indent=None, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_meaningful_docs_hash() -> str:
    catalog_path = DBT_TARGET / "catalog.json"
    semantic_path = DBT_TARGET / "semantic_manifest.json"

    catalog = load_json(catalog_path)
    semantic = load_json(semantic_path)

    cleaned_catalog = normalize_catalog(catalog)
    cleaned_semantic = normalize_semantic_manifest(semantic)

    combined = {
        "catalog": cleaned_catalog,
        "semantic": cleaned_semantic,
    }

    return deterministic_hash(combined)


def fetch_baseline():
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()

    cursor.execute("USE WAREHOUSE SIMBANK_WH")
    cursor.execute("USE DATABASE SIMBANK")
    cursor.execute("USE SCHEMA GOVERNANCE")

    cursor.execute("""
        SELECT RUN_ID, DOCS_HASH, ARTIFACT_ZIP
        FROM SIMBANK.GOVERNANCE.DOCS_BASELINES
        WHERE STATUS = 'APPROVED'
        ORDER BY APPROVED_AT DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return None

    return {"run_id": row[0], "hash": row[1], "zip": row[2]}


def generate_run_id():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def main():
    print("\n" + "=" * 60)
    print("SimBank V4 - Docs Drift Detector")
    print("=" * 60)

    run_id = generate_run_id()
    print(f"\nRun ID: {run_id}\n")

    (OUTPUT_DIR2 / "latest_docs_run_id.txt").write_text(run_id)

    if code_is_unchanged():
        print("✓ NO DOCS DRIFT DETECTED (code unchanged)")
        results = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "has_drift": False,
            "docs_hash": None,
            "bootstrap_mode": False,
        }
        results_path = OUTPUT_DIR / f"docs_drift_results_{run_id}.json"
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to: {results_path}\n")
        return 0

    generate_docs()
    zip_artifacts()

    docs_hash = compute_meaningful_docs_hash()

    baseline = fetch_baseline()
    bootstrap_mode = baseline is None

    if bootstrap_mode:
        print("No docs baseline found — BOOTSTRAP MODE\n")
        drift_detected = True
    else:
        print(f"Baseline RUN_ID: {baseline['run_id']}")
        print(f"Baseline HASH:   {baseline['hash']}")
        print(f"Current HASH:    {docs_hash}")
        drift_detected = docs_hash != baseline["hash"]

    if drift_detected:
        print("🔴 DOCS DRIFT DETECTED")
    else:
        print("✓ NO DOCS DRIFT DETECTED")

    results = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "has_drift": drift_detected,
        "docs_hash": docs_hash,
        "bootstrap_mode": bootstrap_mode,
    }

    results_path = OUTPUT_DIR / f"docs_drift_results_{run_id}.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\nResults written to: {results_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
