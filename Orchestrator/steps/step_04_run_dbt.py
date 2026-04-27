import subprocess
from pathlib import Path


def run(state):
    REPO_ROOT = Path(__file__).resolve().parents[2]

    DBT_PROJECT = REPO_ROOT / "simbank_dbt"

    if not DBT_PROJECT.exists():
        raise FileNotFoundError(f"dbt project not found: {DBT_PROJECT}")

    subprocess.run(
        ["dbt", "run", "--project-dir", str(DBT_PROJECT)],
        check=True
    )
    subprocess.run(
        ["dbt", "test", "--project-dir", str(DBT_PROJECT)],
        check=True
    )

    state.mark("dbt", "success")
