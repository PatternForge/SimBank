"""
SimBank V4 - Docs Publisher (dbt 1.5+ compatible)
"""

import shutil
from pathlib import Path

# Repo root = SimBank/
REPO_ROOT = Path(__file__).resolve().parent

# dbt 1.5+ stores docs under: target/run/<project_name>/
DBT_DOCS_SOURCE = REPO_ROOT / "target"


# Published docs go here:
PUBLISH_ROOT = REPO_ROOT / "SimBank" / "docs"


def publish_docs(run_id: str) -> Path:
    """
    Copies dbt docs from the dbt target folder into the versioned docs folder.
    Returns the destination path.
    """
    dest = PUBLISH_ROOT / run_id
    dest.mkdir(parents=True, exist_ok=True)

    if not DBT_DOCS_SOURCE.exists():
        raise FileNotFoundError(f"dbt docs folder not found: {DBT_DOCS_SOURCE}")

    # Copy everything from dbt's docs output
    for item in DBT_DOCS_SOURCE.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    # Create/update a simple docs index
    index_path = PUBLISH_ROOT / "index.html"
    index_path.write_text(
        f"<html><body><h1>SimBank Docs</h1>"
        f"<p>Latest version: <a href='{run_id}/index.html'>{run_id}</a></p>"
        f"</body></html>"
    )

    return dest
