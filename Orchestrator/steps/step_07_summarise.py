import json


def run(state):
    print(json.dumps({
        "run_id": state.run_id,
        "started_at": state.started_at.isoformat(),
        "steps": state.steps,
        "errors": state.errors,
    }, indent=2))

