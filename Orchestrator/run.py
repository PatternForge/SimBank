from Orchestrator.common.env import validate_env
from Orchestrator.models.run_state import RunState
from Orchestrator.steps import (
    step_00_bootstrap_governance,
    step_01_python_SoT,
    step_02_promote_outputs,
    step_03_check_snowflake_env,
    step_04_run_dbt,
    step_05_run_lineage,
    step_06_run_drift,
    step_07_summarise,
    step_08_render_governance_graphs,
    step_09_build_dashboard_data,
    step_10_launch_dashboard
)


def run():
    validate_env()
    state = RunState()
    try:
        step_00_bootstrap_governance.run(state)
        step_01_python_SoT.run(state)
        step_02_promote_outputs.run(state)
        step_03_check_snowflake_env.run(state)
        step_04_run_dbt.run(state)
        step_05_run_lineage.run(state)
        step_06_run_drift.run(state)
        step_08_render_governance_graphs.run(state)
        step_09_build_dashboard_data.run(state)
        step_10_launch_dashboard.run(state)
    except Exception as e:
        state.fail(e)
    finally:
        step_07_summarise.run(state)


if __name__ == "__main__":
    run()