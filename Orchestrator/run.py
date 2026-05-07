from Orchestrator.common.env import validate_env
from Orchestrator.models.run_state import RunState

from Orchestrator.steps import (
    step_00_bootstrap_governance,
    step_01_python_SoT,
    step_02_promote_outputs,
    step_02b_sync_latest_sources,
    step_02c_load_raw,
    step_03_check_snowflake_env,
    step_04_run_dbt,
    step_05_run_lineage,
    step_06_run_drift,
    step_07_summarise,
    step_08_render_governance_graphs,
    step_09_build_dashboard_data,
    step_10_launch_dashboard
)

import traceback


def run():

    validate_env()

    state = RunState()

    try:

        print("\n[RUN] STEP 00")
        step_00_bootstrap_governance.run(state)

        print("\n[RUN] STEP 01")
        step_01_python_SoT.run(state)

        print("\n[RUN] STEP 02")
        step_02_promote_outputs.run(state)

        print("\n[RUN] STEP 02B")
        step_02b_sync_latest_sources.run(state)

        print("\n[RUN] STEP 02C")
        step_02c_load_raw.run(state)

        print("\n[RUN] STEP 03")
        step_03_check_snowflake_env.run(state)

        print("\n[RUN] STEP 04")
        step_04_run_dbt.run(state)

        print("\n[RUN] STEP 05")
        step_05_run_lineage.run(state)

        print("\n[RUN] STEP 06")
        step_06_run_drift.run(state)

        print("\n[RUN] STEP 08")
        step_08_render_governance_graphs.run(state)

        print("\n[RUN] STEP 09")
        step_09_build_dashboard_data.run(state)

        print("\n[RUN] STEP 10")
        step_10_launch_dashboard.run(state)

        print("\n[RUN] PIPELINE SUCCESS")

    except Exception as e:

        print("\n" + "=" * 80)
        print("PIPELINE FAILURE")
        print("=" * 80)

        print(f"\nERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {str(e)}\n")

        traceback.print_exc()

        # IMPORTANT:
        # temporarily comment this out
        # state.fail(e)

        raise

    finally:

        print("\n[RUN] SUMMARY")

        try:
            step_07_summarise.run(state)
        except Exception:
            print("\n[RUN] SUMMARY FAILED")
            traceback.print_exc()


if __name__ == "__main__":
    run()