import numpy as np
import pandas as pd
from SimBank.models.advanced.stage3_lightgbm import run as run_stage3_lgbm
from SimBank.models.advanced.stage3_tabular_nn import run as run_stage3_nn
from SimBank.models.advanced.balance_lgbm import run as run_balance_lgbm
from SimBank.models.advanced.pd_lightgbm import run as run_pd_lgbm
from SimBank.features.next_balance import add_balance_next_month


def run_advanced_pack(df: pd.DataFrame, sample_frac: float = 0.15):
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac).reset_index(drop=True)
    df_next = add_balance_next_month(df)

    s3_lgbm_model, s3_lgbm_fi, s3_lgbm_metrics = run_stage3_lgbm(df)
    s3_nn_model, s3_nn_metrics = run_stage3_nn(df, sample_frac=1.0)
    reg_model, reg_fi, reg_metrics = run_balance_lgbm(df_next)
    pd_model, pd_calibrator, pd_fi, pd_metrics = run_pd_lgbm(df)

    return {
        "stage3_lgbm": {"model": s3_lgbm_model, "fi": s3_lgbm_fi, "metrics": s3_lgbm_metrics},
        "stage3_nn": {"model": s3_nn_model, "metrics": s3_nn_metrics},
        "balance_lgbm": {"model": reg_model, "fi": reg_fi, "metrics": reg_metrics},
        "pd_lgbm": {"model": pd_model, "calibrator": pd_calibrator, "fi": pd_fi, "metrics": pd_metrics},
    }
