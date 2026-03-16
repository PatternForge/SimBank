import numpy as np


def add_ecl(cfg, df, rng):
    loan = df["TypeOfAccount"].eq("Loan")
    cs = df["CreditScore"].astype("float32")
    lvr = df["LVR"].fillna(0).clip(0, 1.5)
    lmi = df["LMIFlag"].eq("Y").astype("float32")
    mv = df["MarketValue"].replace(0, np.nan)
    arr_amt = df["ArrearsAmount"].fillna(0.0)
    arr_days = df["ArrearsDays"].fillna(0).astype("int32")
    arr_ratio = (arr_amt / mv).fillna(0.0).clip(0, 5.0)
    lvr_eff = np.clip(0.15 * (lvr - 0.60), -0.10, 0.30)
    lmi_relief = -0.05 * lmi
    sev_adj = 0.20 * arr_ratio.clip(0.0, 1.0)
    lgd_base = 0.25 + lvr_eff + lmi_relief + sev_adj
    df["PD"] = 1 - np.clip(cs / 1000.0, 0.01, 0.99)
    df["LGD"] = np.clip(lgd_base, 0.10, 0.90)
    ead = df["ExposureAtDefault"].fillna(0.0)
    pdv = df["PD"].clip(0.001, 0.999)
    lgdv = df["LGD"].clip(0.10, 0.90)
    ecl_all = pdv * lgdv * ead
    ecl_dpd = np.where(arr_days > 0, pdv * lgdv * ead, 0.0)
    df["ExpectedCreditLoss"] = np.where(cfg.use_ifrs9_style, ecl_all, ecl_dpd)
    stage = np.where((arr_days >= 1) & (arr_days <= 29), 1, np.where((arr_days >= 30) & (arr_days <= 59), 2, np.where(arr_days >= 60, 3, 0)))
    df["StageOfECL"] = 0
    df.loc[loan, "StageOfECL"] = stage[loan]
    df = df.copy()
    return df
