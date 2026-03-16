import numpy as np


def add_stress(cfg, df, rng):
    df["MonthlyDepositFrequency"] = rng.poisson(lam=4, size=len(df))
    df["StressScore"] = np.clip(rng.normal(loc=0.5, scale=0.2, size=len(df)), 0, 1)
    df["WithdrawalHistory"] = rng.binomial(n=10, p=0.3, size=len(df))
    df["MacroVolatilityIndex"] = np.clip(rng.normal(loc=0.6, scale=0.15, size=len(df)), 0, 1)
    pdv = df["PD"].clip(0.001, 0.999)
    lgdv = df["LGD"].clip(0.10, 0.90)
    ead = df["ExposureAtDefault"].fillna(0.0)
    pd_stress = np.clip(pdv * 1.5 + 0.05 * df["StressScore"] + 0.05 * df["MacroVolatilityIndex"], 0.001, 0.999)
    lgd_stress = np.clip(lgdv + 0.10 + 0.05 * (df["LVR"].fillna(0) > 0.80), 0.10, 0.90)
    df["StressScenarioFlag"] = np.where(df["StressScore"] + df["MacroVolatilityIndex"] > 1.2, "Severe", "Base")
    df["StressLossEstimate"] = (pd_stress * lgd_stress * ead).round(2)
    df = df.copy()
    return df
