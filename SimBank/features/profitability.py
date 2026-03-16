import numpy as np


def add_profitability(cfg, df, rng):
    df["FundingCostRate"] = (df["FundingIndex"] + df["BasisCost"] + df["LiquidityRate"]).astype("float32")
    df["FundingCost"] = df["InterestAccrued"] * rng.uniform(0.2, 0.5)
    df["FeesCharged"] = rng.uniform(50, 300, size=len(df))
    df["OperationalCost"] = (df["InterestAccrued"] + df["FeesCharged"]) * rng.uniform(0.1, 0.3)
    num = df["InterestAccrued"].fillna(0) + df["FeesCharged"].fillna(0) - df["ExpectedCreditLoss"].fillna(0) - df["FundingCost"].fillna(0) - df["OperationalCost"].fillna(0)
    den = df["CapitalCharge"].replace(0, np.nan)
    df["RAROC"] = (num / den).round(4)
    df = df.copy()
    return df
