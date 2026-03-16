import numpy as np


def add_interest_types_and_ftp(cfg, df, rng):
    loan = df["TypeOfAccount"].eq("Loan")
    dep = df["TypeOfAccount"].eq("Deposit")
    df["InterestRateType"] = None
    df.loc[dep, "InterestRateType"] = "Variable"
    df.loc[loan, "InterestRateType"] = rng.choice(["Variable","Fixed"], p=[0.96,0.04], size=loan.sum())
    fixed = df["InterestRateType"].eq("Fixed")
    variable = df["InterestRateType"].eq("Variable")
    base = np.zeros(len(df))
    addon = np.zeros(len(df))
    basis = np.zeros(len(df))
    liq = np.zeros(len(df))
    def fill(mask, low, high, arr):
        arr[mask] = rng.uniform(low, high, size=mask.sum())
    fill(fixed, 4.00, 5.50, base)
    fill(fixed, 0.75, 2.00, addon)
    fill(fixed, 0.10, 0.30, basis)
    fill(fixed, 0.20, 0.50, liq)
    mv = variable & ~dep
    fill(mv, 3.50, 4.50, base)
    fill(mv, 1.00, 2.50, addon)
    fill(mv, 0.05, 0.25, basis)
    fill(mv, 0.10, 0.40, liq)
    fill(dep, 2.50, 3.50, base)
    fill(dep, -0.25, 0.75, addon)
    fill(dep, 0.00, 0.15, basis)
    fill(dep, 0.05, 0.25, liq)
    df["BaseRate"] = base.round(4)
    df["AddonRate"] = addon.round(4)
    df["BasisCost"] = basis.round(4)
    df["LiquidityRate"] = liq.round(4)
    df["TransferRate"] = df["BaseRate"] + df["AddonRate"]
    df["TransferSpread"] = df["TransferRate"] - df["InterestRate"]
    df["FundingIndex"] = np.where(df["AccountSource"].eq("BBK"), 0.03, 0.015)
    df = df.copy()
    return df
