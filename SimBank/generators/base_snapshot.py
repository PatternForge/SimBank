import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


def build_base_snapshot(cfg, rng):
    date_portfolio = pd.Timestamp.today().normalize() - BDay(1)
    n = cfg.n_records
    account_id = rng.permutation(np.arange(1, n * 3 + 1))[:n]
    account_type = rng.choice(["Business Loan","Business Deposit","Retail Loan","Retail Deposit"], p=[0.03,0.07,0.05,0.85], size=n)
    type_of_account = np.where(np.char.find(account_type.astype(str), "Deposit") >= 0, "Deposit", "Loan")
    account_source = np.where(np.char.find(account_type.astype(str), "Business") >= 0, "BBK", "RTE")
    df = pd.DataFrame({
        "DateOfPortfolio": date_portfolio,
        "AccountID": account_id,
        "AccountType": account_type,
        "NumberID": account_id.astype(str) + account_source.astype(str),
        "TypeOfAccount": type_of_account,
        "AccountSource": account_source
    })
    rl = df["AccountType"].eq("Retail Loan")
    rd = df["AccountType"].eq("Retail Deposit")
    bl = df["AccountType"].eq("Business Loan")
    bd = df["AccountType"].eq("Business Deposit")
    df["AccountBalance"] = 0.0
    df.loc[rl, "AccountBalance"] = rng.lognormal(mean=11.0, sigma=0.6, size=rl.sum())
    df.loc[bl, "AccountBalance"] = rng.lognormal(mean=12.2, sigma=0.7, size=bl.sum())
    df.loc[rd, "AccountBalance"] = -rng.lognormal(mean=9.0, sigma=0.8, size=rd.sum())
    df.loc[bd, "AccountBalance"] = -rng.lognormal(mean=11.2, sigma=0.6, size=bd.sum())
    dep = df["TypeOfAccount"].eq("Deposit")
    df.loc[dep, "AccountBalance"] = df.loc[dep, "AccountBalance"].clip(upper=0)
    loan = df["TypeOfAccount"].eq("Loan")
    mult = rng.uniform(1.15, 2.5, size=n)
    df["MarketValue"] = np.where(loan, np.round(mult * np.abs(df["AccountBalance"]), 2), np.nan)
    df["InterestRate"] = np.where(loan, np.clip(rng.normal(4.5, 1.0, size=n), 0.1, 9.5), np.clip(rng.normal(1.8, 0.5, size=n), 0.1, 4.0))
    df["InterestRate"] = df["InterestRate"].round(2)
    df["AnnualRate"] = df["InterestRate"] / 100.0
    df["MonthlyRate"] = (1 + df["AnnualRate"])**(1/12) - 1
    df["CreditScore"] = rng.integers(300, 999, size=n)
    df["LVR"] = np.where(loan, df["AccountBalance"] / df["MarketValue"], np.nan)
    if not cfg.allow_business_lvr_gt_80:
        df.loc[bl, "LVR"] = df.loc[bl, "LVR"].clip(upper=0.80)
    for c in ["AccountType","TypeOfAccount","AccountSource"]:
        df[c] = df[c].astype("category")
    df["AccountID"] = df["AccountID"].astype(cfg.dtype_int)
    num32 = ["AccountBalance","MarketValue","InterestRate","AnnualRate","MonthlyRate","LVR"]
    df[num32] = df[num32].astype(cfg.dtype_float)
    return df
