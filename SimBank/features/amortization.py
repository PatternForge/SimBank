import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


def add_amortization(cfg, df, rng):
    loan = df["TypeOfAccount"].eq("Loan")
    bb = loan & df["AccountSource"].eq("BBK")
    high = bb & (df["AccountBalance"] > 1_000_000)
    low = loan & ~high
    df["AmortizationType"] = None
    df.loc[high, "AmortizationType"] = rng.choice(["InterestOnly","Bullet","P&I"], p=[0.6,0.25,0.15], size=high.sum())
    df.loc[low, "AmortizationType"] = rng.choice(["InterestOnly","Bullet","P&I"], p=[0.2,0.05,0.75], size=low.sum())
    df["AmortizationType"] = df["AmortizationType"].astype("category")
    df["DateOfMaturity"] = pd.NaT
    df.loc[loan, "DateOfMaturity"] = df.loc[loan, "DateOfPortfolio"] + DateOffset(years=30)
    df["DaysUntilMaturity"] = (df["DateOfMaturity"] - df["DateOfPortfolio"]).dt.days
    df["Term"] = (df["DaysUntilMaturity"] / 30).round()
    df["Term"] = df["Term"].fillna(0).astype(cfg.dtype_int)
    r = df["MonthlyRate"].values
    P = df["AccountBalance"].abs().values
    n = df["Term"].values
    io = loan & df["AmortizationType"].eq("InterestOnly")
    pi = loan & df["AmortizationType"].eq("P&I")
    bu = loan & df["AmortizationType"].eq("Bullet")
    pay = np.zeros(len(df))
    pay[pi] = P[pi] * r[pi] * (1 + r[pi])**n[pi] / ((1 + r[pi])**n[pi] - 1)
    pay[io] = P[io] * r[io]
    df["MonthlyRepayment"] = np.round(pay, 2).astype(cfg.dtype_float)
    df.loc[bu, "MonthlyRepayment"] = 0.0
    df = df.copy()
    return df
