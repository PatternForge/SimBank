import numpy as np
import pandas as pd


def add_exposures(cfg, df, rng):
    df["Age"] = rng.integers(19, 66, size=len(df)).astype(cfg.dtype_int)
    birth_dates = df["DateOfPortfolio"] - pd.to_timedelta(df["Age"] * 365, unit="D")
    loan = df["TypeOfAccount"].eq("Loan")
    dep = df["TypeOfAccount"].eq("Deposit")

    earliest_loan_dates = birth_dates + pd.to_timedelta(18 * 365, unit="D")
    earliest_loan_dates_clipped = df["DateOfPortfolio"] - pd.to_timedelta(30 * 365, unit="D")
    valid_loan_start = pd.DataFrame(
        {"earliest": earliest_loan_dates[loan], "clipped": earliest_loan_dates_clipped[loan]}).max(axis=1)

    loan_days_range = (df.loc[loan, "DateOfPortfolio"] - valid_loan_start).dt.days.clip(lower=1)
    rand_loan_days = rng.integers(0, loan_days_range.values, size=loan_days_range.shape[0]) if loan_days_range.shape[
                                                                                                   0] > 0 else np.array(
        [], dtype=int)
    loan_settlement = valid_loan_start + pd.to_timedelta(rand_loan_days,
                                                         unit="D") if rand_loan_days.size else pd.Series([],
                                                                                                         dtype="datetime64[ns]")

    earliest_dep_dates = birth_dates + pd.to_timedelta(12 * 365, unit="D")
    dep_days_range = (df.loc[dep, "DateOfPortfolio"] - earliest_dep_dates[dep]).dt.days.clip(lower=1)
    rand_dep_days = rng.integers(0, dep_days_range.values, size=dep_days_range.shape[0]) if dep_days_range.shape[
                                                                                                0] > 0 else np.array([],
                                                                                                                     dtype=int)
    dep_orig = earliest_dep_dates[dep] + pd.to_timedelta(rand_dep_days, unit="D") if rand_dep_days.size else pd.Series(
        [], dtype="datetime64[ns]")

    df["DateOfOrigination"] = pd.NaT
    df["DateOfSettlement"] = pd.NaT
    if loan_settlement.size:
        df.loc[loan, "DateOfOrigination"] = loan_settlement.values
        df.loc[loan, "DateOfSettlement"] = df.loc[loan, "DateOfOrigination"]
    if dep_orig.size:
        df.loc[dep, "DateOfOrigination"] = dep_orig.values

    df["DaysSinceSettlement"] = (df["DateOfPortfolio"] - df["DateOfSettlement"]).dt.days.fillna(0)
    df["DaysSinceOrigination"] = (df["DateOfPortfolio"] - df["DateOfOrigination"]).dt.days.fillna(0)
    df["Vintage"] = (df["DateOfOrigination"].dt.year.fillna(df["DateOfPortfolio"].dt.year).astype(cfg.dtype_int))

    df["Limit"] = np.where(loan, df["MarketValue"], 0.0)
    df["AvailableBalance"] = (df["Limit"] - df["AccountBalance"]).clip(lower=0)
    ab = df["AccountBalance"].abs()
    avb = df["AvailableBalance"].abs()
    df["AdvanceAmount"] = df["AccountBalance"] * rng.uniform(0.05, 0.3, size=len(df))
    neg_idx = rng.choice(df.index.values, size=int(0.05 * len(df)), replace=False)
    df.loc[neg_idx, "AdvanceAmount"] = -df.loc[neg_idx, "AdvanceAmount"]
    adv = df["AdvanceAmount"].clip(lower=0)
    df["TotalAccountExposure"] = 0.0
    df.loc[loan, "TotalAccountExposure"] = avb[loan] + ab[loan] + adv[loan]
    df.loc[dep, "TotalAccountExposure"] = np.maximum(ab[dep], df["Limit"][dep])
    df["TotalGroupExposure"] = df.groupby("CustomerID")["TotalAccountExposure"].transform("sum")
    df["TotalBusinessGroupExposure"] = np.where(df["AccountSource"].eq("BBK"), df.groupby("CustomerID")["TotalAccountExposure"].transform("sum"), 0.0)
    df["InterestAccrued"] = df["AccountBalance"] * (df["InterestRate"] / 100.0) * (df["DaysSinceSettlement"]/ 365)
    df["AllocatedCashCollateral"] = 0.0
    df.loc[loan, "AllocatedCashCollateral"] = df.loc[loan, "AccountBalance"].abs() * rng.uniform(0.0, 0.3, size=loan.sum())
    df["InterestNotBoughtToAccount"] = df["AccountBalance"] - df["AllocatedCashCollateral"]
    df["ExposureNetCredit"] = np.where(loan, np.clip(df["InterestAccrued"] + df["AccountBalance"], 0, None), 0.0)
    df["CreditConversionFactor"] = np.where(loan, 1.0, 0.0).astype(cfg.dtype_float)
    df = df.copy()
    return df
