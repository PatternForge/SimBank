import numpy as np
import pandas as pd


def add_backfill_original(cfg, df, rng):
    n = len(df)
    daily_rate = df["AnnualRate"] / 365.0
    liq_calc = np.where(df["TypeOfAccount"].eq("Loan"), df["Term"].astype(float) * 0.01, 0.0)
    loan = df["TypeOfAccount"].eq("Loan")
    asset_class_adv = np.where(loan, np.where(df["AccountType"].str.contains("Retail"), "Residential Mortgage", "Corporate"), "Other")
    exp_return = df["InterestRate"] - df["FundingIndex"]
    impaired_net = df["AccountBalance"] - df["InterestNotBoughtToAccount"]
    retl_src = loan & df["ExposureGroup"].eq("RETL")
    corp_src = loan & df["ExposureGroup"].eq("CORP")
    reg_ac = np.select(
        [
            retl_src & df["CollateralCategory"].eq("Residential Property"),
            corp_src & (df["TotalBusinessGroupExposure"] > 1_500_000),
            corp_src & (df["TotalBusinessGroupExposure"] <= 1_500_000),
            loan & ~df["CollateralCategory"].eq("Residential Property") & retl_src,
            df["TypeOfAccount"].eq("Deposit")
        ],
        ["Retail Mortgage","Corporate","SME Corporate","SME Retail","Deposit"],
        default="Other"
    )
    term_safe = np.where(df["Term"] > 0, df["Term"], 1)
    repay_amt = np.where(loan, df["AccountPrincipal"].values / term_safe, 0.0)
    dev_cost_ratio = rng.uniform(0.1, 0.9, size=n)
    risk_score = (0.4 * df["StressScore"].values
                  + 0.3 * df["MacroVolatilityIndex"].values
                  + 0.2 * (df["WithdrawalHistory"].values / 10.0)
                  - 0.1 * (df["MonthlyDepositFrequency"].values / 10.0))
    withdrawal_risk = (risk_score > 0.5).astype(cfg.dtype_int)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    first_len = rng.integers(4, 9, size=n)
    last_len = rng.integers(4, 13, size=n)
    max_first, max_last = 8, 12
    first_arr = rng.choice(letters, (n, max_first))
    last_arr = rng.choice(letters, (n, max_last))
    first_mask = np.arange(max_first) < first_len[:, None]
    last_mask = np.arange(max_last) < last_len[:, None]
    first_arr = np.where(first_mask, first_arr, " ")
    last_arr = np.where(last_mask, last_arr, " ")
    first_names = np.array(["".join(row).strip().capitalize() for row in first_arr])
    last_names  = np.array(["".join(row).strip().capitalize()  for row in last_arr])
    full_name = np.core.defchararray.add(np.core.defchararray.add(first_names, " "), last_names)
    new_cols_all = {
        "LiquidityCalculation": liq_calc.astype(cfg.dtype_float),
        "AssetClassAdvanced": asset_class_adv,
        "ExpectedReturn": exp_return.astype(cfg.dtype_float),
        "RegulatoryAssetClass": reg_ac,
        "RepaymentAmount": repay_amt.astype(cfg.dtype_float),
        "DevCostRatio": dev_cost_ratio.astype(cfg.dtype_float),
        "WithdrawalRisk": withdrawal_risk,
        "FullName": full_name,
        "DailyRate": daily_rate.astype(cfg.dtype_float),
        "ImpairedNet": impaired_net.astype(cfg.dtype_float)
    }
    to_add = {k: v for k, v in new_cols_all.items() if k not in df.columns}
    if to_add:
        df = df.join(pd.DataFrame(to_add))
    df = df.copy()
    return df
