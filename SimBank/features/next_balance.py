import numpy as np, pandas as pd


def add_balance_next_month(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    out = df.copy()
    if "AccountBalance" not in out.columns: raise ValueError("AccountBalance missing")

    monthly_rate = out["MonthlyRate"].fillna(0.0)
    repayment    = out["MonthlyRepayment"].fillna(0.0)
    stress, macro = out["StressScore"].fillna(0.5), out["MacroVolatilityIndex"].fillna(0.6)

    loan_mask = out["TypeOfAccount"].eq("Loan")
    dep_mask  = out["TypeOfAccount"].eq("Deposit")
    noise = np.random.normal(0.0, 0.02, size=len(out))

    delta_loan = (out["AccountBalance"].abs() * monthly_rate - repayment) * (1 + 0.1*stress + 0.05*macro) + noise * out["AccountBalance"].abs()
    dep_drift  = (out["AccountBalance"] * (monthly_rate - 0.2*stress + 0.1*macro)) + noise * out["AccountBalance"].abs()

    next_bal = out["AccountBalance"].astype(float)
    next_bal[loan_mask] = out.loc[loan_mask, "AccountBalance"].astype(float) - delta_loan[loan_mask]
    next_bal[dep_mask]  = out.loc[dep_mask,  "AccountBalance"].astype(float) + dep_drift[dep_mask]
    next_bal[dep_mask] = np.minimum(next_bal[dep_mask], 0.0)  # keep deposit ≤ 0 per your convention

    out["BalanceNextMonth"] = next_bal.round(2)
    df = df.copy()
    return out
