import numpy as np


def add_arrears_and_provisions(cfg, df, rng):
    loan = df["TypeOfAccount"].eq("Loan")
    n_loans = loan.sum()
    df["ArrearsAmount"] = 0.0
    k = int(0.0306 * n_loans)
    if k > 0:
        idx = df[loan].index.values
        sel = rng.choice(idx, size=k, replace=False)
        pct = rng.beta(0.5, 2.0, size=k)
        df.loc[sel, "ArrearsAmount"] = df.loc[sel, "AccountBalance"].abs() * pct
    df["ArrearsDays"] = 0
    df.loc[loan, "ArrearsDays"] = rng.integers(0, 181, size=n_loans)
    df.loc[df["ArrearsAmount"].eq(0), "ArrearsDays"] = 0
    df["AccountProvision"] = 0.0
    k2 = int(0.00039 * n_loans)
    if k2 > 0:
        idx2 = df[loan].index.values
        sel2 = rng.choice(idx2, size=k2, replace=False)
        pct2 = np.clip(rng.beta(2.0, 1.0, size=k2), 0.05, 0.96)
        df.loc[sel2, "AccountProvision"] = df.loc[sel2, "AccountBalance"].abs() * pct2
    df = df.copy()
    return df
