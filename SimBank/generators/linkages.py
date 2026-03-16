import numpy as np


def add_linkages(cfg, df, rng):
    loan = df["TypeOfAccount"].eq("Loan")
    dep = df["TypeOfAccount"].eq("Deposit")
    rl = df["AccountType"].eq("Retail Loan")
    df["OffsetFlag"] = "N"
    k = int(0.35 * rl.sum())
    if k > 0:
        idx = df[rl].index.values
        pick = rng.choice(idx, size=k, replace=False)
        df.loc[pick, "OffsetFlag"] = "Y"
    n_dep = dep.sum()
    n_non_offset_loans = (loan & df["OffsetFlag"].eq("N")).sum()
    pool = np.arange(1, n_dep + n_non_offset_loans + 1)
    rng.shuffle(pool)
    df.loc[dep, "CustomerID"] = pool[:n_dep]
    df.loc[loan & df["OffsetFlag"].eq("N"), "CustomerID"] = pool[n_dep:]
    dep_ids = df.loc[dep, "CustomerID"].values
    off_mask = loan & df["OffsetFlag"].eq("Y")
    if off_mask.sum() > 0:
        df.loc[off_mask, "CustomerID"] = rng.choice(dep_ids, size=off_mask.sum(), replace=True)
        rs32 = int(rng.integers(0, 2**32 - 1)) if getattr(cfg, "random_seed", None) is None else int(cfg.random_seed % (2**32))
        pool_dep = df.loc[dep, ["NumberID","AccountBalance"]].sample(n=off_mask.sum(), random_state=rs32).reset_index(drop=True)
        df.loc[off_mask, "LinkedDepositAccount"] = pool_dep["NumberID"].values
        df.loc[off_mask, "LinkedDepositBalance"] = pool_dep["AccountBalance"].values
        loans_link = df.loc[off_mask, ["CustomerID","NumberID","AccountBalance"]].rename(columns={"NumberID":"LinkedLoanAccount","AccountBalance":"LinkedLoanBalance"})
        dep_link_mask = dep & df["CustomerID"].isin(loans_link["CustomerID"])
        dep_links = df.loc[dep_link_mask].merge(loans_link, on="CustomerID", how="left")
        df.loc[dep_links.index, "LinkedLoanAccount"] = dep_links["LinkedLoanAccount"].values
        df.loc[dep_links.index, "LinkedLoanBalance"] = dep_links["LinkedLoanBalance"].values
    df["CustomerID"] = df["CustomerID"].astype(cfg.dtype_int)
    return df
