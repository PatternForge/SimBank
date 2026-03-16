def add_collateral(cfg, df, rng):
    loan = df["TypeOfAccount"].eq("Loan")
    bb = loan & df["AccountSource"].eq("BBK")
    n_bb = bb.sum()
    cats = rng.choice(["Agri","Commercial","Dev Finance","PI","Other"], p=[0.51,0.32,0.01,0.11,0.05], size=n_bb)
    df["CollateralCategory"] = None
    df.loc[bb, "CollateralCategory"] = cats
    df.loc[loan & ~bb, "CollateralCategory"] = "Residential Property"
    df["CollateralCategory"] = df["CollateralCategory"].astype("category")
    df["ExposureGroup"] = "N/A"
    res = loan & df["CollateralCategory"].eq("Residential Property")
    comm = loan & df["CollateralCategory"].isin(["Commercial","Agri","Dev Finance"])
    df.loc[res, "ExposureGroup"] = "RESI"
    df.loc[comm, "ExposureGroup"] = "COMM"
    df.loc[loan & df["AccountSource"].eq("BBK") & (df["AccountBalance"] > 1_000_000), "ExposureGroup"] = "CORP"
    df.loc[loan & df["AccountSource"].eq("RTE"), "ExposureGroup"] = "RETL"
    df["ExposureGroup"] = df["ExposureGroup"].astype("category")
    return df
