def validate_lvr_bounds(cfg, df):
    loan = df["TypeOfAccount"].eq("Loan")
    bad = df[loan & ((df["LVR"] < 0) | (df["LVR"] > 1.5))]
    if len(bad):
        raise ValueError(f"lvr_out_of_bounds {len(bad)}")
    if not cfg.allow_business_lvr_gt_80:
        bl = df["AccountType"].eq("Business Loan")
        too_high = df[loan & bl & (df["LVR"] > 0.80)]
        if len(too_high):
            raise ValueError(f"business_lvr_gt_80 {len(too_high)}")
