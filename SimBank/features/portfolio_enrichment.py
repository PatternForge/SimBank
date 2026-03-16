import numpy as np
import pandas as pd


def add_portfolio_enrichment(cfg, df, rng):
    df = df.assign(
        DayOfMonth=df["DateOfPortfolio"].dt.day.astype(cfg.dtype_int),
        AggregatedMonthlyBalance=lambda d: d["AccountBalance"] * d["DayOfMonth"],
        AverageMonthlyBalance=lambda d: d["AggregatedMonthlyBalance"] / d["DayOfMonth"],
        AccountPrincipal=lambda d: d["AccountBalance"].abs(),
        Currency=rng.choice(["AUD","USD","EUR","GBP","JPY"], p=[0.88,0.05,0.03,0.02,0.02], size=len(df)),
        GeographicRegion=rng.choice(["QLD","NSW","VIC","WA","SA","TAS","NT","ACT"], p=[0.20,0.30,0.25,0.10,0.06,0.03,0.03,0.03], size=len(df))
    )

    bins = list(np.arange(0, 101, 5)) + [float("inf")]
    labels = ["0 to <= 5%","5 to <= 10%","10 to <= 15%","15 to <= 20%","20 to <= 25%","25 to <= 30%","30 to <= 35%","35 to <= 40%","40 to <= 45%","45 to <= 50%","50 to <= 55%","55 to <= 60%","60 to <= 65%","65 to <= 70%","70 to <= 75%","75 to <= 80%","80 to <= 85%","85 to <= 90%","90 to <= 95%","95 to <= 100%",">100%"]
    df["LVRBand"] = pd.cut((df["LVR"] * 100).astype("float64"), bins=bins, labels=labels, right=True).astype("category")
    loan = df["TypeOfAccount"].eq("Loan")
    dep = df["TypeOfAccount"].eq("Deposit")
    min_orig = df.groupby("CustomerID")["DateOfOrigination"].transform("min").fillna(df["DateOfPortfolio"])
    df = df.assign(
        RelationshipLengthYears=np.round(((df["DateOfPortfolio"] - min_orig).dt.days / 365).fillna(0), 2),
        GroupExposureRank=df.groupby("CustomerID")["TotalAccountExposure"].rank(method="dense", ascending=False).astype(cfg.dtype_int),
        LiquidityBucket=np.select([df["Term"] <= 12, df["Term"] <= 36, df["Term"] > 36], ["Short","Medium","Long"], default="N/A"),
        FundingSource=np.where(dep, "Retail", np.where((df["AccountSource"].eq("BBK")) & (df["AccountPrincipal"] > 1_000_000), "Wholesale", "Retail")),
        LiquidityPremium=(df["LiquidityRate"] * (df["Term"] / 360)).round(4),
        StableFundingFlag=np.where(dep & (df["Term"] >= 12), "Y", "N"),
        InterestRateShockImpact=(df["AccountPrincipal"] * 0.02 * (df["Term"] / 12)).round(2),
        CreditSpreadShockImpact=(df["ExposureAtDefault"] * np.where(df["ExposureGroup"].eq("CORP"), 0.01, 0.005)).round(2),
        FXShockImpact=np.where(df["Currency"] != "AUD", (df["ExposureAtDefault"] * 0.10), 0.0).round(2)
    )

    pdv = df["PD"].clip(0.001, 0.999)
    lgdv = df["LGD"].clip(0.10, 0.90)
    ead = df["ExposureAtDefault"].fillna(0.0)
    df = df.assign(
        StressAdjustedPD=np.clip(pdv * 1.35, 0.001, 0.999),
        StressAdjustedLGD=np.clip(lgdv + 0.08, 0.10, 0.90)
    )

    res = loan & df["CollateralCategory"].eq("Residential Property")
    comm = loan & df["CollateralCategory"].isin(["Commercial","Agri","Dev Finance"])
    pi = loan & df["CollateralCategory"].eq("PI")
    other = loan & df["CollateralCategory"].eq("Other")
    df["ExposureClass"] = "N/A"
    std_res = res & (df["LVR"] <= 0.8) & df["AmortizationType"].eq("P&I")
    df.loc[std_res, "ExposureClass"] = "Standard Residential Mortgage"
    df.loc[res & ~std_res, "ExposureClass"] = "Non-standard Residential Mortgage"
    std_comm = comm & (df["LVR"] <= 0.6) & df["AmortizationType"].eq("P&I")
    df.loc[std_comm, "ExposureClass"] = "Standard Commercial Property"
    df.loc[comm & ~std_comm, "ExposureClass"] = "Non-standard Commercial Property"
    df.loc[pi | other, "ExposureClass"] = "Other Secured Exposure"
    unsec = loan & ~(res | comm | pi | other)
    df.loc[unsec, "ExposureClass"] = "Unsecured Exposure"

    df = df.assign(
        ExposureSubClass=np.where(df["TypeOfAccount"].ne("Loan"), "Not Applicable",
                                  np.where(df["TotalBusinessGroupExposure"] > 5_000_000, "General Corporate - Other",
                                           np.where(df["TotalBusinessGroupExposure"] > 1_500_000, "General Corporate - SME Corporate",
                                                    "General Corporate - SME Retail"))),
        PortfolioSegment=df["ExposureGroup"].map({"RETL":"Retail Mortgage","COMM":"Commercial Property","RESI":"Residential Mortgage","CORP":"Corporate"}).fillna("Other"),
        IndustrySector=np.select([df["CollateralCategory"].eq("Agri"), df["CollateralCategory"].eq("Commercial"),
                                  df["CollateralCategory"].eq("Dev Finance"), df["CollateralCategory"].eq("PI"),
                                  df["CollateralCategory"].eq("Other")],
                                 ["AGRI","COMM","DEVF","PI","OTHER"], default=np.where(res, "RESI", "UNSEC")),
        BasisCalculation=df["BaseRate"] - df["InterestRate"],
        IRRCalculation=df["InterestRate"] / (1 + df["Term"].replace(0, 1)),
        LVRSource=np.where(df["AccountType"].eq("Retail Loan"), "0 - Resi LVR",
                           np.where(df["AccountSource"].eq("FWD"), "1 - LVR Orig",
                                    np.where(df["AccountType"].eq("Business Loan"), "2 - BB CR",
                                             np.where(df["TypeOfAccount"].eq("Deposit"), "N/A", "Missing")))),
        SourceLVR=np.where(loan, (df["AccountBalance"] / df["Limit"].replace(0, np.nan)).fillna(1.0).clip(upper=1.5), np.nan),
        RiskWeightedAssetLVR=df["AccountPrincipal"] * df["RiskWeight"],
        OffBalAmount=0.0,
        CashbackAmount=0.0
    )

    cashback_mask = loan & df["InterestRateType"].eq("Variable")
    if cashback_mask.sum() > 0:
        df.loc[cashback_mask, "CashbackAmount"] = rng.uniform(1000, 5000, size=cashback_mask.sum())

    df = df.assign(
        CapitalOffBalanceAmount=df["OffBalAmount"].fillna(0) + df["CashbackAmount"].fillna(0),
        OnBalanceExposureAmount=np.where((df["InterestAccrued"] + df["AccountBalance"] - df["InterestNotBoughtToAccount"] - df["AccountProvision"]) < 0, 0,
                                         (df["InterestAccrued"] + df["AccountBalance"] - df["InterestNotBoughtToAccount"] - df["AccountProvision"])),
        CreditExposureAmount=(df["OffBalAmount"].fillna(0) * df["CreditConversionFactor"]) + (df["CashbackAmount"].fillna(0) * df["CreditConversionFactor"]),
        CreditExposureAmountCashback=df["CashbackAmount"].fillna(0) * 0.40,
        RegulatoryPD=np.clip(df["PD"], 0.003, 0.20),
        RegulatoryLGD=np.select([res, comm, unsec], [0.20, 0.45, 0.60], default=df["LGD"]),
        BaselExpectedLoss=lambda d: d["RegulatoryPD"] * d["RegulatoryLGD"] * ead
    )

    retl = loan & df["AccountSource"].eq("RTE")
    corp = loan & df["AccountSource"].eq("BBK")
    dr = np.full(len(df), np.nan, dtype="float32")
    if retl.sum() > 0:
        dr[retl] = 0.25 + 0.20 * rng.random(retl.sum())
    if corp.sum() > 0:
        dr[corp] = 0.20 + 0.20 * rng.random(corp.sum())
    dr = np.nan_to_num(dr, nan=0.30)

    monthly_income = df["MonthlyRepayment"].values / np.where(dr == 0, 0.30, dr)
    if retl.sum() > 0:
        monthly_income[retl] = np.clip(monthly_income[retl], None, 60000)
    annual_income = monthly_income * 12
    loan_to_income = df["AccountPrincipal"].values / np.where(annual_income == 0, np.nan, annual_income)
    est_expenses = np.where(res, 2400, np.nan)
    net_disp = np.where(loan, (monthly_income - df["MonthlyRepayment"].values - est_expenses), np.nan)
    aff_flag = np.where(loan & (net_disp > 500), "Comfortable",
                        np.where(loan & (net_disp > 0), "Tight",
                                 np.where(loan & (net_disp < 0), "Unaffordable", "N/A")))

    rating = df["Rating"].values.copy() if "Rating" in df.columns else np.full(len(df), np.nan)
    base_rg = np.clip(np.ceil((1000 - df["CreditScore"].values) / 35), 1, 20).astype("float32")
    risk_grade = np.where(~np.isnan(rating), rating, base_rg).astype(np.int32)
    risk_class = np.where(risk_grade <= 6, "Low", np.where(risk_grade <= 12, "Medium", "High"))

    new_cols = pd.DataFrame({
        "DebtServiceRatio": dr,
        "MonthlyIncome": monthly_income,
        "AnnualIncome": annual_income,
        "LoanToIncome": loan_to_income,
        "EstimatedLivingExpenses": est_expenses,
        "NetDisposableIncome": net_disp,
        "AffordabilityFlag": aff_flag,
        "RiskGrade": risk_grade,
        "RiskClass": risk_class
    })
    df = df.join(new_cols)

    df["ImpairedFlag"] = df["ArrearsDays"] > 90

    seasoning_months = (df["DaysSinceSettlement"].fillna(0) / 30.0).astype(cfg.dtype_float)
    arrears_days = df["ArrearsDays"].fillna(0).astype(int).values
    arrears_bucket = np.select(
        [arrears_days == 0, (arrears_days >= 1) & (arrears_days <= 29), (arrears_days >= 30) & (arrears_days <= 59),
         (arrears_days >= 60) & (arrears_days <= 89), arrears_days >= 90],
        ["0","1-29","30-59","60-89","90+"], default="0"
    )
    dpd_flag = (arrears_days > 0).astype(int)
    rate_bucket = pd.cut(df["InterestRate"], bins=[0,2,4,6,8,100], labels=["<=2%","2-4%","4-6%","6-8%",">8%"], right=True)
    rw_cat = np.select([df["RiskWeight"] <= 0.50, df["RiskWeight"] <= 0.75, df["RiskWeight"] > 0.75], ["Low","Medium","High"], default="Medium")
    ead_util = np.where(df["Limit"] > 0, df["ExposureAtDefault"] / df["Limit"], np.nan)
    util_bucket = pd.cut(ead_util, bins=[-np.inf,0.25,0.50,0.75,1.00,np.inf], labels=["<=25%","25-50%","50-75%","75-100%",">100%"])
    undrawn_exposure = df["AvailableBalance"].values
    capital_density = np.where(df["ExposureAtDefault"] > 0, df["CapitalCharge"] / df["ExposureAtDefault"], np.nan)
    exposure_density = np.where(df["AccountPrincipal"] > 0, df["TotalRWA"] / df["AccountPrincipal"], np.nan)
    customer_tier = np.select([df["TotalBusinessGroupExposure"] > 5_000_000, df["TotalBusinessGroupExposure"] > 1_500_000, df["TotalBusinessGroupExposure"] > 500_000], ["Platinum","Gold","Silver"], default="Bronze")
    pd_vals = df["PD"].values
    cust_risk_seg = np.where((pd_vals > 0.25) & (arrears_days > 60), "High", np.where((pd_vals > 0.10) & (arrears_days > 30), "Medium", "Low"))
    avg_ir = df.groupby("CustomerID")["InterestRate"].transform("mean")
    tenor_years = (df["Term"].astype(float) / 12.0).astype(cfg.dtype_float)
    io_flag = np.where(df["AmortizationType"].eq("InterestOnly"), "Y", "N")
    bullet_flag = np.where(df["AmortizationType"].eq("Bullet"), "Y", "N")

    add16 = pd.DataFrame({
        "SeasoningMonths": seasoning_months,
        "ArrearsBucket": arrears_bucket,
        "DPDFlag": dpd_flag,
        "RateBucket": rate_bucket.astype("category"),
        "RiskWeightCategory": rw_cat,
        "EADUtilization": ead_util.astype(cfg.dtype_float),
        "UtilizationBucket": util_bucket.astype("category"),
        "UndrawnExposure": undrawn_exposure.astype(cfg.dtype_float),
        "CapitalDensity": capital_density.astype(cfg.dtype_float),
        "ExposureDensity": exposure_density.astype(cfg.dtype_float),
        "CustomerTier": customer_tier,
        "CustomerRiskSegment": cust_risk_seg,
        "AverageInterestRate": avg_ir,
        "TenorYears": tenor_years,
        "IOFlag": io_flag,
        "BulletFlag": bullet_flag
    })
    df = df.join(add16)

    m_imp = df["ImpairedFlag"]
    m_resi = df["CollateralCategory"].eq("Residential Property")
    m_prov = df["AccountProvision"] != 0
    m_neg = (df["AccountBalance"] - df["InterestNotBoughtToAccount"]) < 0

    df["DefaultedExposureClass"] = np.select(
        [m_imp & m_resi, m_imp & ~m_resi & m_prov & m_neg, m_imp & ~m_resi & ~m_prov & ~m_neg],
        ["Defaulted Resi Property","Other Defaulted Property > 0.2","Other Defaulted Property < 0.2"],
        default=np.array(np.nan, dtype=object)
    )

    stage = df["StageOfECL"].astype(int)
    ecl = df["ExpectedCreditLoss"].fillna(0.0).values
    pd_ = df["PD"].fillna(0.0).values
    lgd = df["LGD"].fillna(0.0).values

    df = df.assign(
        Stage1ECL=np.where(stage.values == 1, ecl, 0.0),
        Stage2ECL=np.where(stage.values == 2, ecl, 0.0),
        Stage3ECL=np.where(stage.values == 3, ecl, 0.0),
        Stage1PD=np.where(stage.values == 1, pd_, 0.0),
        Stage2PD=np.where(stage.values == 2, pd_, 0.0),
        Stage3PD=np.where(stage.values == 3, pd_, 0.0),
        Stage1LGD=np.where(stage.values == 1, lgd, 0.0),
        Stage2LGD=np.where(stage.values == 2, lgd, 0.0),
        Stage3LGD=np.where(stage.values == 3, lgd, 0.0)
    )

    df = df.copy()
    return df
