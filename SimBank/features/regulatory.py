import numpy as np


def add_regulatory_fields(cfg, df, rng):
    loan = df["TypeOfAccount"].eq("Loan")
    df["LVRRWA"] = (df["LVR"] * 100).round(2)
    df["LMIFlag"] = np.where(df["LVRRWA"] > 80.0, "Y", "N")
    m_loan = loan
    m_retl = df["ExposureGroup"].eq("RETL")
    m_corp = df["ExposureGroup"].eq("CORP")
    m_lmi = df["LMIFlag"].eq("Y")
    df["RiskWeight"] = np.select([m_loan & m_retl & m_lmi, m_loan & m_retl & ~m_lmi, m_loan & m_corp, m_loan & ~(m_retl | m_corp)], [0.50,0.75,1.00,0.50], default=np.nan).astype("float32")
    ead_base = np.clip(df["InterestAccrued"] + df["AccountBalance"] - df["InterestNotBoughtToAccount"] - df["AllocatedCashCollateral"], 0, None)
    df["ExposureAtDefault"] = np.where(df["AccountBalance"] <= 0, 0.0, ead_base).astype("float32")
    df["OnBalRWA"] = df["ExposureAtDefault"] * df["RiskWeight"]
    df["OffBalRWA"] = np.maximum(df["ExposureNetCredit"] - df["TransferRate"], 0) * df["RiskWeight"]
    df["TotalRWA"] = df["OnBalRWA"] + df["OffBalRWA"]
    df["CapitalCharge"] = df["TotalRWA"] * 0.08
    df["CapitalBufferImpact"] = df["TotalRWA"] * (0.025 + np.where(m_corp, 0.01, 0.0))
    df = df.copy()
    return df
