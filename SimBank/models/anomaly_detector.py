import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

NUM_FEATURES = [
    "InterestRate","AnnualRate","MonthlyRate","AccountBalance","AccountPrincipal","LVR",
    "ArrearsAmount","ArrearsDays","ExposureAtDefault","RiskWeight","CapitalCharge",
    "TotalRWA","DebtServiceRatio","MonthlyIncome","LoanToIncome","PD","LGD"
]


def run_anomaly_detector(df, seed=None):
    cols = [c for c in NUM_FEATURES if c in df.columns]
    X = df[cols].fillna(0.0).values
    ss = StandardScaler(with_mean=False)
    Xs = ss.fit_transform(X)
    iso = IsolationForest(n_estimators=100, max_samples=100000, contamination=0.01, random_state=seed, n_jobs=-1)
    iso.fit(Xs)
    score = -iso.score_samples(Xs)
    flag = (score > np.percentile(score, 99)).astype(int)
    return {"scores": score, "flags": flag, "features": cols}
