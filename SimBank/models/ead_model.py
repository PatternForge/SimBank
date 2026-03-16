import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from .base import select_features


def train_ead_model(df, sample_frac=0.15):
    m = df["ExposureAtDefault"].notna()
    d = df.loc[m].copy()

    n = len(d)
    if 0 < sample_frac < 1.0:
        k = min(int(sample_frac * n), 200000)
        if 0 < k < n:
            d = d.sample(n=k).copy()

    ban = {"ARREAR", "STAGE", "ECL", "IMPAIR", "DEFAULT", "BUCKET", "LOSS", "PROVISION", "DELINQU", "PASTDUE", "DPD", "RESERVE"}
    drop_cols = [c for c in d.columns if c != "ExposureAtDefault" and any(p in c.upper() for p in ban)]
    d = d.drop(columns=drop_cols, errors="ignore")

    X, y, feats, pre = select_features(d, target="ExposureAtDefault")

    if "DateOfPortfolio" in d.columns and pd.api.types.is_datetime64_any_dtype(d["DateOfPortfolio"]):
        d_sorted = d.sort_values("DateOfPortfolio")
        cutoff = int(len(d_sorted) * 0.8)
        X_tr = d_sorted[feats].iloc[:cutoff]
        y_tr = d_sorted["ExposureAtDefault"].iloc[:cutoff].values
        X_va = d_sorted[feats].iloc[cutoff:]
        y_va = d_sorted["ExposureAtDefault"].iloc[cutoff:].values
        idx_va = X_va.index
    elif "CustomerID" in d.columns:
        gkf = GroupKFold(n_splits=5)
        split = next(gkf.split(X, y, groups=d["CustomerID"].values))
        tr_idx, va_idx = split
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        idx_va = X_va.index
    else:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2)
        idx_va = X_va.index

    reg = HistGradientBoostingRegressor()
    pipe = Pipeline([("pre", pre), ("reg", reg)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)

    return {
        "model": pipe,
        "features": feats,
        "r2": float(r2_score(y_va, pred)),
        "mae": float(mean_absolute_error(y_va, pred)),
        "pred": pred,
        "index": idx_va,
        "n_rows": int(len(d))
    }

