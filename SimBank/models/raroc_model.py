import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from .base import select_features


def train_raroc_model(df, seed=None, sample_frac=0.15):
    m = df["RAROC"].notna()
    d = df.loc[m]
    n = len(d)
    k = min(int(sample_frac * n), 200000)
    if k > 0 and k < n:
        d = d.sample(n=k, random_state=seed)
    X, y, feats, pre = select_features(d, target="RAROC")
    reg = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
    pipe = Pipeline([("pre", pre), ("reg", reg)])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    return {"model": pipe, "features": feats, "r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred), "pred": pred, "n_rows": len(d)}
