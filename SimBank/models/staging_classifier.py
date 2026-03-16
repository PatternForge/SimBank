import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from .base import select_features


def train_staging_classifier(df, seed=None, sample_frac=0.15):
    m = df["TypeOfAccount"].eq("Loan") & df["StageOfECL"].notna()
    d = df.loc[m]
    n = len(d)
    k = min(int(sample_frac * n), 200000)
    if k > 0 and k < n:
        d = d.sample(n=k, random_state=seed)
    X, y, feats, pre = select_features(d, target="StageOfECL")
    clf = DecisionTreeClassifier(max_depth=6, random_state=seed)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    return {"model": pipe, "features": feats, "f1_macro": f1_score(y, pred, average="macro"), "pred": pred, "n_rows": len(d)}
