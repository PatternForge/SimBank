import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

SEG_FEATURES = [
    "TotalGroupExposure","TotalBusinessGroupExposure","AccountPrincipal","ExposureAtDefault",
    "RiskWeight","CapitalCharge","RAROC","PD","LGD"
]


def run_customer_segmentation(df, k=6, seed=None):
    cols = [c for c in SEG_FEATURES if c in df.columns]
    X = df[cols].fillna(0.0).values
    ss = StandardScaler(with_mean=False)
    Xs = ss.fit_transform(X)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=10000, n_init=5, max_iter=100)
    labels = km.fit_predict(Xs)
    return {"labels": labels, "features": cols, "centers": km.cluster_centers_}
