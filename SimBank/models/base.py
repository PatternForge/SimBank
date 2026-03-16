import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

ID_COLS = {"AccountID","CustomerID","NumberID"}
DATE_COLS_PREFIX = ("DateOf",)
POST_TARGET_GLOBAL = {
    "StageOfECL","Stage1ECL","Stage2ECL","Stage3ECL",
    "Stage1PD","Stage2PD","Stage3PD",
    "Stage1LGD","Stage2LGD","Stage3LGD"
}


def _is_date_col(c): return c.startswith(DATE_COLS_PREFIX)


def _global_exclude(df):
    excl = set()
    for c in df.columns:
        if c in ID_COLS or _is_date_col(c):
            excl.add(c)
    return excl


def _task_exclusions(df, target):
    t = target.upper()
    cols = set()
    for c in df.columns:
        uc = c.upper()
        if uc == t:
            cols.add(c)
    for c in df.columns:
        uc = c.upper()
        if "PD" in t:
            if any(x in uc for x in ["PD","EXPECTEDCREDITLOSS","BASELEXPECTEDLOSS","REGULATORYPD","STRESSADJUSTEDPD"]):
                cols.add(c)
        if "LGD" in t:
            if any(x in uc for x in ["LGD","EXPECTEDCREDITLOSS","BASELEXPECTEDLOSS","REGULATORYLGD","STRESSADJUSTEDLGD"]):
                cols.add(c)
        if "EXPOSUREATDEFAULT" in t or t == "EAD":
            if any(x in uc for x in ["EXPOSUREATDEFAULT","TOTALRWA","ONBALRWA","OFFBALRWA","CAPITALCHARGE","CAPITALBUFFERIMPACT"]):
                cols.add(c)
        if "RAROC" in t:
            if any(x in uc for x in ["RAROC","CAPITALCHARGE","EXPECTEDCREDITLOSS","FUNDINGCOST","OPERATIONALCOST","FEESCHARGED"]):
                cols.add(c)
        if "STAGEOFECL" in t:
            if any(x in uc for x in ["STAGEOFECL","STAGE1","STAGE2","STAGE3","EXPECTEDCREDITLOSS"]):
                cols.add(c)
    for c in POST_TARGET_GLOBAL:
        if c in df.columns:
            cols.add(c)
    return cols


def select_features(df, target):
    excl = _global_exclude(df) | _task_exclusions(df, target)
    feats = [c for c in df.columns if c not in excl]
    y = df[target].values
    X = df[feats].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in cat_cols if X[c].nunique() < 200]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
    return X, y, feats, pre
