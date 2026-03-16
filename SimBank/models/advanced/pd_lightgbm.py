import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from SimBank.models.base import select_features


def make_default_proxy(df):

    df = df.copy()
    cond_stage3 = (df["StageOfECL"] == 3)
    cond_income = (df["NetDisposableIncome"] < 0) & (df["LVR"] > 0.85)
    cond_arrears = (df["ArrearsDays"] >= 60) & (df["LoanToIncome"] > 6)
    df["Default12M"] = (cond_stage3 | cond_income | cond_arrears).astype(int)
    return df


def run(df):

    df = make_default_proxy(df.copy())
    X, y, feats, _ = select_features(df, target="Default12M")
    name_patterns = {"ARREAR", "STAGE", "ECL", "IMPAIR", "DEFAULT", "BUCKET", "LOSS", "PROVISION", "DELINQU", "PASTDUE", "DPD"}
    feats = [c for c in feats if c not in {"AccountID", "NumberID", "CustomerID", "DateOfPortfolio"}]
    feats = [c for c in feats if not any(p in c.upper() for p in name_patterns)]

    cat_cols = df[feats].select_dtypes(exclude=["number"]).columns.tolist()
    num_cols = df[feats].select_dtypes(include=["number"]).columns.tolist()

    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))

    X = df[feats].copy()
    y = df["Default12M"].values

    splits = GroupKFold(n_splits=5).split(X, y, groups=df["CustomerID"].values)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_samples": 50,
        "verbose": -1
    }

    mono_map = {"LVR": +1, "LoanToIncome": +1, "DebtServiceRatio": +1, "AnnualIncome": -1, "CreditScore": -1}
    params["monotone_constraints"] = [mono_map.get(c, 0) for c in feats]

    aucs = []
    best_iterations = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        p = params.copy()
        pos_ratio = float(y_tr.mean())
        if 0 < pos_ratio < 0.5:
            p["scale_pos_weight"] = (1 - pos_ratio) / pos_ratio

        d_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=list(X_tr.columns))
        d_va = lgb.Dataset(X_va, label=y_va, feature_name=list(X_va.columns))
        model = lgb.train(
            p,
            d_tr,
            num_boost_round=3000,
            valid_sets=[d_tr, d_va],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        y_pred = model.predict(X_va, num_iteration=model.best_iteration)
        aucs.append(float(roc_auc_score(y_va, y_pred)))
        best_iterations.append(int(model.best_iteration))
        print(f"Fold {fold} AUC: {aucs[-1]:.4f}")

    print(f"Mean CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    final_iter = int(np.mean(best_iterations)) if best_iterations else 1500

    d_all = lgb.Dataset(X, label=y, feature_name=list(X.columns))
    final_model = lgb.train(params, d_all, num_boost_round=final_iter)

    importances = final_model.feature_importance(importance_type="gain")
    fi = pd.DataFrame({"feature": feats, "importance": importances}).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 10))
    plt.barh(fi["feature"].head(30)[::-1], fi["importance"].head(30)[::-1])
    plt.title("Top 30 Feature Importances (gain) – PD")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    cal = CalibratedClassifierCV(estimator=lgb.LGBMClassifier(**params, n_estimators=final_iter), method="isotonic", cv=3)
    cal.fit(X, y)

    metrics = {"cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs))}
    return final_model, cal, fi, metrics
