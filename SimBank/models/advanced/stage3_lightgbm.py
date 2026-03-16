import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from SimBank.models.base import select_features


def run(df):

    df = df.copy()
    if "StageOfECL" not in df.columns:
        raise ValueError("StageOfECL not found")
    df["Stage3Flag"] = (df["StageOfECL"] == 3).astype(int)

    X, y, feats, _ = select_features(df, target="Stage3Flag")

    name_patterns = {"ARREAR", "STAGE", "ECL", "IMPAIR", "DEFAULT", "BUCKET", "LOSS", "PROVISION", "DELINQU", "PASTDUE", "DPD"}
    feats = [c for c in feats if c not in {"AccountID", "NumberID", "CustomerID", "DateOfPortfolio"}]
    feats = [c for c in feats if not any(p in c.upper() for p in name_patterns)]

    num_cols = df[feats].select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df[feats].select_dtypes(exclude=["number"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))

    X = df[feats].copy()
    y = df["Stage3Flag"].values

    if len(np.unique(y)) < 2:
        raise ValueError("Stage3Flag has a single class")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)

    mono_map = {"LVR": +1, "LoanToIncome": +1, "DebtServiceRatio": +1, "AnnualIncome": -1, "CreditScore": -1}
    mono_constraints = [mono_map.get(c, 0) for c in feats]

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

    pos_ratio = float(y_tr.mean())
    if 0 < pos_ratio < 0.5:
        params["scale_pos_weight"] = (1 - pos_ratio) / pos_ratio
    params["monotone_constraints"] = mono_constraints

    d_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=list(X_tr.columns))
    d_te = lgb.Dataset(X_te, label=y_te, feature_name=list(X_te.columns))

    model = lgb.train(
        params,
        d_tr,
        num_boost_round=3000,
        valid_sets=[d_tr, d_te],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )

    y_prob = model.predict(X_te, num_iteration=model.best_iteration)
    auc_roc = roc_auc_score(y_te, y_prob)
    pr_p, pr_r, _ = precision_recall_curve(y_te, y_prob)
    pr_auc = auc(pr_r, pr_p)
    brier = brier_score_loss(y_te, y_prob)
    metrics = {"auc": auc_roc, "pr_auc": pr_auc, "brier": brier}

    plt.figure(figsize=(7, 5))
    plt.plot(pr_r, pr_p, label=f"PR (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall – Stage 3 (LGBM)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    importances = model.feature_importance(importance_type="gain")
    fi = pd.DataFrame({"feature": feats, "importance": importances}).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 10))
    plt.barh(fi["feature"].head(30)[::-1], fi["importance"].head(30)[::-1])
    plt.title("Top 30 Feature Importances (gain)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    calibrated = CalibratedClassifierCV(estimator=lgb.LGBMClassifier(**params, n_estimators=model.best_iteration), method="isotonic", cv=3)
    calibrated.fit(X_tr, y_tr)
    y_cal = calibrated.predict_proba(X_te)[:, 1]

    from sklearn.calibration import calibration_curve
    pt, pp = calibration_curve(y_te, y_prob, n_bins=10, strategy="uniform")
    pt_c, pp_c = calibration_curve(y_te, y_cal, n_bins=10, strategy="uniform")

    plt.figure(figsize=(7, 5))
    plt.plot(pp, pt, "o-", label="Uncalibrated")
    plt.plot(pp_c, pt_c, "o-", label="Isotonic")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram – Stage 3 (LGBM)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return model, fi, metrics
