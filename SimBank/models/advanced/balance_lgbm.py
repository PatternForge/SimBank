import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from SimBank.models.base import select_features


def run(df: pd.DataFrame):

    df = df.copy()
    if "BalanceNextMonth" not in df.columns:
        raise ValueError("BalanceNextMonth not found. Call add_balance_next_month(df) before this.")

    X, y, feats, _ = select_features(df, target="BalanceNextMonth")
    feats = [c for c in feats if c not in {"AccountID", "NumberID", "CustomerID"}]

    cat_cols = df[feats].select_dtypes(exclude=["number"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))

    X = df[feats].copy()
    y = df["BalanceNextMonth"].astype(float).values

    if "DateOfPortfolio" in df.columns and pd.api.types.is_datetime64_any_dtype(df["DateOfPortfolio"]):
        data_sorted = df.sort_values("DateOfPortfolio")
        cutoff = int(len(data_sorted) * 0.8)
        X_tr = data_sorted[feats].iloc[:cutoff]
        y_tr = data_sorted["BalanceNextMonth"].iloc[:cutoff].values
        X_te = data_sorted[feats].iloc[cutoff:]
        y_te = data_sorted["BalanceNextMonth"].iloc[cutoff:].values
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    params = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_samples=50, verbose=-1)

    d_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=list(X_tr.columns))
    d_te = lgb.Dataset(X_te, label=y_te, feature_name=list(X_te.columns))

    model = lgb.train(params, d_tr, num_boost_round=3000, valid_sets=[d_tr, d_te], valid_names=["train", "valid"], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

    y_pred = model.predict(X_te, num_iteration=model.best_iteration)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mae = float(mean_absolute_error(y_te, y_pred))
    metrics = {"rmse": rmse, "mae": mae}

    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    residuals = y_te - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, s=10, alpha=0.6)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted – BalanceNextMonth")
    plt.grid(True, alpha=0.3)
    plt.show()

    importances = model.feature_importance(importance_type="gain")
    fi = pd.DataFrame({"feature": feats, "importance": importances}).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 10))
    plt.barh(fi["feature"].head(30)[::-1], fi["importance"].head(30)[::-1])
    plt.title("Top 30 Feature Importances (gain) – Regression")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return model, fi, metrics
