
import logging
import time
import numpy as np
import pandas as pd

from SimBank.config import Config
from SimBank.logging_conf import setup_logging
from SimBank.pipeline import run_pipeline
from SimBank.models.train import train_all_models
from SimBank.models.advanced_pack import run_advanced_pack
from SimBank.models.capital_engine import run_capital_scenarios
from SimBank.features.capital import CapitalParams
from SimBank.Output.write_sources import write_sources


def sense_check(df):
    n_rows = len(df)
    n_cols = len(df.columns)
    logging.info(f"records {n_rows} fields {n_cols}")
    first3 = list(df.columns[:3])
    last2 = list(df.columns[-2:])
    out = []
    for t in df["AccountType"].unique():
        g = df[df["AccountType"] == t]
        s = g.loc[:, first3 + last2].sample(min(5, len(g)), random_state=0)
        out.append(s)
    print("sense_check_sample:")
    print(pd.concat(out, ignore_index=True))



def print_models_summary(results):
    print("models_summary:")
    print(f"PD R2: {results['pd']['r2']:.4f} MAE: {results['pd']['mae']:.4f}")
    print(f"LGD R2: {results['lgd']['r2']:.4f} MAE: {results['lgd']['mae']:.4f}")
    print(f"EAD R2: {results['ead']['r2']:.4f} MAE: {results['ead']['mae']:.4f}")
    print(f"Staging F1 (macro): {results['staging']['f1_macro']:.4f}")
    print(f"RAROC R2: {results['raroc']['r2']:.4f} MAE: {results['raroc']['mae']:.4f}")
    print(f"Anomaly features: {len(results['anomaly']['features'])} Top 1% flagged: {int(results['anomaly']['flags'].sum())}")
    print(f"Segmentation clusters: {len(set(results['segment']['labels']))}")



def print_top_scores(df, results, n=5):
    def _match_ids_by_len(pred_len):
        candidates = [
            df["AccountType"].str.contains("Loan", na=False),
            df["AccountType"].eq("Retail Loan"),
            df["AccountType"].eq("Business Loan"),
        ]
        for m in candidates:
            ids = df.loc[m, "AccountID"]
            if len(ids) == pred_len:
                return ids.reset_index(drop=True)
        return None

    frames = []
    for key, col_name in [("pd", "PD_pred"), ("lgd", "LGD_pred"), ("ead", "EAD_pred"), ("raroc", "RAROC_pred")]:
        if key not in results or "pred" not in results[key]:
            continue
        preds = results[key]["pred"]
        ids = _match_ids_by_len(len(preds))
        if ids is not None:
            frames.append(pd.DataFrame({"AccountID": ids, col_name: preds}))
        else:
            frames.append(pd.DataFrame({col_name: preds}))

    print("top_scores_sample:")
    if not frames:
        print("no predictions available")
        return
    out = frames[0]
    for r in frames[1:]:
        if "AccountID" in out.columns and "AccountID" in r.columns:
            out = out.merge(r, on="AccountID", how="outer")
        else:
            out = pd.concat([out, r], axis=1)
    print(out.sample(n=min(n, len(out)), random_state=0))



def write_models_report(path, results):
    lines = []
    lines.append("PD R2: %.6f \nMAE: %.6f" % (results["pd"]["r2"], results["pd"]["mae"]))
    lines.append("LGD R2: %.6f \nMAE: %.6f" % (results["lgd"]["r2"], results["lgd"]["mae"]))
    lines.append("EAD R2: %.6f \nMAE: %.6f" % (results["ead"]["r2"], results["ead"]["mae"]))
    lines.append("Staging F1 (macro): %.6f" % results["staging"]["f1_macro"])
    lines.append("RAROC R2: %.6f \nMAE: %.6f" % (results["raroc"]["r2"], results["raroc"]["mae"]))
    lines.append("Anomaly features: %d \nTop 1%% flagged: %d" % (len(results["anomaly"]["features"]), int(results["anomaly"]["flags"].sum())))
    lines.append("Segmentation clusters: %d" % len(set(results["segment"]["labels"])))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def print_adv_summary(results):
    print("advanced_models_summary:")
    s3 = results["stage3_lgbm"]["metrics"]
    print(f"Stage3 (LGBM) AUC: {s3['auc']:.4f} PR-AUC: {s3['pr_auc']:.4f} Brier: {s3['brier']:.4f}")
    s3n = results["stage3_nn"]["metrics"]
    print(f"Stage3 (NN) AUC: {s3n['auc']:.4f} PR-AUC: {s3n['pr_auc']:.4f}")
    reg = results["balance_lgbm"]["metrics"]
    print(f"BalanceNextMonth RMSE: {reg['rmse']:.4f} MAE: {reg['mae']:.4f}")
    pdm = results["pd_lgbm"]["metrics"]
    print(f"PD proxy CV AUC: {pdm['cv_auc_mean']:.4f} ± {pdm['cv_auc_std']:.4f}")



def _run_advanced_with_banner(df):
    results_adv = run_advanced_pack(df)
    logging.info("Advanced models completed successfully.")
    return results_adv



def run_cap(df):
    scenarios = {
        "Base": CapitalParams(
            cet1=6.5e9,
            at1=1.5e9,
            tier2=1.0e9,
            min_total_cap_ratio=0.08,
            ccb=0.025,
            ccyb=0.0,
            dsib=0.0,
            pillar2=0.0,
            use_stress_addon=False
        ),
        "Mild_Stress": CapitalParams(
            cet1=6.5e9,
            at1=1.5e9,
            tier2=1.0e9,
            min_total_cap_ratio=0.08,
            ccb=0.025,
            ccyb=0.0,
            dsib=0.0,
            pillar2=0.0,
            use_stress_addon=True,
            stress_multiplier=0.10
        ),
    }
    return run_capital_scenarios(df, scenarios)



def print_capital(results):
    for scen, tables in results.items():
        print(f"\n=== {scen} : OVERALL ===")
        print(tables["overall"].round(4).to_string(index=False))
        if not tables["by_asset_class"].empty:
            print(f"\n--- Top RWA by RegulatoryAssetClass ({scen}) ---")
            print(tables["by_asset_class"].head(10).round(4).to_string(index=False))
        if not tables["by_exposure_group"].empty:
            print(f"\n--- Top RWA by ExposureGroup ({scen}) ---")
            print(tables["by_exposure_group"].head(10).round(4).to_string(index=False))



def main():
    setup_logging()
    rng = np.random.default_rng()
    n = int(rng.integers(500_000, 1_000_001))
    cfg = Config(n_records=n, use_ifrs9_style=True)
    t0 = time.time()
    df = run_pipeline(cfg)
    t1 = time.time()
    logging.info(f"pipeline_elapsed {t1 - t0:.2f}s")
    sense_check(df)
    write_sources(df)
    # m0 = time.time()
    # results = train_all_models(df)
    # m1 = time.time()
    # logging.info(f"models_elapsed {m1 - m0:.2f}s")
    # print_models_summary(results)
    # print_top_scores(df, results, n=5)
    # results_adv = _run_advanced_with_banner(df)
    # print_adv_summary(results_adv)
    # cap_res = run_cap(df)
    # print_capital(cap_res)
    return 0
