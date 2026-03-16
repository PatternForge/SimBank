from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd


@dataclass
class CapitalParams:
    cet1: float
    at1: float
    tier2: float
    min_total_cap_ratio: float = 0.08
    ccb: float = 0.025
    ccyb: float = 0.0
    dsib: float = 0.0
    pillar2: float = 0.0
    use_stress_addon: bool = False
    stress_multiplier: float = 0.0
    leverage_exposure_cols: Tuple[str, ...] = ("AccountPrincipal",)



def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df.get(col, default), errors="coerce").fillna(default)



def compute_rwa_components(df: pd.DataFrame) -> pd.DataFrame:
    ead_on = _safe_series(df, "ExposureAtDefault", 0.0).clip(lower=0.0)
    off_bal = _safe_series(df, "OffBalAmount", 0.0).clip(lower=0.0)
    ccf = _safe_series(df, "CreditConversionFactor", 0.0).clip(lower=0.0, upper=1.0)
    rw = _safe_series(df, "RiskWeight", 0.5).clip(lower=0.0)
    ead_off = (off_bal * ccf).astype(float)
    rwa_on = ead_on * rw
    rwa_off = ead_off * rw
    rwa_total = rwa_on + rwa_off
    out = pd.DataFrame({
        "EAD_On": ead_on,
        "EAD_Off": ead_off,
        "RiskWeight": rw,
        "RWA_On": rwa_on,
        "RWA_Off": rwa_off,
        "RWA_Total_Computed": rwa_total
    }, index=df.index)
    for col in ("OnBalRWA", "OffBalRWA", "TotalRWA"):
        if col in df.columns:
            out[col] = _safe_series(df, col, 0.0)
    return out



def capital_summary(df: pd.DataFrame, params: CapitalParams, group_bys: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    calc = compute_rwa_components(df)
    rwa = calc["RWA_Total_Computed"].copy()
    if params.use_stress_addon and params.stress_multiplier > 0:
        rwa = rwa * (1.0 + params.stress_multiplier)
    total_rwa = float(rwa.sum())
    total_required_ratio = float(params.min_total_cap_ratio + params.ccb + params.ccyb + params.dsib + params.pillar2)
    required_capital_amt = total_rwa * total_required_ratio
    required_min_amt = total_rwa * params.min_total_cap_ratio
    cet1_ratio = params.cet1 / total_rwa if total_rwa > 0 else np.nan
    tier1_capital = float(params.cet1 + params.at1)
    total_capital = float(params.cet1 + params.at1 + params.tier2)
    tier1_ratio = tier1_capital / total_rwa if total_rwa > 0 else np.nan
    total_cap_ratio = total_capital / total_rwa if total_rwa > 0 else np.nan
    lev_exposure = 0.0
    for c in params.leverage_exposure_cols:
        lev_exposure += float(_safe_series(df, c, 0.0).abs().sum())
    lev_exposure += float(_safe_series(df, "OffBalAmount", 0.0).clip(lower=0.0).sum())
    leverage_ratio = tier1_capital / lev_exposure if lev_exposure > 0 else np.nan
    overall = pd.DataFrame([{
        "Total_RWA": total_rwa,
        "Min_Capital_Ratio": params.min_total_cap_ratio,
        "Buffers_Ratio": total_required_ratio - params.min_total_cap_ratio,
        "Total_Req_Ratio": total_required_ratio,
        "Required_Capital_Amount": required_capital_amt,
        "Available_CET1": params.cet1,
        "Available_Tier1": tier1_capital,
        "Available_TotalCapital": total_capital,
        "CET1_Ratio": cet1_ratio,
        "Tier1_Ratio": tier1_ratio,
        "Total_Capital_Ratio": total_cap_ratio,
        "Leverage_Exposure_Proxy": lev_exposure,
        "Leverage_Ratio": leverage_ratio
    }])

    def _breakdown(by_col: str) -> pd.DataFrame:
        base = df[[by_col]].join(calc[["EAD_On", "EAD_Off", "RWA_On", "RWA_Off"]])
        grp = (
            base
            .assign(RWA_Total=(base["RWA_On"] + base["RWA_Off"]))
            .groupby(by_col, dropna=False, observed=False)[["EAD_On", "EAD_Off", "RWA_On", "RWA_Off", "RWA_Total"]]
            .sum()
            .sort_values("RWA_Total", ascending=False)
            .reset_index()
        )
        grp["RWA_Density"] = grp["RWA_Total"] / (grp["EAD_On"] + grp["EAD_Off"]).replace(0.0, np.nan)
        return grp

    by_asset_class = _breakdown("RegulatoryAssetClass") if "RegulatoryAssetClass" in df.columns else pd.DataFrame()
    by_exposure_group = _breakdown("ExposureGroup") if "ExposureGroup" in df.columns else pd.DataFrame()
    by_geography = _breakdown("GeographicRegion") if "GeographicRegion" in df.columns else pd.DataFrame()
    by_vintage = _breakdown("Vintage") if "Vintage" in df.columns else pd.DataFrame()
    out = {
        "overall": overall,
        "by_asset_class": by_asset_class,
        "by_exposure_group": by_exposure_group,
        "by_geography": by_geography,
        "by_vintage": by_vintage
    }
    return out
