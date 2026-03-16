from __future__ import annotations
from typing import Dict
import pandas as pd
from SimBank.features.capital import CapitalParams, capital_summary


def run_capital_scenarios(df: pd.DataFrame, scenarios: Dict[str, CapitalParams]) -> Dict[str, Dict[str, pd.DataFrame]]:
    results = {}
    for name, params in scenarios.items():
        results[name] = capital_summary(df, params=params)
    return results
