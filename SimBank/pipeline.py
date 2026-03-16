import logging
from SimBank.utils.seed import make_rng
from SimBank.utils.perf import time_it
from SimBank.utils.dtype import optimize_dtypes
from SimBank.generators.base_snapshot import build_base_snapshot
from SimBank.generators.linkages import add_linkages
from SimBank.generators.collateral import add_collateral
from SimBank.features.amortization import add_amortization
from SimBank.features.arrears_provision import add_arrears_and_provisions
from SimBank.features.exposures import add_exposures
from SimBank.features.ftp_rates import add_interest_types_and_ftp
from SimBank.features.regulatory import add_regulatory_fields
from SimBank.features.ecl import add_ecl
from SimBank.features.portfolio_enrichment import add_portfolio_enrichment
from SimBank.features.profitability import add_profitability
from SimBank.features.stress import add_stress
from SimBank.validators.schema import validate_required_columns
from SimBank.validators.business_rules import validate_lvr_bounds
from SimBank.features.backfill_original import add_backfill_original


@time_it("pipeline")
def run_pipeline(cfg):
    rng = make_rng(cfg.random_seed)
    df = build_base_snapshot(cfg, rng)
    df = add_linkages(cfg, df, rng)
    df = add_collateral(cfg, df, rng)
    df = add_amortization(cfg, df, rng)
    df = add_arrears_and_provisions(cfg, df, rng)
    df = add_exposures(cfg, df, rng)
    df = add_interest_types_and_ftp(cfg, df, rng)
    df = add_regulatory_fields(cfg, df, rng)
    df = add_ecl(cfg, df, rng)
    df = add_portfolio_enrichment(cfg, df, rng)
    df = add_profitability(cfg, df, rng)
    df = add_stress(cfg, df, rng)
    df = add_backfill_original(cfg, df, rng)
    df = optimize_dtypes(cfg, df)
    validate_required_columns(df)
    validate_lvr_bounds(cfg, df)
    logging.info(f"done {df.shape}")
    return df

