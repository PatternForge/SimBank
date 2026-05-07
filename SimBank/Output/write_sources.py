import logging
from datetime import datetime
from pathlib import Path
from SimBank.generators.simulated_parameters import generate_simulated_parameters

def write_sources(df):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    run_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = REPO_ROOT / "Orchestrator" / "sources" / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["RunTimestamp"] = run_ts

    retail_loans_cols = [
        "RunTimestamp","DateOfPortfolio","AccountID","NumberID","AccountSource","AccountType",
        "CustomerID","OffsetFlag","LinkedDepositAccount","LinkedDepositBalance","AccountBalance",
        "MarketValue","InterestRate","CreditScore","CollateralCategory","AmortizationType",
        "DateOfOrigination","DateOfSettlement","DateOfMaturity","ArrearsAmount","ArrearsDays",
        "AccountProvision","Age","GeographicRegion","Currency","FullName"
    ]

    retail_deposits_cols = [
        "RunTimestamp","DateOfPortfolio","AccountID","NumberID","AccountSource","AccountType",
        "CustomerID","LinkedLoanAccount","LinkedLoanBalance","AccountBalance","InterestRate",
        "CreditScore","DateOfOrigination","Age","GeographicRegion","Currency","FullName"
    ]

    business_loans_cols = [
        "RunTimestamp","DateOfPortfolio","AccountID","NumberID","AccountSource","AccountType",
        "CustomerID","AccountBalance","MarketValue","InterestRate","CreditScore",
        "CollateralCategory","AmortizationType","DateOfOrigination","DateOfSettlement",
        "DateOfMaturity","ArrearsAmount","ArrearsDays","AccountProvision","Age",
        "GeographicRegion","Currency","FullName"
    ]

    business_deposits_cols = [
        "RunTimestamp","DateOfPortfolio","AccountID","NumberID","AccountSource","AccountType",
        "CustomerID","AccountBalance","InterestRate","CreditScore","DateOfOrigination",
        "Age","GeographicRegion","Currency","FullName"
    ]

    ftp_inputs_cols = [
        "RunTimestamp","AccountID","DateOfPortfolio","InterestRateType","BaseRate","AddonRate",
        "BasisCost","LiquidityRate","FundingIndex"
    ]

    stress_inputs_cols = [
        "RunTimestamp","AccountID","DateOfPortfolio","StressScore","MacroVolatilityIndex",
        "WithdrawalHistory"
    ]

    fees_costs_cols = [
        "RunTimestamp","AccountID","DateOfPortfolio","FeesCharged","FundingCost","OperationalCost"
    ]

    sim_params_cols = [
        "RunTimestamp","AccountID","ArrearsFlag","ArrearsPct","ProvisionFlag","ProvisionPct",
        "CashbackFlag","CashbackPct","AdvancePct","CollateralPct","FundingCostPct","FeesAmountPct",
        "OperationalCostPct","MonthlyDepositFrequency","DebtServiceRatio","DevCostRatio", "AnnualReviewDate",
        "RatingDate", "InsuranceExpiryDate"
    ]

    df[df["AccountType"] == "Retail Loan"][retail_loans_cols].to_csv(run_dir / "retail_loans.csv", index=False)
    df[df["AccountType"] == "Retail Deposit"][retail_deposits_cols].to_csv(run_dir / "retail_deposits.csv", index=False)
    df[df["AccountType"] == "Business Loan"][business_loans_cols].to_csv(run_dir / "business_loans.csv", index=False)
    df[df["AccountType"] == "Business Deposit"][business_deposits_cols].to_csv(run_dir / "business_deposits.csv", index=False)

    df[ftp_inputs_cols].to_csv(run_dir / "ftp_inputs.csv", index=False)
    df[stress_inputs_cols].to_csv(run_dir / "stress_inputs.csv", index=False)
    df[fees_costs_cols].to_csv(run_dir / "fees_costs.csv", index=False)

    sp = generate_simulated_parameters(df)[sim_params_cols]
    sp.to_csv(run_dir / "simulated_parameters.csv", index=False)

    logging.info(f"source files written to {run_dir}/")
