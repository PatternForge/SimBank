import os
import logging
from datetime import datetime
from SimBank.generators.simulated_parameters import generate_simulated_parameters


def write_sources(df, output_dir="sources"):
    run_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = f"{output_dir}/{run_ts}"
    os.makedirs(run_dir, exist_ok=True)

    df = df.copy()
    df["RunTimestamp"] = run_ts

    # Retail Loans
    rl_cols = [
        "RunTimestamp", "DateOfPortfolio", "AccountID", "NumberID", "AccountSource",
        "AccountType", "CustomerID", "OffsetFlag", "LinkedDepositAccount",
        "LinkedDepositBalance", "AccountBalance", "MarketValue",
        "InterestRate", "CreditScore", "CollateralCategory",
        "AmortizationType", "DateOfOrigination", "DateOfSettlement",
        "DateOfMaturity", "ArrearsAmount", "ArrearsDays",
        "AccountProvision", "Age", "GeographicRegion", "Currency", "FullName"
    ]
    df[df["AccountType"] == "Retail Loan"][[c for c in rl_cols if c in df.columns]].to_csv(
        f"{run_dir}/retail_loans.csv", index=False)

    # Retail Deposits
    rd_cols = [
        "RunTimestamp", "DateOfPortfolio", "AccountID", "NumberID", "AccountSource",
        "AccountType", "CustomerID", "LinkedLoanAccount", "LinkedLoanBalance",
        "AccountBalance", "InterestRate", "CreditScore",
        "DateOfOrigination", "Age", "GeographicRegion", "Currency", "FullName"
    ]
    df[df["AccountType"] == "Retail Deposit"][[c for c in rd_cols if c in df.columns]].to_csv(
        f"{run_dir}/retail_deposits.csv", index=False)

    # Business Loans
    bl_cols = [
        "RunTimestamp", "DateOfPortfolio", "AccountID", "NumberID", "AccountSource",
        "AccountType", "CustomerID", "AccountBalance", "MarketValue",
        "InterestRate", "CreditScore", "CollateralCategory",
        "AmortizationType", "DateOfOrigination", "DateOfSettlement",
        "DateOfMaturity", "ArrearsAmount", "ArrearsDays",
        "AccountProvision", "Age", "GeographicRegion", "Currency", "FullName"
    ]
    df[df["AccountType"] == "Business Loan"][[c for c in bl_cols if c in df.columns]].to_csv(
        f"{run_dir}/business_loans.csv", index=False)

    # Business Deposits
    bd_cols = [
        "RunTimestamp", "DateOfPortfolio", "AccountID", "NumberID", "AccountSource",
        "AccountType", "CustomerID", "AccountBalance", "InterestRate",
        "CreditScore", "DateOfOrigination", "Age",
        "GeographicRegion", "Currency", "FullName"
    ]
    df[df["AccountType"] == "Business Deposit"][[c for c in bd_cols if c in df.columns]].to_csv(
        f"{run_dir}/business_deposits.csv", index=False)

    # FTP Inputs — Treasury system
    ftp_cols = [
        "RunTimestamp", "AccountID", "DateOfPortfolio", "InterestRateType",
        "BaseRate", "AddonRate", "BasisCost", "LiquidityRate",
        "FundingIndex"
    ]
    df[[c for c in ftp_cols if c in df.columns]].to_csv(f"{run_dir}/ftp_inputs.csv", index=False)

    # Stress Inputs — Risk system
    stress_cols = [
        "RunTimestamp", "AccountID", "DateOfPortfolio",
        "StressScore", "MacroVolatilityIndex",
        "WithdrawalHistory", "MonthlyDepositFrequency"
    ]
    df[[c for c in stress_cols if c in df.columns]].to_csv(f"{run_dir}/stress_inputs.csv", index=False)

    # Fees & Costs — Finance system
    fees_cols = [
        "RunTimestamp", "AccountID", "DateOfPortfolio",
        "FeesCharged", "FundingCost", "OperationalCost"
    ]
    df[[c for c in fees_cols if c in df.columns]].to_csv(f"{run_dir}/fees_costs.csv", index=False)

    # Simulated Parameters
    sp_cols = generate_simulated_parameters(df)
    sp_cols.to_csv(f"{run_dir}/simulated_parameters.csv", index=False)

    logging.info(f"source files written to {run_dir}/")
