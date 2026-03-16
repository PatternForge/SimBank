import pandas as pd
import numpy as np


def generate_simulated_parameters(df):

    rng = np.random.default_rng()
    n = len(df)

    is_loan = df['AccountType'].isin(['Retail Loan', 'Business Loan'])
    is_deposit = df['AccountType'].isin(['Retail Deposit', 'Business Deposit'])
    is_business = df['AccountType'].isin(['Business Loan', 'Business Deposit'])
    has_arrears = df['ArrearsAmount'] > 0

    df_sp = df[['RunTimestamp', 'AccountID', 'AccountType']].copy()

    # ARREARS
    df_sp['ArrearsFlag'] = np.where(is_loan & has_arrears, 'Y', np.where(is_loan, 'N', None))
    df_sp['ArrearsPct'] = np.where(df_sp['ArrearsFlag'] == 'Y', rng.uniform(0.01, 0.15, n), None)

    # PROVISION
    df_sp['ProvisionFlag'] = np.where(is_loan & (df['AccountProvision'] > 0), 'Y', np.where(is_loan, 'N', None))
    df_sp['ProvisionPct'] = np.where(df_sp['ProvisionFlag'] == 'Y', rng.uniform(0.01, 0.20, n), None)

    # CASHBACK
    df_sp['CashbackFlag'] = np.where(rng.random(n) < 0.10, 'Y', 'N')
    df_sp['CashbackPct'] = np.where(df_sp['CashbackFlag'] == 'Y', rng.uniform(0.005, 0.02, n), None)

    # ADVANCE
    df_sp['AdvancePct'] = np.where(is_loan, rng.uniform(0.005, 0.03, n), None)

    # COLLATERAL
    df_sp['CollateralPct'] = np.where(is_loan, rng.uniform(0.60, 0.95, n), None)

    # FUNDING COST
    df_sp['FundingCostPct'] = rng.uniform(0.01, 0.04, n)

    # FEES & OPERATIONAL
    df_sp['FeesAmountPct'] = rng.uniform(0.001, 0.01, n)
    df_sp['OperationalCostPct'] = rng.uniform(0.001, 0.008, n)

    # DEBT SERVICE & DEV COST — business loans only
    df_sp['DebtServiceRatio'] = np.where(is_business, rng.uniform(1.20, 3.50, n), None)
    df_sp['DevCostRatio'] = np.where(is_business, rng.uniform(0.05, 0.25, n), None)

    # REVIEW/RATING/INSURANCE DATES — business loans only
    base_date = pd.Timestamp('today').normalize()
    df_sp['AnnualReviewDate'] = np.where(
        is_business,
        pd.to_datetime(base_date + pd.to_timedelta(rng.integers(1, 365, n), unit='D')).strftime('%Y-%m-%d'),
        None
    )
    df_sp['RatingDate'] = np.where(
        is_business,
        pd.to_datetime(base_date - pd.to_timedelta(rng.integers(1, 365, n), unit='D')).strftime('%Y-%m-%d'),
        None
    )
    df_sp['InsuranceExpiryDate'] = np.where(
        is_business,
        pd.to_datetime(base_date + pd.to_timedelta(rng.integers(30, 730, n), unit='D')).strftime('%Y-%m-%d'),
        None
    )

    # MONTHLY DEPOSIT FREQUENCY — deposits only
    df_sp['MonthlyDepositFrequency'] = np.where(is_deposit, rng.integers(1, 30, n), None)

    return df_sp.drop(columns=['AccountType'])
