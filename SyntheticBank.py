import pandas as pd
import numpy as np
import time
from pandas.tseries.offsets import BDay, DateOffset
from scipy.stats import skewnorm

start_time = time.time()
num_records = np.random.randint(500000, 1000000)
today = pd.Timestamp.today().normalize()

AccountID = np.random.permutation(np.arange(1, (num_records * 3) + 1))[:num_records]
AccountType = np.random.choice(['Business Loan', 'Business Deposit', 'Retail Loan', 'Retail Deposit'], p=[0.03, 0.07, 0.05, 0.85], size=num_records)
TypeOfAccount = np.where(AccountType == 'Business Loan', 'Loan',
                         np.where(AccountType == 'Retail Loan', 'Loan',
                                  'Deposit'))
AccountSource = np.where(AccountType == 'Business Loan', 'BBK',
                         np.where(AccountType == 'Business Deposit', 'BBK',
                                  'RTE'))

SyntheticData = pd.DataFrame({
    'DateOfPortfolio': today - BDay(1),
    'AccountID': AccountID,
    'AccountType': AccountType,
    'NumberID': AccountID.astype(str) + AccountSource.astype(str),
    'TypeOfAccount': TypeOfAccount,
    'AccountSource': AccountSource
})

retail_loan_mask = SyntheticData['AccountType'] == 'Retail Loan'
retail_deposit_mask = SyntheticData['AccountType'] == 'Retail Deposit'
business_loan_mask = SyntheticData['AccountType'] == 'Business Loan'
business_deposit_mask = SyntheticData['AccountType'] == 'Business Deposit'

SyntheticData['AccountBalance'] = 0.0
SyntheticData.loc[retail_loan_mask, 'AccountBalance'] = skewnorm.rvs(a=4, loc=650000, scale=150000, size=retail_loan_mask.sum())
SyntheticData.loc[retail_deposit_mask, 'AccountBalance'] = -1 * skewnorm.rvs(a=-3, loc=10000, scale=8000, size=retail_deposit_mask.sum())
SyntheticData.loc[business_loan_mask, 'AccountBalance'] = skewnorm.rvs(a=3, loc=1500000, scale=900000, size=business_loan_mask.sum())
SyntheticData.loc[business_deposit_mask, 'AccountBalance'] = -1 * skewnorm.rvs(a=-2, loc=90000, scale=45000, size=business_deposit_mask.sum())
deposit_mask = SyntheticData['TypeOfAccount'] == 'Deposit'
SyntheticData.loc[deposit_mask, 'AccountBalance'] = SyntheticData.loc[deposit_mask, 'AccountBalance'].clip(upper=0)
SyntheticData['AccountBalance'] = SyntheticData['AccountBalance'].round(2)
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
multiplier = np.random.uniform(1.15, 2.5, size=len(SyntheticData))
SyntheticData['MarketValue'] = np.where(loan_mask, np.round(multiplier * abs(SyntheticData['AccountBalance']), 2), np.nan)
SyntheticData['InterestRate'] = np.where(loan_mask, np.clip(skewnorm.rvs(a=-4, loc=4.0, scale=2.0, size=len(SyntheticData)), 0.1, 9.5),
                                         np.clip(skewnorm.rvs(a=-4, loc=1.5, scale=1.0, size=len(SyntheticData)), 0.1, 4.0))
SyntheticData['InterestRate'] = SyntheticData['InterestRate'].round(2)
SyntheticData['CreditScore'] = np.random.randint(300, 999, size=num_records)
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
SyntheticData['LVR'] = np.where(loan_mask, SyntheticData['AccountBalance'] / SyntheticData['MarketValue'], np.nan)
business_loan_mask = SyntheticData['AccountType'] == 'Business Loan'
SyntheticData.loc[business_loan_mask, 'LVR'] = SyntheticData.loc[business_loan_mask, 'LVR'].clip(upper=0.8)
SyntheticData['LVRRWA'] = (SyntheticData['LVR'] * 100).round(2)
SyntheticData['LMIFlag'] = np.where(SyntheticData['LVRRWA'] > 80.00, 'Y', 'N').astype(str)
bins = list(np.arange(0, 101, 5)) + [float('inf')]
labels = [
    '0 to <= 5%', '5 to <= 10%', '10 to <= 15%', '15 to <= 20%', '20 to <= 25%', '25 to <= 30%',
    '30 to <= 35%', '35 to <= 40%', '40 to <= 45%', '45 to <= 50%', '50 to <= 55%', '55 to <= 60%',
    '60 to <= 65%', '65 to <= 70%', '70 to <= 75%', '75 to <= 80%', '80 to <= 85%', '85 to <= 90%',
    '90 to <= 95%', '95 to <= 100%', '>100%'
]
SyntheticData['LVRBand'] = pd.cut(SyntheticData['LVRRWA'], bins=bins, labels=labels, right=True).astype(str)
retail_loan_mask = SyntheticData['AccountType'] == 'Retail Loan'
num_offsets = (0.35 * retail_loan_mask.sum()).astype(int)
SyntheticData['OffsetFlag'] = 'N'
offset_indices = SyntheticData[retail_loan_mask].sample(n=num_offsets, random_state=42).index
SyntheticData.loc[offset_indices, 'OffsetFlag'] = 'Y'
SyntheticData['Age'] = np.random.randint(19, 66, size=num_records)

letters = np.array(list('abcdefghijklmnopqrstuvwxyz'))
first_len = np.random.randint(4, 8, size=num_records)
last_len = np.random.randint(4, 12, size=num_records)
max_first, max_last = 8, 12
first_arr = np.random.choice(letters, (num_records, max_first))
last_arr = np.random.choice(letters, (num_records, max_last))
first_mask = np.arange(max_first) < first_len[:, None]
last_mask = np.arange(max_last) < last_len[:, None]
first_arr = np.where(first_mask, first_arr, ' ')
last_arr = np.where(last_mask, last_arr, ' ')
first_arr_fixed = first_arr.astype(f'U{max_first}')
last_arr_fixed = last_arr.astype(f'U{max_last}')
first_names = np.array([''.join(row).strip().capitalize() for row in first_arr])
last_names = np.array([''.join(row).strip().capitalize() for row in last_arr])
SyntheticData.loc[:, 'FullName'] = first_names + ' ' + last_names

SyntheticData['DateOfPortfolio'] = pd.to_datetime(SyntheticData['DateOfPortfolio'])
SyntheticData['DateOfSettlement'] = pd.NaT
SyntheticData['DateOfMaturity'] = pd.NaT
SyntheticData['DateOfOrigination'] = pd.NaT
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
birth_dates = SyntheticData['DateOfPortfolio'] - pd.to_timedelta(SyntheticData['Age'] * 365, unit='D')
min_loan_age = 18
max_loan_term_days = 30 * 365
earliest_loan_dates = birth_dates + pd.to_timedelta(min_loan_age * 365, unit='D')
earliest_loan_dates_clipped = SyntheticData['DateOfPortfolio'] - pd.to_timedelta(max_loan_term_days, unit='D')
valid_loan_start_dates = pd.DataFrame({
    'earliest': earliest_loan_dates[loan_mask],
    'clipped': earliest_loan_dates_clipped[loan_mask]
}).max(axis=1)
loan_days_range = (SyntheticData.loc[loan_mask, 'DateOfPortfolio'] - valid_loan_start_dates).dt.days.clip(lower=1)
random_loan_days = np.random.randint(0, loan_days_range.values, size=loan_days_range.shape[0])
loan_settlement_dates = valid_loan_start_dates + pd.to_timedelta(random_loan_days, unit='D')
SyntheticData.loc[loan_mask, 'DateOfOrigination'] = loan_settlement_dates
SyntheticData.loc[loan_mask, 'DateOfSettlement'] = SyntheticData.loc[loan_mask, 'DateOfOrigination']
SyntheticData.loc[loan_mask, 'DateOfMaturity'] = SyntheticData.loc[loan_mask, 'DateOfSettlement'] + DateOffset(years=30)
deposit_mask = SyntheticData['TypeOfAccount'] == 'Deposit'
earliest_deposit_dates = birth_dates + pd.to_timedelta(12 * 365, unit='D')
deposit_days_range = (SyntheticData.loc[deposit_mask, 'DateOfPortfolio'] - earliest_deposit_dates[deposit_mask]).dt.days.clip(lower=1)
random_deposit_days = np.random.randint(0, deposit_days_range.values, size=deposit_days_range.shape[0])
deposit_origination_dates = earliest_deposit_dates[deposit_mask] + pd.to_timedelta(random_deposit_days, unit='D')
SyntheticData.loc[deposit_mask, 'DateOfOrigination'] = deposit_origination_dates
SyntheticData.loc[deposit_mask, 'DateOfMaturity'] = SyntheticData.loc[deposit_mask, 'DateOfPortfolio'] + pd.Timedelta(days=1)
SyntheticData['DaysSinceSettlement'] = (SyntheticData['DateOfPortfolio'] - SyntheticData['DateOfSettlement']).dt.days
SyntheticData['DaysSinceOrigination'] = (SyntheticData['DateOfPortfolio'] - SyntheticData['DateOfOrigination']).dt.days
SyntheticData['DaysUntilMaturity'] = (SyntheticData['DateOfMaturity'] - SyntheticData['DateOfPortfolio']).dt.days

SyntheticData['CollateralCategory'] = None
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
bb_mask = loan_mask & (SyntheticData['AccountSource'] == 'BBK')
num_bb = bb_mask.sum()
bb_categories = np.random.choice(['Agri', 'Commercial', 'Dev Finance', 'PI', 'Other'], p=[0.51, 0.32, 0.01, 0.11, 0.05], size=num_bb)
SyntheticData.loc[bb_mask, 'CollateralCategory'] = bb_categories
SyntheticData.loc[loan_mask & ~bb_mask, 'CollateralCategory'] = 'Residential Property'

deposit_mask = SyntheticData['TypeOfAccount'] == 'Deposit'
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
offset_mask = SyntheticData['OffsetFlag'] == 'Y'
non_offset_loan_mask = loan_mask & ~offset_mask
offset_loan_mask = loan_mask & offset_mask
num_deposits = deposit_mask.sum()
num_non_offset_loans = non_offset_loan_mask.sum()
unique_ids = np.arange(1, num_deposits + num_non_offset_loans + 1)
np.random.shuffle(unique_ids)
SyntheticData.loc[deposit_mask, 'CustomerID'] = unique_ids[:num_deposits]
SyntheticData.loc[non_offset_loan_mask, 'CustomerID'] = unique_ids[num_deposits:]
deposit_customer_ids = SyntheticData.loc[deposit_mask, 'CustomerID'].values
offset_customer_ids = np.random.choice(deposit_customer_ids, size=offset_loan_mask.sum())
SyntheticData.loc[offset_loan_mask, 'CustomerID'] = offset_customer_ids
SyntheticData['CustomerID'] = SyntheticData['CustomerID'].astype(int)

loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
deposit_mask = SyntheticData['TypeOfAccount'] == 'Deposit'
offset_loan_mask = loan_mask & (SyntheticData['OffsetFlag'] == 'Y')
linked_deposit_pool = SyntheticData.loc[deposit_mask, ['NumberID', 'AccountBalance']]
selected_deposits = linked_deposit_pool.sample(n=offset_loan_mask.sum(), random_state=42).reset_index(drop=True)
SyntheticData.loc[offset_loan_mask, 'LinkedDepositAccount'] = selected_deposits['NumberID'].values
SyntheticData.loc[offset_loan_mask, 'LinkedDepositBalance'] = selected_deposits['AccountBalance'].values
offset_loan_customers = SyntheticData.loc[offset_loan_mask, ['CustomerID', 'NumberID', 'AccountBalance']]
deposit_linked_mask = deposit_mask & SyntheticData['CustomerID'].isin(offset_loan_customers['CustomerID'])
linked_loans = offset_loan_customers.rename(columns={
    'NumberID': 'LinkedLoanAccount',
    'AccountBalance': 'LinkedLoanBalance'
})
deposits_with_links = SyntheticData.loc[deposit_linked_mask].merge(linked_loans, on='CustomerID', how='left')
SyntheticData.loc[deposits_with_links.index, 'LinkedLoanAccount'] = deposits_with_links['LinkedLoanAccount'].values
SyntheticData.loc[deposits_with_links.index, 'LinkedLoanBalance'] = deposits_with_links['LinkedLoanBalance'].values

SyntheticData['Term'] = np.where(SyntheticData['TypeOfAccount'] == 'Deposit', 0, (SyntheticData['DaysUntilMaturity'] / 30).round().astype(int))
SyntheticData['AccountPrincipal'] = SyntheticData['AccountBalance'].abs()
SyntheticData['AnnualRate'] = SyntheticData['InterestRate'] / 100
SyntheticData['DailyRate'] = SyntheticData['AnnualRate'] / 365
SyntheticData['MonthlyRate'] = (1 + SyntheticData['AnnualRate']) ** (1 / 12) - 1

loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
bb_mask = loan_mask & (SyntheticData['AccountSource'] == 'BBK')
high_balance_mask = bb_mask & (SyntheticData['AccountBalance'] > 1000000)
low_balance_mask = loan_mask & ~high_balance_mask
SyntheticData['AmortizationType'] = None
SyntheticData.loc[high_balance_mask, 'AmortizationType'] = np.random.choice(
    ['InterestOnly', 'Bullet', 'P&I'],
    p=[0.6, 0.25, 0.15],
    size=high_balance_mask.sum()
)
SyntheticData.loc[low_balance_mask, 'AmortizationType'] = np.random.choice(
    ['InterestOnly', 'Bullet', 'P&I'],
    p=[0.2, 0.05, 0.75],
    size=low_balance_mask.sum()
)
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
SyntheticData['MonthlyRepayment'] = 0.0
SyntheticData.loc[loan_mask, 'MonthlyRepayment'] = (
        SyntheticData.loc[loan_mask, 'AccountPrincipal'] *
        ((1 + SyntheticData.loc[loan_mask, 'AnnualRate']) ** (1 / 12) - 1) *
        ((1 + ((1 + SyntheticData.loc[loan_mask, 'AnnualRate']) ** (1 / 12) - 1)) ** SyntheticData.loc[loan_mask, 'Term']) /
        (((1 + ((1 + SyntheticData.loc[loan_mask, 'AnnualRate']) ** (1 / 12) - 1)) ** SyntheticData.loc[loan_mask, 'Term']) - 1)
).round(2)
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
pi_mask = loan_mask & (SyntheticData['AmortizationType'] == 'P&I')
io_mask = loan_mask & (SyntheticData['AmortizationType'] == 'InterestOnly')
bullet_mask = loan_mask & (SyntheticData['AmortizationType'] == 'Bullet')
SyntheticData['MonthlyRepayment'] = 0.0
monthly_rate = (1 + SyntheticData['AnnualRate']) ** (1 / 12) - 1
SyntheticData.loc[pi_mask, 'MonthlyRepayment'] = (
        SyntheticData.loc[pi_mask, 'AccountPrincipal'] *
        monthly_rate[pi_mask] *
        (1 + monthly_rate[pi_mask]) ** SyntheticData.loc[pi_mask, 'Term'] /
        ((1 + monthly_rate[pi_mask]) ** SyntheticData.loc[pi_mask, 'Term'] - 1)
).round(2)
SyntheticData.loc[io_mask, 'MonthlyRepayment'] = (SyntheticData.loc[io_mask, 'AccountPrincipal'] * monthly_rate[io_mask]).round(2)
SyntheticData.loc[bullet_mask, 'MonthlyRepayment'] = 0.0

SyntheticData['DayOfMonth'] = SyntheticData['DateOfPortfolio'].dt.day
SyntheticData['AggregatedMonthlyBalance'] = SyntheticData['AccountBalance'] * SyntheticData['DayOfMonth']
SyntheticData['AverageMonthlyBalance'] = SyntheticData['AggregatedMonthlyBalance'] / SyntheticData['DayOfMonth']
SyntheticData['ArrearsAmount'] = 0.0
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
num_loans = loan_mask.sum()
num_arrears = int(0.0306 * num_loans)
arrears_indices = np.random.choice(SyntheticData[loan_mask].index, size=num_arrears, replace=False)
arrears_percentages = np.random.beta(a=0.5, b=2.0, size=num_arrears)
SyntheticData.loc[arrears_indices, 'ArrearsAmount'] = SyntheticData.loc[arrears_indices, 'AccountBalance'].abs() * arrears_percentages
SyntheticData['ArrearsDays'] = 0
SyntheticData.loc[loan_mask, 'ArrearsDays'] = np.random.randint(0, 181, size=num_loans)
SyntheticData.loc[SyntheticData['ArrearsAmount'] == 0, 'ArrearsDays'] = 0
SyntheticData['AccountProvision'] = 0.0
num_provisions = int(0.00039 * num_loans)
provision_indices = np.random.choice(SyntheticData[loan_mask].index, size=num_provisions, replace=False)
provision_percentages = np.random.beta(a=2.0, b=1.0, size=num_provisions)
provision_percentages = np.clip(provision_percentages, 0.05, 0.96)
SyntheticData.loc[provision_indices, 'AccountProvision'] = SyntheticData.loc[provision_indices, 'AccountBalance'].abs() * provision_percentages
SyntheticData['InterestAccrued'] = SyntheticData['AccountBalance'] * (SyntheticData['InterestRate'] / 100) * (SyntheticData['DaysSinceSettlement'] / 365)

SyntheticData['AdvanceAmount'] = SyntheticData['AccountBalance'] * np.random.uniform(0.05, 0.3, size=num_records)
negative_indices = np.random.choice(SyntheticData.index, size=int(0.05 * num_records), replace=False)
SyntheticData.loc[negative_indices, 'AdvanceAmount'] *= -1
SyntheticData['Limit'] = np.where((SyntheticData['TypeOfAccount'] == 'Loan'), SyntheticData['MarketValue'],
                                  0)
SyntheticData['AvailableBalance'] = (SyntheticData['Limit'] - SyntheticData['AccountBalance']).clip(lower=0)
account_balance = SyntheticData['AccountBalance'].abs()
available_balance = SyntheticData['AvailableBalance'].abs()
advance_amount = SyntheticData['AdvanceAmount'].clip(lower=0)
SyntheticData['TotalAccountExposure'] = 0.0
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
SyntheticData.loc[loan_mask, 'TotalAccountExposure'] = (
        available_balance[loan_mask] + account_balance[loan_mask] + advance_amount[loan_mask]
)
deposit_mask = SyntheticData['TypeOfAccount'] == 'Deposit'
SyntheticData.loc[deposit_mask, 'TotalAccountExposure'] = np.maximum(
    account_balance[deposit_mask], SyntheticData['Limit'][deposit_mask]
)

SyntheticData['AllocatedCashCollateral'] = 0.0
SyntheticData.loc[loan_mask, 'AllocatedCashCollateral'] = (SyntheticData.loc[loan_mask, 'AccountBalance'].abs() *
                                                           np.random.uniform(0.0, 0.3, size=loan_mask.sum()))
SyntheticData['InterestNotBoughtToAccount'] = SyntheticData['AccountBalance'] - SyntheticData['AllocatedCashCollateral']
SyntheticData['TotalGroupExposure'] = SyntheticData.groupby('CustomerID')['TotalAccountExposure'].transform('sum')
SyntheticData['TotalBusinessGroupExposure'] = np.where((SyntheticData['AccountSource'] == 'BBK'),
                                                       SyntheticData.groupby('CustomerID')['TotalAccountExposure'].transform('sum'), 0)

loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
SyntheticData['AssetClassAdvanced'] = np.where(loan_mask,
                                               np.where(SyntheticData['AccountType'].str.contains('Retail'), 'Residential Mortgage',
                                                        'Corporate'),
                                               'Other')
SyntheticData['ExposureNetCredit'] = np.where(loan_mask,
                                              np.where((SyntheticData['InterestAccrued'] + SyntheticData['AccountBalance'] < 0), 0,
                                                       np.where(SyntheticData['AccountBalance'] == 0, 0,
                                                                SyntheticData['InterestAccrued'] + SyntheticData['AccountBalance'])),
                                              0.0)
SyntheticData['ExposureAtDefault'] = np.where(SyntheticData['AccountBalance'] <= 0, 0,
                                              np.where(SyntheticData['InterestAccrued'] + SyntheticData['AccountBalance'] -
                                                       SyntheticData['InterestNotBoughtToAccount'] - SyntheticData['AllocatedCashCollateral'] <= 0, 0,
                                                       SyntheticData['InterestAccrued'] + SyntheticData['AccountBalance'] -
                                                       SyntheticData['InterestNotBoughtToAccount'] - SyntheticData['AllocatedCashCollateral']
                                                       ))
SyntheticData['CreditConversionFactor'] = np.where(loan_mask, 1.0, 0.0)
SyntheticData['InterestRateType'] = None
SyntheticData.loc[deposit_mask, 'InterestRateType'] = 'Variable'
SyntheticData.loc[loan_mask, 'InterestRateType'] = np.random.choice(['Variable', 'Fixed'], size=loan_mask.sum(), p=[0.96, 0.04])
SyntheticData['CashbackAmount'] = 0.0
cashback_mask = (loan_mask & (SyntheticData['InterestRateType'] == 'Variable'))
SyntheticData.loc[cashback_mask, 'CashbackAmount'] = np.random.uniform(1000, 5000, size=cashback_mask.sum())
others_mask = (~loan_mask) & (~deposit_mask)
SyntheticData['OffBalAmount'] = np.where(others_mask,
                                         np.where(SyntheticData['InterestRateType'] == 'Fixed', 0, SyntheticData['AvailableBalance']),
                                         np.where((SyntheticData['Limit'].notna()) & (SyntheticData['AccountBalance'] <= 0) &
                                                  ((SyntheticData['Limit'] + SyntheticData['AccountBalance']) > 0),
                                                  SyntheticData['Limit'] + SyntheticData['AccountBalance'],
                                                  np.where((SyntheticData['Limit'] > 0) & (SyntheticData['AccountBalance'] > 0),
                                                           SyntheticData['Limit'], 0)))
SyntheticData.loc[loan_mask, 'OffBalAmount'] = 0.0
SyntheticData.loc[deposit_mask, 'OffBalAmount'] = 0.0
SyntheticData['CapitalOffBalanceAmount'] = SyntheticData['OffBalAmount'].fillna(0) + SyntheticData['CashbackAmount'].fillna(0)
SyntheticData['OnBalanceExposureAmount'] = np.where((SyntheticData['InterestAccrued'] + SyntheticData['AccountBalance'] -
                                                     SyntheticData['InterestNotBoughtToAccount'] - SyntheticData['AccountProvision']) < 0, 0,
                                                    (SyntheticData['InterestAccrued'] + SyntheticData['AccountBalance'] -
                                                     SyntheticData['InterestNotBoughtToAccount'] - SyntheticData['AccountProvision']))
SyntheticData['LVRSource'] = np.where(SyntheticData['AccountType'] == 'Retail Loan', '0 - Resi LVR',
                                      np.where(SyntheticData['AccountSource'] == 'FWD', '1 - LVR Orig',
                                               np.where(SyntheticData['AccountType'] == 'Business Loan', '2 - BB CR',
                                                        np.where(SyntheticData['TypeOfAccount'] == 'Deposit', 'N/A', 'Missing'))))

SyntheticData['ImpairedFlag'] = SyntheticData['ArrearsDays'] > 90
m_imp = SyntheticData['ImpairedFlag']
m_resi = SyntheticData['CollateralCategory'].eq('Residential Property')
m_prov = SyntheticData['AccountProvision'] != 0
m_neg = (SyntheticData['AccountBalance'] - SyntheticData['InterestNotBoughtToAccount']) < 0
SyntheticData['DefaultedExposureClass'] = np.select([m_imp & m_resi, m_imp & ~m_resi & m_prov & m_neg, m_imp & ~m_resi & (~m_prov | ~m_neg), ],
                                                    ['Defaulted Resi Property',
                                                     'Other Defaulted Property > 0.2',
                                                     'Other Defaulted Property < 0.2', ],
                                                    default=np.array(np.nan, dtype=object))
SyntheticData['ExposureClass'] = 'N/A'
loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
res_mask = loan_mask & (SyntheticData['CollateralCategory'] == 'Residential Property')
standard_res_mask = res_mask & (SyntheticData['LVR'] <= 0.8) & (SyntheticData['AmortizationType'] == 'P&I')
SyntheticData.loc[standard_res_mask, 'ExposureClass'] = 'Standard Residential Mortgage'
SyntheticData.loc[res_mask & ~standard_res_mask, 'ExposureClass'] = 'Non-standard Residential Mortgage'
comm_mask = loan_mask & SyntheticData['CollateralCategory'].isin(['Commercial', 'Agri', 'Dev Finance'])
standard_comm_mask = comm_mask & (SyntheticData['LVR'] <= 0.6) & (SyntheticData['AmortizationType'] == 'P&I')
SyntheticData.loc[standard_comm_mask, 'ExposureClass'] = 'Standard Commercial Property'
SyntheticData.loc[comm_mask & ~standard_comm_mask, 'ExposureClass'] = 'Non-standard Commercial Property'
other_secured_mask = loan_mask & SyntheticData['CollateralCategory'].isin(['PI', 'Other'])
SyntheticData.loc[other_secured_mask, 'ExposureClass'] = 'Other Secured Exposure'
unsecured_mask = loan_mask & ~(res_mask | comm_mask | other_secured_mask)
SyntheticData.loc[unsecured_mask, 'ExposureClass'] = 'Unsecured Exposure'
SyntheticData['ExposureGroup'] = 'N/A'
SyntheticData.loc[res_mask, 'ExposureGroup'] = 'RESI'
SyntheticData.loc[comm_mask, 'ExposureGroup'] = 'COMM'
SyntheticData.loc[(loan_mask & SyntheticData['CollateralCategory'] == 'PI'), 'ExposureGroup'] = 'COMD'
SyntheticData.loc[loan_mask & (SyntheticData['AccountSource'] == 'BBK') & (SyntheticData['AccountBalance'] > 1000000), 'ExposureGroup'] = 'CORP'
SyntheticData.loc[(loan_mask & SyntheticData['AccountSource'] == 'RTE'), 'ExposureGroup'] = 'RETL'
SyntheticData.loc[(loan_mask & SyntheticData['CollateralCategory'] == 'Lease'), 'ExposureGroup'] = 'LEAS'
SyntheticData.loc[(loan_mask & SyntheticData['ExposureGroup'].isna()), 'ExposureGroup'] = 'UNKN'
SyntheticData['ExposureSubClass'] = np.where(
    (SyntheticData['TypeOfAccount'] != 'Loan') | (SyntheticData['AccountType'] == 'Retail Loan'), 'Not Applicable',
    np.where(SyntheticData['TotalBusinessGroupExposure'] > 5000000, 'General Corporate - Other',
             np.where(SyntheticData['TotalBusinessGroupExposure'] > 1500000, 'General Corporate - SME Corporate',
                      'General Corporate - SME Retail'))
)
SyntheticData['CreditExposureAmount'] = ((SyntheticData['OffBalAmount'].fillna(0) * SyntheticData['CreditConversionFactor']) +
                                         (SyntheticData['CashbackAmount'].fillna(0) * SyntheticData['CreditConversionFactor']))
SyntheticData['CreditExposureAmountCashback'] = SyntheticData['CashbackAmount'].fillna(0) * 0.40
SyntheticData['DevCostRatio'] = np.random.uniform(0.1, 0.9, size=num_records)
SyntheticData['SourceLVR'] = np.nan
SyntheticData.loc[loan_mask, 'SourceLVR'] = (SyntheticData['AccountBalance'] / SyntheticData['Limit'].replace(0, np.nan)).fillna(1.0).clip(upper=1.5)
SyntheticData['RiskWeight'] = np.nan
m_loan = loan_mask
m_retl = SyntheticData['ExposureGroup'].eq('RETL')
m_corp = SyntheticData['ExposureGroup'].eq('CORP')
m_lmi = SyntheticData['LMIFlag'].eq('Y')
SyntheticData['RiskWeight'] = np.select([m_loan & m_retl & m_lmi, m_loan & m_retl & ~m_lmi, m_loan & m_corp, m_loan & ~(m_retl | m_corp), ],
                                        [0.50, 0.75, 1.00, 0.50, ], default=np.nan).astype(float)
SyntheticData['RiskWeightedAssetLVR'] = SyntheticData['CreditExposureAmount'] * SyntheticData['RiskWeight']
SyntheticData['OnBalRWA'] = SyntheticData['CreditExposureAmount'] * SyntheticData['RiskWeight']
SyntheticData['OffBalRWA'] = (SyntheticData['ExposureAtDefault'] - SyntheticData['CreditExposureAmount']) * SyntheticData['RiskWeight']

SyntheticData['RepaymentAmount'] = 0.0
SyntheticData.loc[loan_mask, 'RepaymentAmount'] = SyntheticData['AccountPrincipal'] / SyntheticData['Term']
retl_mask = loan_mask & SyntheticData['AccountSource'].eq('RTE')
corp_mask = loan_mask & SyntheticData['AccountSource'].eq('BBK')
SyntheticData['DebtServiceRatio'] = np.nan
SyntheticData.loc[retl_mask, 'DebtServiceRatio'] = (0.25 + 0.20 * np.random.rand(retl_mask.sum()))
SyntheticData.loc[corp_mask, 'DebtServiceRatio'] = (0.20 + 0.20 * np.random.rand(corp_mask.sum()))
SyntheticData['DebtServiceRatio'] = SyntheticData['DebtServiceRatio'].fillna(0.30)
SyntheticData['MonthlyIncome'] = SyntheticData['RepaymentAmount'] / SyntheticData['DebtServiceRatio']
SyntheticData.loc[retl_mask, 'MonthlyIncome'] = SyntheticData.loc[retl_mask, 'MonthlyIncome'].clip(upper=60000)
SyntheticData['AnnualIncome'] = SyntheticData['MonthlyIncome'] * 12
SyntheticData['LoanToIncome'] = SyntheticData['AccountPrincipal'] / SyntheticData['AnnualIncome']
SyntheticData['EstimatedLivingExpenses'] = np.where(res_mask, 2400, np.nan)
SyntheticData['NetDisposableIncome'] = np.where(loan_mask, (SyntheticData['MonthlyIncome'] - SyntheticData['RepaymentAmount'] -
                                                            SyntheticData['EstimatedLivingExpenses']), np.nan)
SyntheticData['AffordabilityFlag'] = np.where(loan_mask & (SyntheticData['NetDisposableIncome'] > 500), 'Comfortable',
                                              np.where(loan_mask & (SyntheticData['NetDisposableIncome'] > 0), 'Tight',
                                                       np.where(loan_mask & (SyntheticData['NetDisposableIncome'] < 0), 'Unaffordable', 'N/A')))

rate_ranges = {
    'Fixed': {
        'base': (4.00, 5.50),
        'addon': (0.75, 2.00),
        'basis': (0.10, 0.30),
        'liquidity': (0.20, 0.50)
    },
    'Variable': {
        'base': (3.50, 4.50),
        'addon': (1.00, 2.50),
        'basis': (0.05, 0.25),
        'liquidity': (0.10, 0.40)
    },
    'Deposit': {
        'base': (2.50, 3.50),
        'addon': (-0.25, 0.75),
        'basis': (0.00, 0.15),
        'liquidity': (0.05, 0.25)
    },
}
fixed_mask = SyntheticData['InterestRateType'] == 'Fixed'
variable_mask = SyntheticData['InterestRateType'] == 'Variable'
deposit_mask = SyntheticData['AccountType'] == 'Deposit'
base_rates = np.zeros(len(SyntheticData))
addon_rates = np.zeros(len(SyntheticData))
basis_costs = np.zeros(len(SyntheticData))
liquidity_rates = np.zeros(len(SyntheticData))
base_rates[fixed_mask] = np.random.uniform(*rate_ranges['Fixed']['base'], size=fixed_mask.sum())
addon_rates[fixed_mask] = np.random.uniform(*rate_ranges['Fixed']['addon'], size=fixed_mask.sum())
basis_costs[fixed_mask] = np.random.uniform(*rate_ranges['Fixed']['basis'], size=fixed_mask.sum())
liquidity_rates[fixed_mask] = np.random.uniform(*rate_ranges['Fixed']['liquidity'], size=fixed_mask.sum())
base_rates[variable_mask & ~deposit_mask] = np.random.uniform(*rate_ranges['Variable']['base'], size=(variable_mask & ~deposit_mask).sum())
addon_rates[variable_mask & ~deposit_mask] = np.random.uniform(*rate_ranges['Variable']['addon'], size=(variable_mask & ~deposit_mask).sum())
basis_costs[variable_mask & ~deposit_mask] = np.random.uniform(*rate_ranges['Variable']['basis'], size=(variable_mask & ~deposit_mask).sum())
liquidity_rates[variable_mask & ~deposit_mask] = np.random.uniform(*rate_ranges['Variable']['liquidity'], size=(variable_mask & ~deposit_mask).sum())
base_rates[deposit_mask] = np.random.uniform(*rate_ranges['Deposit']['base'], size=deposit_mask.sum())
addon_rates[deposit_mask] = np.random.uniform(*rate_ranges['Deposit']['addon'], size=deposit_mask.sum())
basis_costs[deposit_mask] = np.random.uniform(*rate_ranges['Deposit']['basis'], size=deposit_mask.sum())
liquidity_rates[deposit_mask] = np.random.uniform(*rate_ranges['Deposit']['liquidity'], size=deposit_mask.sum())
SyntheticData['BaseRate'] = np.round(base_rates, 4)
SyntheticData['AddonRate'] = np.round(addon_rates, 4)
SyntheticData['BasisCost'] = np.round(basis_costs, 4)
SyntheticData['LiquidityRate'] = np.round(liquidity_rates, 4)
SyntheticData['BasisCalculation'] = SyntheticData['BaseRate'] - SyntheticData['InterestRate']
SyntheticData['FundingIndex'] = np.where(SyntheticData['AccountSource'] == 'BBK', 0.03, 0.015)
SyntheticData['IRRCalculation'] = SyntheticData['InterestRate'] / (1 + SyntheticData['Term'])
SyntheticData['LiquidityCalculation'] = np.where(SyntheticData['TypeOfAccount'] == 'Loan', SyntheticData['Term'] * 0.01, 0.0)
SyntheticData['AverageAddonRate'] = SyntheticData.groupby('CustomerID')['AddonRate'].transform('mean')
SyntheticData['AverageBaseRate'] = SyntheticData.groupby('CustomerID')['BaseRate'].transform('mean')
SyntheticData['AverageBasisCost'] = SyntheticData.groupby('CustomerID')['BasisCost'].transform('mean')
SyntheticData['TransferRate'] = SyntheticData['BaseRate'] + SyntheticData['AddonRate']
SyntheticData['TransferSpread'] = SyntheticData['TransferRate'] - SyntheticData['InterestRate']
SyntheticData['ExpectedReturn'] = SyntheticData['InterestRate'] - SyntheticData['FundingIndex']

loan_mask = SyntheticData['TypeOfAccount'] == 'Loan'
credit_score = pd.to_numeric(SyntheticData['CreditScore'], errors='coerce').fillna(600)
arrears_amt = pd.to_numeric(SyntheticData['ArrearsAmount'], errors='coerce').fillna(0.0)
market_value = pd.to_numeric(SyntheticData['MarketValue'], errors='coerce').fillna(np.nan)
lvr = pd.to_numeric(SyntheticData['LVR'], errors='coerce').fillna(0.0).clip(0, 1.5)
arrears_days = pd.to_numeric(SyntheticData['ArrearsDays'], errors='coerce').fillna(0).astype(int)
lmi_flag = SyntheticData['LMIFlag'].eq('Y').astype(float)
arrears_ratio = (arrears_amt / market_value.replace(0, np.nan)).fillna(0.0).clip(0, 5.0)
lvr_effect = 0.15 * (lvr - 0.60)
lvr_effect = np.clip(lvr_effect, -0.10, 0.30)
lmi_relief = -0.05 * lmi_flag
severity_adj = 0.20 * arrears_ratio.clip(0.0, 1.0)
lgd_base = 0.25
lgd_fixed = (lgd_base + lvr_effect + lmi_relief + severity_adj).clip(0.10, 0.90)

SyntheticData['PD'] = np.nan
SyntheticData['LGD'] = np.nan
SyntheticData['ExpectedCreditLoss'] = np.nan
SyntheticData.loc[loan_mask, 'PD'] = np.clip(1 - (pd.to_numeric(SyntheticData.loc[loan_mask, 'CreditScore'], errors='coerce').fillna(600) / 1000), 0.01, 0.99)
SyntheticData.loc[loan_mask, 'LGD'] = pd.to_numeric(SyntheticData.loc[loan_mask, 'LGD'], errors='coerce').fillna(lgd_fixed[loan_mask]).clip(0.10, 0.90)
ead_primary = pd.to_numeric(SyntheticData['ExposureAtDefault'], errors='coerce').fillna(0.0)
ead_fallback = pd.to_numeric(SyntheticData['ExposureNetCredit'], errors='coerce').fillna(0.0)
acct_princ = pd.to_numeric(SyntheticData['AccountPrincipal'], errors='coerce').fillna(0.0)
ead = np.where(ead_primary > 0, ead_primary, np.where(ead_fallback > 0, ead_fallback, acct_princ)).astype(float)
pd_vals = pd.to_numeric(SyntheticData['PD'], errors='coerce').fillna(0.0).clip(0.001, 0.999)
lgd_vals = pd.to_numeric(SyntheticData['LGD'], errors='coerce').fillna(0.25).clip(0.10, 0.90)
ecl_all_loans = pd_vals * lgd_vals * ead
ecl_dpd_only = np.where(arrears_days > 0, pd_vals * lgd_vals * ead, 0.0)
USE_IFRS9_STYLE = True
ecl_final = np.where(USE_IFRS9_STYLE, ecl_all_loans, ecl_dpd_only)
SyntheticData['ExpectedCreditLoss'] = 0.0
SyntheticData.loc[loan_mask, 'ExpectedCreditLoss'] = ecl_final[loan_mask]

bl_mask = (SyntheticData['AccountType'] == 'Business Loan') & (SyntheticData['AccountBalance'] > 1000000)
num_bl_mask = bl_mask.sum()
SyntheticData['Rating'] = np.nan
SyntheticData.loc[bl_mask, 'Rating'] = np.random.randint(6, 19, size=num_bl_mask)
SyntheticData['ImpairedNet'] = (pd.to_numeric(SyntheticData['AccountBalance'], errors='coerce').fillna(0.0)
                                - pd.to_numeric(SyntheticData['InterestNotBoughtToAccount'], errors='coerce').fillna(0.0))

SyntheticData['StageOfECL'] = 0
arr_days = pd.to_numeric(SyntheticData['ArrearsDays'], errors='coerce').fillna(0).astype(int)
stage_all = np.where(((arr_days >= 1) & (arr_days <= 29)), 1,
                     np.where(((arr_days >= 30) & (arr_days <= 59)), 2,
                              np.where((arr_days >= 60), 3, 0)))
SyntheticData['StageOfECL'] = 0
SyntheticData.loc[loan_mask, 'StageOfECL'] = stage_all[loan_mask]

stage = pd.to_numeric(SyntheticData['StageOfECL'], errors='coerce').fillna(0).astype(int)
ecl = pd.to_numeric(SyntheticData['ExpectedCreditLoss'], errors='coerce').fillna(0.0)
pd_ = pd.to_numeric(SyntheticData['PD'], errors='coerce').fillna(0.0)
lgd = pd.to_numeric(SyntheticData['LGD'], errors='coerce').fillna(0.0)

SyntheticData = SyntheticData.assign(
    Stage1ECL=np.where(stage == 1, ecl, 0.0),
    Stage2ECL=np.where(stage == 2, ecl, 0.0),
    Stage3ECL=np.where(stage == 3, ecl, 0.0),
    Stage1PD=np.where(stage == 1, pd_, 0.0),
    Stage2PD=np.where(stage == 2, pd_, 0.0),
    Stage3PD=np.where(stage == 3, pd_, 0.0),
    Stage1LGD=np.where(stage == 1, lgd, 0.0),
    Stage2LGD=np.where(stage == 2, lgd, 0.0),
    Stage3LGD=np.where(stage == 3, lgd, 0.0),
)

df_size = len(SyntheticData)
SyntheticData['FundingCost'] = SyntheticData['InterestAccrued'] * np.random.uniform(0.2, 0.5)
SyntheticData['FeesCharged'] = np.random.uniform(50, 300, size=df_size)
SyntheticData['OperationalCost'] = (SyntheticData['InterestAccrued'] + SyntheticData['FeesCharged']) * np.random.uniform(0.1, 0.3)
SyntheticData['MonthlyDepositFrequency'] = np.random.poisson(lam=4, size=df_size)
SyntheticData['StressScore'] = np.clip(np.random.normal(loc=0.5, scale=0.2, size=df_size), 0, 1)
SyntheticData['WithdrawalHistory'] = np.random.binomial(n=10, p=0.3, size=df_size)
SyntheticData['MacroVolatilityIndex'] = np.clip(np.random.normal(loc=0.6, scale=0.15, size=df_size), 0, 1)
risk_score = (
        0.4 * SyntheticData['StressScore'] +
        0.3 * SyntheticData['MacroVolatilityIndex'] +
        0.2 * (SyntheticData['WithdrawalHistory'] / 10) -
        0.1 * (SyntheticData['MonthlyDepositFrequency'] / 10)
)
SyntheticData['WithdrawalRisk'] = (risk_score > 0.5).astype(int)

loan_mask = SyntheticData['TypeOfAccount'].eq('Loan')
deposit_mask = SyntheticData['TypeOfAccount'].eq('Deposit')
corp_mask = loan_mask & SyntheticData['AccountSource'].eq('BBK')
retl_mask = loan_mask & SyntheticData['AccountSource'].eq('RTE')
res_mask = loan_mask & SyntheticData['CollateralCategory'].eq('Residential Property')
comm_mask = loan_mask & SyntheticData['CollateralCategory'].isin(['Commercial', 'Agri', 'Dev Finance'])
unsecured_mask = loan_mask & ~(res_mask | comm_mask | SyntheticData['CollateralCategory'].isin(['PI', 'Other']))
ead = pd.to_numeric(SyntheticData['ExposureAtDefault'], errors='coerce').fillna(0.0)
pd_ = pd.to_numeric(SyntheticData['PD'], errors='coerce').fillna(0.0).clip(0.001, 0.999)
lgd_ = pd.to_numeric(SyntheticData['LGD'], errors='coerce').fillna(0.25).clip(0.10, 0.90)
rw = pd.to_numeric(SyntheticData['RiskWeight'], errors='coerce').fillna(0.50)
term_m = pd.to_numeric(SyntheticData['Term'], errors='coerce').fillna(0).clip(0, 360)
stress = pd.to_numeric(SyntheticData['StressScore'], errors='coerce').fillna(0.5)
macro = pd.to_numeric(SyntheticData['MacroVolatilityIndex'], errors='coerce').fillna(0.6)
arrears_days = pd.to_numeric(SyntheticData['ArrearsDays'], errors='coerce').fillna(0).astype(int)
lvr = pd.to_numeric(SyntheticData['LVR'], errors='coerce').fillna(0).clip(0, 2)
acct_princ = pd.to_numeric(SyntheticData['AccountPrincipal'], errors='coerce').fillna(0.0)
group_exp = pd.to_numeric(SyntheticData['TotalBusinessGroupExposure'], errors='coerce').fillna(0.0)
rating = pd.to_numeric(SyntheticData['Rating'], errors='coerce')
cs = pd.to_numeric(SyntheticData['CreditScore'], errors='coerce').fillna(600)

SyntheticData['Currency'] = np.random.choice(['AUD', 'USD', 'EUR', 'GBP', 'JPY'], p=[0.88, 0.05, 0.03, 0.02, 0.02], size=len(SyntheticData))
SyntheticData['GeographicRegion'] = np.random.choice(['QLD', 'NSW', 'VIC', 'WA', 'SA', 'TAS', 'NT', 'ACT'], p=[0.20, 0.30, 0.25, 0.10, 0.06, 0.03, 0.03, 0.03],
                                                     size=len(SyntheticData))
prod_cnt = SyntheticData.groupby('CustomerID')['AccountID'].transform('count')
min_orig = SyntheticData.groupby('CustomerID')['DateOfOrigination'].transform('min').fillna(SyntheticData['DateOfPortfolio'])

reg_pd = np.clip(pd_, 0.003, 0.20)
reg_lgd = np.select([res_mask, comm_mask, unsecured_mask], [0.20, 0.45, 0.60], default=lgd_)
liq_bucket = np.select([term_m <= 12, term_m <= 36, term_m > 36], ['Short', 'Medium', 'Long'], default='N/A')
fund_source = np.where(deposit_mask, 'Retail', np.where(corp_mask & (acct_princ > 1e6), 'Wholesale', 'Retail'))
fund_cost_rate = SyntheticData['FundingIndex'] + SyntheticData['BasisCost'] + SyntheticData['LiquidityRate']
liq_prem = SyntheticData['LiquidityRate'] * (term_m / 360)
stable_flag = np.where(deposit_mask | (term_m >= 12), 'Y', 'N')
ir_shock = acct_princ * 0.02 * (term_m / 12)
cs_shock = ead * np.where(corp_mask, 0.01, 0.005)
pd_stress = np.clip(pd_ * 1.5 + 0.05 * stress + 0.05 * macro, 0.001, 0.999)
lgd_stress = np.clip(reg_lgd + 0.10 + 0.05 * (lvr > 0.80), 0.10, 0.90)
stress_flag = np.where(stress + macro > 1.2, 'Severe', 'Base')
total_rwa = ead * rw
capital_charge = total_rwa * 0.08
capital_buffer = total_rwa * (0.025 + np.where(corp_mask, 0.01, 0.0))
risk_grade = np.where(~rating.isna(), rating, np.clip(np.ceil((1000 - cs) / 35), 1, 20)).astype(int)
reg_asset_class = np.select(
    [retl_mask & res_mask, corp_mask & (group_exp > 1.5e6), corp_mask & (group_exp <= 1.5e6), loan_mask & ~res_mask & retl_mask, deposit_mask],
    ['Retail Mortgage', 'Corporate', 'SME Corporate', 'SME Retail', 'Deposit'], default='Other'
)
portfolio_seg = SyntheticData['ExposureGroup'].map(
    {'RETL': 'Retail Mortgage', 'COMM': 'Commercial Property', 'RESI': 'Residential Mortgage', 'CORP': 'Corporate', 'COMD': 'Commodities', 'LEAS': 'Leasing',
     'UNKN': 'Unknown'}).fillna('Other')
industry_sector = np.select(
    [SyntheticData['CollateralCategory'].eq('Agri'),
     SyntheticData['CollateralCategory'].eq('Commercial'),
     SyntheticData['CollateralCategory'].eq('Dev Finance'),
     SyntheticData['CollateralCategory'].eq('PI'),
     SyntheticData['CollateralCategory'].eq('Other')],
    ['AGRI', 'COMM', 'DEVF', 'PI', 'OTHER'], default=np.where(res_mask, 'RESI', 'UNSEC')
)

SyntheticData = SyntheticData.assign(
    TotalRWA=total_rwa,
    CapitalCharge=capital_charge,
    CapitalBufferImpact=capital_buffer,
    RegulatoryAssetClass=reg_asset_class,
    RiskGrade=risk_grade,
    BaselExpectedLoss=reg_pd * reg_lgd * ead,
    RegulatoryPD=reg_pd,
    RegulatoryLGD=reg_lgd,
    LiquidityBucket=liq_bucket,
    FundingSource=fund_source,
    FundingCostRate=fund_cost_rate.round(4),
    LiquidityPremium=liq_prem.round(4),
    StableFundingFlag=stable_flag,
    InterestRateShockImpact=ir_shock.round(2),
    CreditSpreadShockImpact=cs_shock.round(2),
    FXShockImpact=np.where(SyntheticData['Currency'].ne('AUD'), (ead * 0.10), 0.0).round(2),
    StressAdjustedPD=pd_stress,
    StressAdjustedLGD=lgd_stress,
    StressScenarioFlag=stress_flag,
    StressLossEstimate=(pd_stress * lgd_stress * ead).round(2),
    CustomerTier=np.select([group_exp > 5e6, group_exp > 1.5e6, group_exp > 5e5], ['Platinum', 'Gold', 'Silver'], default='Bronze'),
    CustomerRiskSegment=np.select([(pd_ > 0.25) | (arrears_days > 60), (pd_ > 0.10) | (arrears_days > 30)], ['High', 'Medium'], default='Low'),
    PortfolioSegment=portfolio_seg,
    IndustrySector=industry_sector,
    CrossSellFlag=np.where(prod_cnt > 1, 'Y', 'N'),
    GroupExposureRank=SyntheticData.groupby('CustomerID')['TotalAccountExposure'].rank(method='dense', ascending=False).astype(int),
    RelationshipLengthYears=np.round(((SyntheticData['DateOfPortfolio'] - min_orig).dt.days / 365).fillna(0), 2),
    Vintage=SyntheticData['DateOfOrigination'].dt.year.fillna(SyntheticData['DateOfPortfolio'].dt.year)
)

SyntheticData['RAROC'] = np.where(SyntheticData['CapitalCharge'] > 0,
                                  (
                                          SyntheticData['InterestAccrued'].fillna(0)
                                          + SyntheticData['FeesCharged'].fillna(0)
                                          - SyntheticData['ExpectedCreditLoss'].fillna(0)
                                          - SyntheticData['FundingCost'].fillna(0)
                                          - SyntheticData['OperationalCost'].fillna(0)
                                  ) / SyntheticData['CapitalCharge'], np.nan).round(4)

snapshot = SyntheticData.iloc[:, [1, 2, 6, -2, -1]].groupby(SyntheticData['AccountType']).apply(
    lambda x: x.sample(min(len(x), 5)), include_groups=False).reset_index(drop=True)
print(snapshot)
print(num_records)
print(f'Number of fields: {SyntheticData.shape[1]}')
end_time = time.time()
run_time = end_time - start_time
print(f'Script completed in {run_time:.2f} seconds')
