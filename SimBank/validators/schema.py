def validate_required_columns(df):
    req = ["DateOfPortfolio","AccountID","AccountType","TypeOfAccount","AccountSource","AccountBalance","MarketValue","InterestRate","AnnualRate","MonthlyRate","CreditScore","LVR","CustomerID"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"missing {miss}")
