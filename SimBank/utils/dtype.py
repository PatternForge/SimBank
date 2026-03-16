def optimize_dtypes(cfg, df):
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(cfg.dtype_float)
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype(cfg.dtype_int)
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].nunique() < max(256, int(0.01 * len(df))):
            df[c] = df[c].astype("category")
    return df
