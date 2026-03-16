import numpy as np

def train_all_models(df, seed=None, sample_frac=0.15, skip_heavy=False):
    from .pd_model import train_pd_model
    from .lgd_model import train_lgd_model
    from .ead_model import train_ead_model
    from .staging_classifier import train_staging_classifier
    from .raroc_model import train_raroc_model
    from .anomaly_detector import run_anomaly_detector
    from .customer_segmentation import run_customer_segmentation

    out = {}
    out["pd"] = train_pd_model(df, sample_frac=sample_frac)
    out["lgd"] = train_lgd_model(df, sample_frac=sample_frac)
    out["ead"] = train_ead_model(df, sample_frac=sample_frac)
    out["staging"] = train_staging_classifier(df, sample_frac=sample_frac)
    out["raroc"] = train_raroc_model(df, sample_frac=sample_frac)
    if skip_heavy:
        out["anomaly"] = {"scores": [], "flags": np.array([], dtype=int), "features": []}
        out["segment"] = {"labels": np.array([], dtype=int)}
    else:
        out["anomaly"] = run_anomaly_detector(df, seed=seed)
        out["segment"] = run_customer_segmentation(df, seed=seed)
    return out
