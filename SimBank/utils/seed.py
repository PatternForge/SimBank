import numpy as np, secrets


def make_rng(seed):
    return np.random.default_rng(None if seed is None else seed)


def generate_run_seed():
    return secrets.randbits(32)
