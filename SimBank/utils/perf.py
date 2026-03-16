import time, logging
from functools import wraps


def time_it(label):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            logging.info(f"{label} {dt:.3f}s")
            return out
        return wrapper
    return deco
