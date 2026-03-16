from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    n_records: int = np.random.randint(500000, 1000000)
    random_seed: int | None = None
    use_ifrs9_style: bool = True
    dtype_float: str = "float32"
    dtype_int: str = "int32"
    allow_business_lvr_gt_80: bool = False
