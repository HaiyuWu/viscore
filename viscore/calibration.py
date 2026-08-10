"""Turn a VIScore into a predicted success rate, using the map frozen in the paper.

    from viscore import predict_success
    predict_success(0.72)                 # -> success points, on the three fitted tasks' scale

The score itself is dimensionless: it orders checkpoints but says nothing about how many episodes
will succeed. The paper's calibration column comes from a monotone (isotonic) map from score to
success rate, fitted once on the development labels and then applied unchanged -- to held-out
methods, to an unseen task, and to whatever model you bring. Fitting it again on your own
checkpoints would make your number incomparable to the published ones, so this reads the shipped
fit pool and nothing else.

What it is not: a success predictor for a single checkpoint. Its error on the transfer pools is
8-12 success points, against a constant-predictor reference of 10-12, so it is useful for ranking
candidates and for spotting a model far below the pool, not for promising an absolute rate.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

# the pool the map is fitted on: every development-labelled cell of the three fitted tasks
FIT_POOL = Path(__file__).resolve().parents[1] / "reproduce/pools/pool_manifest.csv"
GRADED = ("pusht", "dmc", "tworoom")


@lru_cache(maxsize=8)
def _fit(metric: str = "vis"):
    """The isotonic map, fitted on the shipped manifest. Cached: the fit is deterministic."""
    from sklearn.isotonic import IsotonicRegression

    if not FIT_POOL.exists():
        raise FileNotFoundError(
            f"{FIT_POOL} is missing. The frozen calibration lives in the published manifest; "
            "clone the repository or fetch reproduce/pools/pool_manifest.csv.")
    x, y = [], []
    for r in csv.DictReader(open(FIT_POOL)):
        if r["in_calibration_fit"] != "1":
            continue
        if r[metric] in ("", "nan") or r["sr_development"] in ("", "nan"):
            continue
        x.append(float(r[metric]))
        y.append(float(r["sr_development"]))
    if len(x) < 50:
        raise ValueError(f"fit pool has only {len(x)} cells; expected the published 472")
    return IsotonicRegression(out_of_bounds="clip").fit(x, y), len(x)


def predict_success(score, metric: str = "vis") -> np.ndarray | float:
    """Predicted success rate in points, for one score or an array of them."""
    iso, _ = _fit(metric)
    arr = np.atleast_1d(np.asarray(score, dtype=float))
    out = iso.predict(arr)
    return float(out[0]) if np.isscalar(score) or np.ndim(score) == 0 else out


def fit_pool_size(metric: str = "vis") -> int:
    """How many cells the shipped map was fitted on -- 472 for the published calibration."""
    return _fit(metric)[1]
