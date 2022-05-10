from typing import Callable

import numpy as np
import pandas as pd


def generate_data(
    n_per_dim: int,
    radius: float,
    data_generating_function: Callable[[float, float], float],
    perc_withheld=0.2,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    n_periods = 4

    x = np.linspace(-n_periods, n_periods, n_per_dim) * np.pi
    y = np.linspace(-n_periods, n_periods, n_per_dim) * np.pi
    xv, yv = np.meshgrid(x, y)
    z = data_generating_function(xv, yv)

    n_withheld = int(perc_withheld * n_per_dim**2)

    to_withhold = set()
    while len(to_withhold) < n_withheld:
        i, j = np.random.randint([n_per_dim, n_per_dim])
        #         print(i, j)
        candidates = [
            (a, b)
            for a in range(i - radius, i + radius + 1)
            for b in range(j - radius, j + radius + 1)
        ]
        candidates = [
            tup
            for tup in candidates
            if all(x >= 0 and x < n_per_dim for x in tup) and tup not in to_withhold
        ]
        to_withhold.update(candidates[: n_withheld - len(to_withhold)])
    to_withhold = list(to_withhold)

    withheld_idxs = [i * n_per_dim + j for i, j in to_withhold]

    data = pd.DataFrame({"x": xv.flatten(), "y": yv.flatten()})
    labels = pd.Series(z.flatten(), name="z")

    return data, labels, withheld_idxs
