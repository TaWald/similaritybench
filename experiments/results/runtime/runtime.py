from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import pairwise_distances

# from scipy.spatial.distance import pdist

# ns = [20000, 30000, 40000,]
ns = [60000, 80000]
d = 1000
jobs = [1, 2, 8]
repeat = 1
rng = np.random.default_rng(1234567)

cols = ["function", "n", "d", "time", "job"]
records = []

for i in range(repeat):
    logger.info(f"{i=}")
    for n in ns:
        logger.info(f"{n=}")
        x = rng.random((n, d))

        # logger.info("Scipy...")
        # start = perf_counter()
        # pdist(x, metric="euclidean")
        # end = perf_counter()
        # records.append(("pdist", n, d, end - start, "scipy"))

        for job in jobs:
            logger.info(f"sklearn {job=}...")

            start = perf_counter()
            pairwise_distances(x, n_jobs=job, metric="euclidean")
            end = perf_counter()
            records.append(("pairwise_distances", n, d, end - start, job))

df = pd.DataFrame.from_records(records, columns=cols).to_csv("runtime.csv")
