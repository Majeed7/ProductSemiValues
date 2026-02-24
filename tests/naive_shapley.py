import math
import numpy as np
from itertools import combinations


def naive_shapley(a):
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    phi = np.zeros(n, dtype=np.float64)
    fact_n = math.factorial(n)

    for i in range(n):
        rest = [j for j in list(range(n)) if j != i]
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / fact_n
            for S in combinations(rest, size):
                prod_S = 1.0
                for j in S:
                    prod_S *= a[j]
                phi[i] += weight * (a[i] - 1.0) * prod_S

    return phi
