import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd


def two_prop_stats(xsum, ysum, n1, n2):
    # sums are arrays of len M with counts of 1s
    M = xsum.shape[0]
    p1_hat = xsum / n1
    p2_hat = ysum / n2
    p_hat = (xsum + ysum) / (n1 + n2)
    num = p1_hat - p2_hat

    # denom sqrs
    denom1_sq = p_hat * (1 - p_hat) * (1 / n1 + 1 / n2)
    denom2_sq = p1_hat * (1 - p1_hat) / n1 + p2_hat * (1 - p2_hat) / n2

    # Z1
    denom1 = np.sqrt(denom1_sq)
    z1 = np.zeros(M)
    # avoid 0 in the denom
    mask1 = denom1 > 0
    z1[mask1] = num[mask1] / denom1[mask1]
    z1[~mask1] = np.where(num[~mask1] == 0, 0.0, np.sign(num[~mask1]) * np.inf)

    # Z2 (and Z3, identical here)
    denom2 = np.sqrt(denom2_sq)
    z2 = np.zeros(M)
    mask2 = denom2 > 0
    z2[mask2] = num[mask2] / denom2[mask2]
    z2[~mask2] = np.where(num[~mask2] == 0, 0.0, np.sign(num[~mask2]) * np.inf)

    # Z3 is Z2 for B with s2 defined with denom n
    z3 = z2.copy()
    return z1, z2, z3


def simulate_rejections(p1, p2, n1, n2, alpha=0.05, M=5000):
    rng = np.random.default_rng(42)

    # Gen M x n1 and M x n2 B matrices
    X = rng.binomial(1, p1, size=(M, n1))
    Y = rng.binomial(1, p2, size=(M, n2))
    xsum = X.sum(axis=1)
    ysum = Y.sum(axis=1)

    z1, z2, z3 = two_prop_stats(xsum, ysum, n1, n2)
    c = norm.ppf(1 - alpha / 2)
    rej1 = np.mean(np.abs(z1) > c)
    rej2 = np.mean(np.abs(z2) > c)
    rej3 = np.mean(np.abs(z3) > c)
    return rej1, rej2, rej3


def run_grid(p_pairs, n_pairs, alpha=0.05, M=5000):
    rng = np.random.default_rng(42)
    res = []
    for p1, p2 in p_pairs:
        for n1, n2 in n_pairs:
            r1, r2, r3 = simulate_rejections(p1, p2, n1, n2, alpha=alpha, M=M)
            res.append(
                {
                    "p1": p1,
                    "p2": p2,
                    "n1": n1,
                    "n2": n2,
                    "rej_Z1": r1,
                    "rej_Z2": r2,
                    "rej_Z3": r3,
                }
            )

    return res


alpha = 0.05
M = 5000
p_pairs = [
    (0.1, 0.1),
    (0.5, 0.5),
    (0.9, 0.9),  # size (H0 true)
    (0.1, 0.2),
    (0.4, 0.6),
    (0.7, 0.5),  # power (H1 true)
]
n_pairs = [(50, 50), (200, 200), (50, 200)]  # balanced vs unbalanced

res = run_grid(p_pairs, n_pairs, alpha, M)

# Print results
for r in res:
    print(
        f"p1={r['p1']:.2f}, p2={r['p2']:.2f}, n1={r['n1']}, n2={r['n2']} rej Z1={r['rej_Z1']:.3f}, Z2={r['rej_Z2']:.3f}, Z3={r['rej_Z3']:.3f}"
    )