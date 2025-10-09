import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 200_00000

x_norm = rng.normal(loc=0.0, scale=1.0, size=n)
cummean_norm = np.cumsum(x_norm) / np.arange(1, n + 1)

x_cauchy = rng.standard_cauchy(size=n)
cummean_cauchy = np.cumsum(x_cauchy) / np.arange(1, n + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
axes[0].plot(cummean_norm, lw=1, color="tab:blue")
axes[0].axhline(0, color="k", ls="--", lw=1)
axes[0].set_title("Running mean: Normal N(0,1)")
axes[0].set_xlabel("n")
axes[0].set_ylabel("cumulative average")
axes[0].set_ylim(-0.2, 0.2)  # zoom to see stabilization

axes[1].plot(cummean_cauchy, lw=0.6, color="tab:red", alpha=0.9)
axes[1].axhline(0, color="k", ls="--", lw=1)
axes[1].set_title("Running mean: Cauchy(0,1)")
axes[1].set_xlabel("n")
axes[1].set_ylabel("cumulative average")
axes[1].set_ylim(-10, 10)  # limit for visibility; extremes may lie outside the frame

plt.tight_layout()
# plt.show()
plt.savefig("exercises/2_statistics/image.png")


print(f"Final running mean (Normal): {cummean_norm[-1]:.6f}")
print(f"Final running mean (Cauchy): {cummean_cauchy[-1]:.6f}")
