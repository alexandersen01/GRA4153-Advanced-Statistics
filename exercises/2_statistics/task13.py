import numpy as np
from numpy import tanh
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ----------------------------
# Logistic log-likelihood etc.
# ----------------------------
def logistic_logpdf(x, theta):
    """
    Log-pdf of logistic(loc=theta, scale=1) evaluated at x.
    Vectorized in x.
    """
    z = x - theta
    # log f(z) = -z - 2 log(1 + exp(-z))
    # Use logaddexp for numerical stability: log(1+e^a) = logaddexp(0, a)
    return -z - 2.0 * np.logaddexp(0.0, -z)

def loglik(theta, x):
    """
    Log-likelihood l(theta) for data x under logistic(loc=theta, scale=1).
    """
    x = np.asarray(x)
    return np.sum(logistic_logpdf(x, theta))

def score(theta, x):
    """
    Score l'(theta) = sum tanh((x_i - theta)/2).
    """
    x = np.asarray(x)
    return np.sum(np.tanh((x - theta)/2.0))

def hess(theta, x):
    """
    Hessian l''(theta) = -(1/2) * sum sech^2((x_i - theta)/2).
    """
    x = np.asarray(x)
    u = (x - theta)/2.0
    # sech^2(u) = 1/cosh^2(u)
    return -0.5 * np.sum(1.0 / np.cosh(u)**2)

# ----------------------------
# MLE via robust root-finding
# ----------------------------
def mle_theta(x, bracket=None):
    """
    Compute the MLE of theta by solving score(theta)=0 with a robust bracketing method.
    Because the score is strictly decreasing in theta and tends to +n at -inf and -n at +inf,
    a root always exists and is unique.

    Parameters
    ----------
    x : array-like
        Sample.
    bracket : tuple (a, b), optional
        Optional bracket for the root. If not provided, a wide bracket based on min/max(x) is used.

    Returns
    -------
    theta_hat : float
        MLE of theta.
    """
    x = np.asarray(x)
    g = lambda th: score(th, x)

    if bracket is None:
        a, b = np.min(x) - 50.0, np.max(x) + 50.0
    else:
        a, b = bracket

    # Ensure the bracket straddles the root; widen if necessary (should almost never be needed)
    fa, fb = g(a), g(b)
    if fa * fb > 0:
        m = np.median(x)
        a, b = m - 1.0, m + 1.0
        fa, fb = g(a), g(b)
        step = 1.0
        for _ in range(60):
            if fa * fb <= 0:
                break
            a -= step
            b += step
            fa, fb = g(a), g(b)
            step *= 2.0

    return brentq(g, a, b)

# ----------------------------
# Monte Carlo study
# ----------------------------
def monte_carlo_study(theta_true=0.0, n_list=(10, 20, 50, 100, 200, 500, 1000),
                      M=2000, seed=12345, make_plots=True):
    """
    Monte Carlo:
    - For each n in n_list, draw M iid samples from Logistic(theta_true, 1).
    - Compute MLE for each sample.
    - Report bias, SD, MAE, RMSE.
    - Optionally make plots and compare RMSE to sqrt(3/n), the asymptotic RMSE.

    Returns
    -------
    results : dict
        Keys 'n', 'theta_hat_all', 'bias', 'sd', 'mae', 'rmse', 'theory_rmse'.
    """
    rng = np.random.default_rng(seed)
    results = {
        'n': [],
        'theta_hat_all': [],
        'bias': [],
        'sd': [],
        'mae': [],
        'rmse': [],
        'theory_rmse': []
    }

    for n in n_list:
        theta_hats = np.empty(M)
        for m in range(M):
            x = rng.logistic(loc=theta_true, scale=1.0, size=n)
            theta_hats[m] = mle_theta(x)

        err = theta_hats - theta_true
        bias = err.mean()
        sd = err.std(ddof=1)
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        theory_rmse = np.sqrt(3.0/n)  # Fisher information per obs is 1/3

        results['n'].append(n)
        results['theta_hat_all'].append(theta_hats)
        results['bias'].append(bias)
        results['sd'].append(sd)
        results['mae'].append(mae)
        results['rmse'].append(rmse)
        results['theory_rmse'].append(theory_rmse)

    if make_plots:
        n_arr = np.array(results['n'], dtype=float)
        rmse_arr = np.array(results['rmse'])
        mae_arr = np.array(results['mae'])
        th_rmse = np.array(results['theory_rmse'])

        # RMSE vs n with theoretical curve
        plt.figure(figsize=(6.0, 4.2))
        plt.loglog(n_arr, rmse_arr, 'o-', label='RMSE (MC)')
        plt.loglog(n_arr, th_rmse, 'k--', label='sqrt(3/n) (theory)')
        plt.xlabel('n')
        plt.ylabel('RMSE of $\hat{\\theta}$')
        plt.title('Logistic location MLE: RMSE vs n')
        plt.legend()
        plt.tight_layout()

        # MAE vs n
        plt.figure(figsize=(6.0, 4.2))
        plt.loglog(n_arr, mae_arr, 'o-', color='tab:orange', label='MAE (MC)')
        plt.xlabel('n')
        plt.ylabel('MAE of $\hat{\\theta}$')
        plt.title('Logistic location MLE: MAE vs n')
        plt.legend()
        plt.tight_layout()

        # Distribution for a representative n (largest n)
        n_rep = results['n'][-1]
        err_rep = results['theta_hat_all'][-1] - theta_true
        scale_err = np.sqrt(n_rep/3.0) * err_rep
        plt.figure(figsize=(6.0, 4.2))
        plt.hist(scale_err, bins=40, density=True, alpha=0.7, color='tab:green')
        plt.xlabel(r'$\sqrt{n/3}\,(\hat{\theta}-\theta)$')
        plt.ylabel('Density')
        plt.title(f'Scaled error for n={n_rep} (approx N(0,1))')
        plt.tight_layout()

        plt.show()

    return results

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Small demo
    rng = np.random.default_rng(0)
    x_demo = rng.logistic(loc=1.5, scale=1.0, size=50)
    theta_hat_demo = mle_theta(x_demo)
    print(f"Demo MLE (n=50, true theta=1.5): {theta_hat_demo:.4f}")

    # Run Monte Carlo and plot
    res = monte_carlo_study(theta_true=0.0,
                            n_list=(10, 20, 50, 100, 200, 500, 1000),
                            M=2000, seed=1, make_plots=True)

    # Print a brief summary
    for n, bias, sd, rmse in zip(res['n'], res['bias'], res['sd'], res['rmse']):
        print(f"n={n:4d}  bias={bias: .4f}  sd={sd: .4f}  rmse={rmse: .4f}")