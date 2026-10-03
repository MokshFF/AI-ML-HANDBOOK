"""
Probability and Statistics for ML - Statistical Computing Engine
Implements distribution functions, Bayesian updates, sample moments,
Central Limit Theorem simulation, bootstrap confidence intervals, and hypothesis testing.
"""

from __future__ import annotations
import math
from typing import Callable
import numpy as np


# ---------------------------------------------------------
# 1. Probability Distributions & Bayes Theorem
# ---------------------------------------------------------

def normal_pdf(x: float | np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> float | np.ndarray:
    """Evaluates Gaussian Probability Density Function."""
    if sigma <= 0:
        raise ValueError("Standard deviation sigma must be strictly positive.")
    denom = sigma * np.sqrt(2.0 * np.pi)
    exponent = -0.5 * (((x - mu) / sigma) ** 2)
    return np.exp(exponent) / denom


def bayes_posterior(prior: float, likelihood: float, evidence: float) -> float:
    """
    Computes Bayes posterior probability:
    P(A | B) = (P(B | A) * P(A)) / P(B)
    """
    if evidence <= 0:
        raise ValueError("Evidence P(B) must be strictly positive.")
    return (likelihood * prior) / evidence


def bayes_binary_update(prior: float, p_pos_given_disease: float, p_pos_given_healthy: float) -> float:
    """
    Calculates posterior probability P(Disease | Test = Positive) using law of total probability.
    """
    p_healthy = 1.0 - prior
    evidence = (p_pos_given_disease * prior) + (p_pos_given_healthy * p_healthy)
    return bayes_posterior(prior, p_pos_given_disease, evidence)


# ---------------------------------------------------------
# 2. Sample Moments & Correlations
# ---------------------------------------------------------

def sample_mean(x: np.ndarray) -> float:
    """E[X] = (1 / N) * sum(x_i)"""
    return float(np.sum(x) / len(x))


def sample_variance(x: np.ndarray, ddof: int = 1) -> float:
    """Var(X) = (1 / (N - ddof)) * sum((x_i - mean)^2)"""
    mu = sample_mean(x)
    return float(np.sum((x - mu) ** 2) / (len(x) - ddof))


def sample_covariance(x: np.ndarray, y: np.ndarray, ddof: int = 1) -> float:
    """Cov(X, Y) = (1 / (N - ddof)) * sum((x_i - mu_x) * (y_i - mu_y))"""
    if len(x) != len(y):
        raise ValueError("Arrays must have identical length.")
    mu_x = sample_mean(x)
    mu_y = sample_mean(y)
    return float(np.sum((x - mu_x) * (y - mu_y)) / (len(x) - ddof))


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson linear correlation:
    rho = Cov(X, Y) / (std(X) * std(Y)) in [-1, 1]
    """
    cov = sample_covariance(x, y)
    std_x = np.sqrt(sample_variance(x))
    std_y = np.sqrt(sample_variance(y))
    if std_x == 0 or std_y == 0:
        return 0.0
    return float(cov / (std_x * std_y))


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation computed by ranking observations."""
    def rank_array(a: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(a)).astype(np.float64)

    rx = rank_array(x)
    ry = rank_array(y)
    return pearson_correlation(rx, ry)


# ---------------------------------------------------------
# 3. Central Limit Theorem & Sampling
# ---------------------------------------------------------

def simulate_clt(
    sampler: Callable[[int], np.ndarray],
    sample_size: int = 30,
    num_trials: int = 1000
) -> np.ndarray:
    """
    Draws samples of size sample_size across num_trials and returns empirical means.
    Regardless of the underlying distribution, the distribution of sample means
    converges to a Gaussian N(mu, sigma^2 / n).
    """
    sample_means = np.zeros(num_trials, dtype=np.float64)
    for i in range(num_trials):
        batch = sampler(sample_size)
        sample_means[i] = np.mean(batch)
    return sample_means


# ---------------------------------------------------------
# 4. Confidence Intervals (Analytical & Bootstrap)
# ---------------------------------------------------------

def normal_confidence_interval(
    x: np.ndarray,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Computes analytical normal confidence interval for sample mean.
    Returns: (mean, lower_bound, upper_bound)
    """
    from scipy import stats
    n = len(x)
    mu = sample_mean(x)
    se = np.sqrt(sample_variance(x)) / np.sqrt(n)
    alpha = 1.0 - confidence
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    margin = z_crit * se
    return mu, mu - margin, mu + margin


def bootstrap_confidence_interval(
    x: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    num_resamples: int = 2000,
    seed: int | None = None
) -> tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval via percentile method.
    Resamples with replacement to approximate the sampling distribution.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(x)
    resample_stats = np.zeros(num_resamples, dtype=np.float64)
    for i in range(num_resamples):
        resample = np.random.choice(x, size=n, replace=True)
        resample_stats[i] = statistic_fn(resample)

    alpha = (1.0 - confidence) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    lower = float(np.percentile(resample_stats, lower_pct))
    upper = float(np.percentile(resample_stats, upper_pct))
    original_stat = float(statistic_fn(x))
    return original_stat, lower, upper


# ---------------------------------------------------------
# 5. Hypothesis Testing
# ---------------------------------------------------------

def two_sample_welch_t_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Welch's two-sample t-test (unequal variances assumed).
    Returns: (t_statistic, degrees_of_freedom, two_tailed_p_value)
    """
    from scipy import stats

    n1, n2 = len(x), len(y)
    m1, m2 = sample_mean(x), sample_mean(y)
    v1, v2 = sample_variance(x), sample_variance(y)

    se = np.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite equation for degrees of freedom
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = df_num / df_den

    # Two-tailed p-value
    p_value = 2.0 * float(1.0 - stats.t.cdf(np.abs(t_stat), df=df))
    return float(t_stat), float(df), p_value


def permutation_test_difference(
    x: np.ndarray,
    y: np.ndarray,
    num_permutations: int = 5000,
    seed: int | None = None
) -> tuple[float, float]:
    """
    Exact non-parametric permutation test for difference in means under null hypothesis H0: mu_x == mu_y.
    Returns: (observed_difference, p_value)
    """
    if seed is not None:
        np.random.seed(seed)

    observed_diff = np.abs(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    n_total = len(pooled)

    exceed_count = 0
    for _ in range(num_permutations):
        permuted = np.random.permutation(pooled)
        perm_x = permuted[:n_x]
        perm_y = permuted[n_x:]
        perm_diff = np.abs(np.mean(perm_x) - np.mean(perm_y))
        if perm_diff >= observed_diff:
            exceed_count += 1

    p_value = float(exceed_count / num_permutations)
    return float(observed_diff), p_value
