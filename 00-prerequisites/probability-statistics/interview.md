# Probability & Statistics for Machine Learning: Interview Question Bank

Technical screening questions, probability puzzles, A/B testing scenarios, and statistical estimation concepts.

---

## 1. Probability Theory & Bayesian Reasoning

### Q1: State Bayes' Theorem and explain how Maximum Likelihood Estimation (MLE) relates to Maximum A Posteriori (MAP) estimation.
- **Answer Outline**:
  - Bayes' Rule: $P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{P(\mathcal{D})}$.
  - Taking logs: $\log P(\theta \mid \mathcal{D}) = \log P(\mathcal{D} \mid \theta) + \log P(\theta) - \text{const}$.
  - **MLE**: Finds $\theta$ maximizing solely the empirical likelihood: $\hat{\theta}_{\text{MLE}} = \arg\max_\theta \log P(\mathcal{D} \mid \theta)$.
  - **MAP**: Finds $\theta$ maximizing the posterior, incorporating prior belief: $\hat{\theta}_{\text{MAP}} = \arg\max_\theta [\log P(\mathcal{D} \mid \theta) + \log P(\theta)]$.
  - **Connection to Regularization**: If the prior $P(\theta) \sim \mathcal{N}(0, \sigma^2 I)$ is Gaussian, MAP estimation is mathematically equivalent to $L_2$ regularization (Weight Decay / Ridge). If the prior is Laplace ($P(\theta) \sim e^{-\lambda \|\theta\|_1}$), MAP is equivalent to $L_1$ regularization (Lasso).

### Q2: If events $X$ and $Y$ have zero covariance ($\text{Cov}(X, Y) = 0$), are they necessarily independent?
- **Answer Outline**:
  - **No**. Zero covariance implies no *linear* relationship, but strong non-linear relationships can exist.
  - **Counterexample**: Let $X \sim \mathcal{U}[-1, 1]$ and $Y = X^2$.
    - $\mathbb{E}[X] = 0$.
    - $\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X^3] - 0 = 0$.
    - Yet $Y$ is completely deterministic given $X$; knowing $X$ completely determines $Y$. They are clearly dependent.
  - **Exception**: If $X$ and $Y$ are jointly bivariate Gaussian, zero covariance does imply independence.

---

## 2. Hypothesis Testing & A/B Experimentation

### Q3: How do you interpret a p-value of 0.03 in an A/B test between two recommendation algorithms?
- **Answer Outline**:
  - Assuming the null hypothesis is true (the two algorithms have identical true performance), there is a 3% probability of observing a performance difference as large as or larger than the observed sample difference purely due to random sampling variance.
  - Because $p = 0.03 < 0.05$, we reject $H_0$ at the $\alpha = 0.05$ significance level.
  - **Caveat**: It does not imply a 97% probability that the variant is better, nor does it guarantee business/practical significance without inspecting effect size and confidence intervals.

### Q4: What is the Multiple Testing Problem (p-hacking), and how is it addressed?
- **Answer Outline**:
  - If you test $K$ independent metrics at significance level $\alpha = 0.05$, the probability of at least one false positive (Type I error) is $1 - (1 - \alpha)^K$. For $K = 20$, this is $1 - 0.95^{20} \approx 64\%$.
  - **Corrections**:
    1. **Bonferroni Correction**: Set conservative threshold $\alpha_{\text{new}} = \alpha / K$.
    2. **Benjamini-Hochberg Procedure**: Controls False Discovery Rate (FDR) with less loss of statistical power.

---

## 3. Sampling & The Central Limit Theorem

### Q5: Why is the Central Limit Theorem foundational to ML evaluation, and when does it break down?
- **Answer Outline**:
  - Cross-validation test set metrics (e.g., mean accuracy, mean absolute error) are sample averages over $N$ test instances. By the CLT, the distribution of the sample mean metric approaches Gaussian as test set size increases, allowing construction of valid confidence intervals ($z$-scores / $t$-scores).
  - **Failure Modes**:
    1. **Infinite Variance**: Heavy-tailed Cauchy or Pareto distributions ($\alpha \le 2$) violate the finite variance assumption $\sigma^2 < \infty$.
    2. **Dependent Samples**: Time series, clustered network data, or spatial samples violate the i.i.d. assumption.

---

## 4. Coding Drill: Empirical Bootstrap Confidence Interval

### Task
Implement a Python function computing the non-parametric $95\%$ bootstrap confidence interval for any arbitrary statistic (e.g., median, trimmed mean).

```python
import numpy as np
from typing import Callable

def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.median,
    confidence: float = 0.95,
    num_samples: int = 2000
) -> tuple[float, float, float]:
    n = len(data)
    boot_stats = np.zeros(num_samples)
    for i in range(num_samples):
        resample = np.random.choice(data, size=n, replace=True)
        boot_stats[i] = stat_fn(resample)

    alpha = 1.0 - confidence
    low = float(np.percentile(boot_stats, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0)))
    obs = float(stat_fn(data))
    return obs, low, high
```
