# Probability & Statistics for Machine Learning: Theoretical Frameworks & Applied Inference

A rigorous guide to probability theory, random variables, sample moments, statistical estimation, confidence intervals, and hypothesis testing in machine learning.

---

## 1. Axiomatic Probability & Conditional Probability

### 1.1 Intuition
Machine learning algorithms operate on imperfect, noisy data. Probability provides the formal calculus to model uncertainty, quantify belief, and make optimal decisions under risk.

### 1.2 Formal Definition & Kolmogorov's Axioms
A probability space is a triple $(\Omega, \mathcal{F}, P)$, where $\Omega$ is the sample space, $\mathcal{F}$ is the event $\sigma$-algebra, and $P: \mathcal{F} \to [0, 1]$ satisfies:
1. **Non-negativity**: $P(E) \ge 0, \quad \forall E \in \mathcal{F}$.
2. **Unitarity**: $P(\Omega) = 1$.
3. **Countable Additivity**: For mutually disjoint events $\{E_i\}_{i=1}^\infty$:
   $$P\left( \bigcup_{i=1}^\infty E_i \right) = \sum_{i=1}^\infty P(E_i)$$

### 1.3 Conditional Probability & Independence
The conditional probability of event $A$ given event $B$ ($P(B) > 0$) is:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$
Events $A$ and $B$ are **statistically independent** ($A \perp B$) if and only if:
$$P(A \cap B) = P(A) P(B) \iff P(A \mid B) = P(A)$$

---

## 2. Bayes' Theorem & Statistical Updating

### 2.1 Intuition
Bayes' theorem is the engine of statistical learning: it formalizes how initial beliefs (prior) should be updated in the presence of observed data (likelihood) to arrive at refined beliefs (posterior).

### 2.2 Mathematical Formulation
$$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{P(\mathcal{D})} = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{\int P(\mathcal{D} \mid \theta') P(\theta') d\theta'}$$
- **$P(\theta)$ (Prior)**: Probability distribution over model parameters before observing data.
- **$P(\mathcal{D} \mid \theta)$ (Likelihood)**: Probability of the observed data under parameter hypothesis $\theta$.
- **$P(\mathcal{D})$ (Evidence / Marginal Likelihood)**: Normalization constant integrating over all possible hypotheses.
- **$P(\theta \mid \mathcal{D})$ (Posterior)**: Updated parameter belief distribution.

### 2.3 The Base Rate Fallacy (False Positive Paradox)
When a condition has very low prior prevalence $P(C) \ll 1$, even an accurate test ($99\%$ sensitivity, $95\%$ specificity) produces a low posterior probability of disease given a positive test, because the vast majority of positive signals come from false alarms among the healthy majority.

---

## 3. Random Variables & Key Probability Distributions

### 3.1 Random Variables (RVs)
A random variable $X: \Omega \to \mathbb{R}$ maps experimental outcomes to real values.
- **Discrete RVs**: Characterized by a **Probability Mass Function (PMF)** $p(x) = P(X = x)$, where $\sum p(x) = 1$.
- **Continuous RVs**: Characterized by a **Probability Density Function (PDF)** $f(x) \ge 0$, where $P(a \le X \le b) = \int_a^b f(x) dx$ and $\int_{-\infty}^\infty f(x) dx = 1$.
- **Cumulative Distribution Function (CDF)**: $F(x) = P(X \le x) = \int_{-\infty}^x f(t) dt$.

### 3.2 Canonical Distributions in Machine Learning

| Distribution | Type | PDF / PMF | Parameters | Typical ML Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Bernoulli** | Discrete | $p^x (1-p)^{1-x}, x \in \{0, 1\}$ | $p \in [0, 1]$ | Binary classification target |
| **Binomial** | Discrete | $\binom{n}{k} p^k (1-p)^{n-k}$ | $n \in \mathbb{N}, p \in [0, 1]$ | Click-through rates, trial successes |
| **Poisson** | Discrete | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda > 0$ | Count data, server arrival queries |
| **Gaussian (Normal)**| Continuous | $\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$ | $\mu \in \mathbb{R}, \sigma > 0$ | Residual noise, weight initialization |
| **Exponential** | Continuous | $\lambda e^{-\lambda x}, x \ge 0$ | $\lambda > 0$ | Survival analysis, inter-arrival times |
| **Uniform** | Continuous | $\frac{1}{b - a}, x \in [a, b]$ | $a < b$ | Hyperparameter search sampling |

---

## 4. Expectation, Variance, Covariance & Correlation

### 4.1 Expectation (First Moment)
The probability-weighted average of an RV:
$$\mathbb{E}[X] = \begin{cases} \sum_x x \, p(x) & (\text{Discrete}) \\ \int_{-\infty}^\infty x \, f(x) dx & (\text{Continuous}) \end{cases}$$
- **Linearity of Expectation**: $\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$, holds even if $X$ and $Y$ are dependent!

### 4.2 Variance & Standard Deviation (Second Central Moment)
Measures dispersion around the mean:
$$\text{Var}(X) = \sigma^2 = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$
- Properties: $\text{Var}(aX + b) = a^2 \text{Var}(X)$. If $X \perp Y$, $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.

### 4.3 Covariance & Correlation
$$\text{Cov}(X, Y) = \sigma_{XY} = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[XY] - \mu_X \mu_Y$$
- **Pearson Correlation Coefficient ($\rho$)**:
  $$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$
  Measures strictly **linear** relationships ($\rho = 0$ implies no linear correlation, but does *not* imply independence!).
- **Spearman Rank Correlation**: Computes Pearson correlation on ranked variables, capturing monotonic non-linear dependencies.

---

## 5. Sampling Distributions & The Central Limit Theorem (CLT)

### 5.1 The Central Limit Theorem
Let $X_1, X_2, \dots, X_n$ be independent, identically distributed (i.i.d.) random variables with mean $\mu$ and finite variance $\sigma^2 < \infty$.
As sample size $n \to \infty$, the sample mean $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i$ satisfies:
$$\sqrt{n} \left( \frac{\bar{X}_n - \mu}{\sigma} \right) \xrightarrow{d} \mathcal{N}(0, 1)$$

$$\bar{X}_n \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$
- The standard error $\text{SE} = \frac{\sigma}{\sqrt{n}}$ shrinks at rate $\mathcal{O}(1/\sqrt{n})$.
- This guarantees asymptotic normality of sample averages even if underlying populations are heavily skewed, bimodal, or discrete.

---

## 6. Estimation & Confidence Intervals

### 6.1 Confidence Interval (CI) Definition & Proper Interpretation
A $95\%$ confidence interval $[L_n, U_n]$ is a random interval constructed from sample data such that under hypothetical repeated sampling from the population:
$$P(L_n \le \mu \le U_n) = 0.95$$

> **Critical Misconception**: Once a specific numerical interval $[12.4, 15.8]$ is calculated, it is incorrect to say "There is a 95% probability that the true parameter lies in $[12.4, 15.8]$." The parameter $\mu$ is a fixed unknown constant; the interval either contains it or does not. The 95% refers to the long-run coverage frequency of the estimation procedure.

### 6.2 Non-parametric Bootstrap Confidence Intervals
When analytical distribution assumptions fail:
1. Resample $N$ observations with replacement from dataset $\mathcal{D}$ $B$ times (e.g., $B = 2000$).
2. Compute the test statistic $\hat{\theta}^{*(b)}$ on each resample.
3. The empirical $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles form the $95\%$ bootstrap confidence interval.

---

## 7. Hypothesis Testing, p-values & Statistical Significance

### 7.1 Framework: Null ($H_0$) vs. Alternative ($H_1$)
- **Null Hypothesis ($H_0$)**: Baseline assumption of no effect, no difference, or no relationship (e.g., model A is equivalent to model B).
- **Alternative Hypothesis ($H_1$)**: The effect or difference under investigation.

| Decision \ Truth | $H_0$ is True | $H_0$ is False |
| :--- | :--- | :--- |
| **Reject $H_0$** | **Type I Error ($\alpha$, False Positive)** | Correct Decision ($1 - \beta$, Power) |
| **Fail to Reject $H_0$**| Correct Decision ($1 - \alpha$) | **Type II Error ($\beta$, False Negative)** |

### 7.2 What a p-value Actually Is (and Is NOT)
- **Definition**: The probability, assuming the null hypothesis $H_0$ is true, of observing a test statistic at least as extreme as the one computed from the data:
  $$p = P(T(\mathcal{D}) \ge t_{\text{obs}} \mid H_0)$$
- **What it is NOT**:
  - $p$ is NOT the probability that the null hypothesis is true: $p \neq P(H_0 \mid \mathcal{D})$.
  - $p$ is NOT the probability that the finding is a fluke.
  - A small $p$-value ($p < 0.05$) does not measure effect magnitude or practical importance.

---

## 8. Topic Module Resources
- Interactive Lab: [`notebook.ipynb`](./notebook.ipynb)
- Source Implementation: [`code/stats_engine.py`](./code/stats_engine.py)
- Pytest Suite: [`code/test_stats.py`](./code/test_stats.py)
- Interview Drill: [`interview.md`](./interview.md)
- Curated Citations: [`references.md`](./references.md)
