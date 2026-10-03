# Feature Engineering: Technical Interview Question Bank

Technical screening questions, production preprocessing architectures, feature selection trade-offs, and data leakage diagnostics.

---

## 1. Missingness & Transformations

### Q1: What is the difference between MCAR, MAR, and MNAR, and how does your imputation strategy change for each?
- **Answer Outline**:
  - **MCAR (Missing Completely at Random)**: The probability of missingness is completely unrelated to any observed or unobserved variable. Simple mean, median, or random imputation introduces no systematic estimation bias.
  - **MAR (Missing at Random)**: Missingness depends on observed features (e.g., younger users are less likely to report home ownership). Imputation should condition on observed features (MICE / Iterative Imputer, KNN imputer).
  - **MNAR (Missing Not at Random)**: Missingness depends directly on the unobserved value itself (e.g., severe depression patients dropping out of clinical surveys). Imputation cannot restore missing information; you **must** append a binary missingness indicator column ($\mathbb{I}(x_{ij} = \text{NaN})$) so models can learn patterns from the missingness event itself.

### Q2: Why is RobustScaler preferred over StandardScaler when data contains extreme outliers?
- **Answer Outline**:
  - **StandardScaler**: Computes $z = \frac{x - \mu}{\sigma}$. Both sample mean $\mu$ and sample standard deviation $\sigma$ are non-robust statistics with a **breakdown point of 0%** (a single extreme outlier can shift $\mu$ and artificially inflate $\sigma$ toward infinity). Consequently, non-outlier points are compressed into an infinitesimally narrow band.
  - **RobustScaler**: Computes $z = \frac{x - \text{median}}{\text{IQR}}$. Both median and Interquartile Range (IQR = $Q_3 - Q_1$) have a **breakdown point of 25-50%**, meaning up to 25% of the data can be arbitrarily corrupted without distorting the scaling parameters.

---

## 2. Categorical Encodings & Leakage Prevention

### Q3: What is Target Encoding, why does it overfit without smoothing, and how do you prevent target leakage?
- **Answer Outline**:
  - **Target Encoding**: Replaces categorical level $c$ with the sample mean of the target variable among instances with category $c$: $\bar{y}_c = \frac{1}{n_c} \sum_{i \in c} y_i$.
  - **Overfitting Risk**: For rare categories (e.g., $n_c = 1$), if that single sample has $y=1$, target encoding assigns a score of 1.0, memorizing the training target.
  - **Mitigation (Smoothing & Out-of-Fold)**:
    1. **Empirical Bayes Smoothing**: Shrink small categories toward the global prior: $S_c = \frac{n_c}{n_c + m} \bar{y}_c + \frac{m}{n_c + m} \bar{y}_{\text{global}}$.
    2. **K-Fold Target Encoding**: Compute target means strictly out-of-fold during training, ensuring a sample's own label never influences its encoded feature value.

### Q4: Describe an insidious form of data leakage in feature selection and how you would diagnose it.
- **Answer Outline**:
  - **Scenario**: A practitioner performs ANOVA F-test or Mutual Information feature selection on the complete dataset of 1,000,000 rows to pick the top 50 features, and *then* runs 5-fold cross-validation.
  - **The Leakage**: Selecting features using the entire dataset allows the selection algorithm to pick features that happen to correlate with the test set targets by random chance, artificially inflating validation accuracy.
  - **Diagnosis**: Run the entire pipeline on pure synthetic Gaussian noise ($X \sim \mathcal{N}(0, I), y \sim \text{Bernoulli}(0.5)$). A leaked pipeline will report $\approx 70-80\%$ accuracy on pure random noise; a proper pipeline wrapped inside `sklearn.pipeline.Pipeline` will report exactly the true baseline of $50\%$.

---

## 3. Coding Drill: Out-of-Fold Target Encoding

### Task
Implement a function computing out-of-fold target encodings with additive smoothing.

```python
import numpy as np
from sklearn.model_selection import KFold

def kfold_target_encode(categories: np.ndarray, y: np.ndarray, smoothing: float = 10.0, n_splits: int = 5) -> np.ndarray:
    n = len(categories)
    encoded = np.zeros(n)
    global_mean = float(np.mean(y))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(categories):
        cats_tr, y_tr = categories[train_idx], y[train_idx]
        cats_val = categories[val_idx]

        # Calculate smoothed map on training fold
        unique_c, counts = np.unique(cats_tr, return_counts=True)
        cat_map = {}
        for c, count in zip(unique_c, counts):
            mean_c = np.mean(y_tr[cats_tr == c])
            weight = count / (count + smoothing)
            cat_map[c] = weight * mean_c + (1.0 - weight) * global_mean

        encoded[val_idx] = [cat_map.get(c, global_mean) for c in cats_val]

    return encoded
```
