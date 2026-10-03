# Ensemble Learning: Technical Interview Question Bank

Technical screening questions, architectural comparisons, mathematical derivations, and production trade-offs in ensemble systems.

---

## 1. Bagging vs. Boosting

### Q1: Why do Random Forests use fully grown, deep decision trees, while Gradient Boosting uses shallow trees?
- **Answer Outline**:
  - **Random Forest (Variance Reduction)**: Individual unpruned trees have low bias (high expressive capacity) but high variance. Bagging averages $B$ trees, which drives variance down proportionally to $\frac{1 - \rho}{B} + \rho$, while leaving the low bias intact. Using shallow trees in Random Forest would result in an ensemble with high bias that averaging cannot fix.
  - **Gradient Boosting (Bias Reduction)**: Boosting builds an additive model sequentially ($F_m = F_{m-1} + \nu h_m$). Shallow trees (typically depth 3-6) have high bias and low variance. Because boosting sequentially reduces residual bias at each step, starting with low-variance base learners prevents the ensemble from overfitting early errors and noise.

### Q2: Can increasing the number of trees in a Random Forest cause it to overfit? What about Gradient Boosting?
- **Answer Outline**:
  - **Random Forest**: **No**. By the Strong Law of Large Numbers, as $B \to \infty$, the average generalization error of Random Forest converges almost surely to an asymptotic upper bound. Adding more trees does not cause overfitting; it merely reduces variance until reaching the correlation floor $\rho \sigma^2$.
  - **Gradient Boosting**: **Yes**. Because boosting greedily fits residuals, continuing to add trees with high learning rates will eventually cause the ensemble to fit idiosyncratic noise in the training set, leading to severe overfitting. Early stopping and shrinkage $\nu$ are essential.

---

## 2. GBDT Engine Architecture: XGBoost vs. LightGBM vs. CatBoost

### Q3: How does LightGBM's Leaf-Wise tree growth differ from XGBoost's traditional Level-Wise growth, and what is the associated trade-off?
- **Answer Outline**:
  - **Level-Wise (Depth-Wise) Growth (XGBoost baseline)**: Splits all leaves at the current depth simultaneously before moving to the next level. Produces balanced, symmetric trees.
  - **Leaf-Wise (Best-First) Growth (LightGBM)**: At each step, scans all existing leaves and splits the single leaf that achieves the largest reduction in global loss, regardless of its depth in the tree.
  - **Trade-off**: Leaf-wise growth achieves substantially lower loss for a fixed number of splits and converges faster. However, it can produce deep asymmetric branches on noisy features, increasing the risk of overfitting on small datasets. This is controlled using `max_depth` and `min_data_in_leaf`.

### Q4: Explain how Out-of-Bag (OOB) error is computed in Random Forest and why it serves as an unbiased validation estimate.
- **Answer Outline**:
  - In bootstrapping with replacement, each sample has probability $(1 - 1/N)^N \approx 1/e \approx 36.8\%$ of being excluded from any given tree's training set.
  - For observation $\mathbf{x}_i$, we aggregate predictions *only* across the $\approx 0.368 \times B$ trees where $\mathbf{x}_i$ was out-of-bag.
  - Because none of these trees saw $\mathbf{x}_i$ during training, the resulting prediction is effectively an out-of-sample evaluation. Empirical studies have proven OOB error is nearly identical to $K$-fold cross-validation error, providing cross-validation for zero additional compute cost.

---

## 3. Stacking & Meta-Learning

### Q5: Why is K-fold cross-validation strictly required when generating meta-features for Stacking?
- **Answer Outline**:
  - If base models make predictions directly on the training data they were fit on, their predictions will be overly optimistic (overfitted).
  - A complex Level-1 meta-learner would simply learn to trust whichever base model overfit the training data most aggressively (e.g., an unpruned tree with 100% training accuracy), rather than learning a balanced blending of diverse generalizations.
  - **Out-of-Fold Generation**: Training base models on $K-1$ folds and predicting on the held-out fold ensures meta-features reflect realistic out-of-sample prediction distributions, preventing target leakage into the meta-learner.

---

## 4. Coding Drill: Out-of-Fold Stacking Feature Generation

### Task
Implement a function generating out-of-fold meta-features for a list of base estimators.

```python
import numpy as np
from sklearn.model_selection import KFold
from typing import Any

def generate_oof_features(base_models: list[Any], X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    n_samples = len(X)
    n_models = len(base_models)
    oof_features = np.zeros((n_samples, n_models))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]

        for m_idx, model in enumerate(base_models):
            clf = type(model)(**model.get_params())
            clf.fit(X_tr, y_tr)
            oof_features[val_idx, m_idx] = clf.predict(X_val)

    return oof_features
```
