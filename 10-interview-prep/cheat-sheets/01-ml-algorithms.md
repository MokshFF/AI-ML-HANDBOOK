# Cheat Sheet: Machine Learning Algorithms

Dense comparison matrix of classical supervised and unsupervised machine learning algorithms.

| Algorithm | Type | Objective Function | Key Hyperparameters | Primary Assumptions | Failure Modes / Pitfalls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Reg | $\min \sum (y_i - w^T x_i)^2$ | $\alpha$ (L1/L2 penalty) | Linearity, homoscedasticity, no multicollinearity | Severe bias under non-linear manifolds |
| **Logistic Regression** | Cls | $\min -\sum [y_i \log p_i + (1-y_i) \log(1-p_i)]$ | $C$ (inverse reg strength), penalty | Linear log-odds boundary | High dimensional collinearity |
| **Ridge Regression** | Reg | $\min \|y - Xw\|^2 + \lambda \|w\|_2^2$ | $\alpha$ / $\lambda$ | Normal Gaussian prior on weights | Does not yield sparse weights |
| **Lasso Regression** | Reg | $\min \|y - Xw\|^2 + \lambda \|w\|_1$ | $\alpha$ / $\lambda$ | Laplace prior on weights | Arbitrary selection among collinear features |
| **ElasticNet** | Reg | $\min \|y - Xw\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$ | $\alpha$, $l_1\_ratio$ | Balance sparsity and grouping | Slower convergence than pure Ridge |
| **k-Nearest Neighbors** | Both | Non-parametric majority vote / mean | $k$, distance metric ($L_1, L_2$) | Local smoothness | Curse of dimensionality; $\mathcal{O}(ND)$ test inference |
| **Decision Tree (CART)** | Both | Maximize Information Gain / Gini reduction | `max_depth`, `min_samples_split` | Orthogonal axis-aligned splits | Severe overfitting; high variance |
| **Random Forest** | Both | Bagging: Average of de-correlated trees | `n_estimators`, `max_features` | Low correlation across trees | Slow inference; cannot extrapolate trends |
| **Gradient Boosting** | Both | Sequential pseudo-residual minimization | `learning_rate`, `n_estimators`, `max_depth` | Additive weak learners | Sensitive to outliers and noise |
| **Support Vector Machine** | Both | Maximize margin: $\min \frac{1}{2} \|w\|^2 + C \sum \xi_i$ | $C$, kernel (`rbf`, `poly`, `linear`), $\gamma$ | Separability in kernel Hilbert space | $\mathcal{O}(N^3)$ computational scaling |
| **K-Means Clustering** | Clust | $\min \sum_{k} \sum_{i \in S_k} \|x_i - \mu_k\|^2$ | $k$, `init='k-means++'` | Spherical clusters, equal variance | Fails on non-convex shapes, sensitive to scales |
| **PCA** | DimRed | Maximize variance: $X^T X v = \lambda v$ | `n_components` | Linear subspace variance captures information | Ignores non-linear manifold structure |
