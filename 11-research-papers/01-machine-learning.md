# Seminal Research Papers: Machine Learning

Annotated compendium of foundational machine learning papers establishing statistical learning theory, ensemble methods, and scalable gradient boosting.

---

## 1. Support-Vector Networks
- **Title**: Support-Vector Networks
- **Authors**: Corinna Cortes, Vladimir Vapnik
- **Year**: 1995
- **Link**: https://doi.org/10.1007/BF00994018
- **Problem**: Constructing high-capacity binary classifiers that maximize margin separation while avoiding overfitting on limited dimensional sample sets.
- **Main Idea**: Map input vectors non-linearly into a high-dimensional feature space where an optimal separating hyperplane is constructed with maximal margin.
- **Key Contribution**: Introduced the soft-margin formulation with slack variables $\xi_i$ to handle non-separable overlapping class distributions.
- **Important Architecture/Math**:
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(w^T \Phi(x_i) + b) \ge 1 - \xi_i, \; \xi_i \ge 0$$
- **Why It Matters**: Formed the gold standard for high-dimensional classification (genomics, text categorization) prior to the deep learning revolution.
- **Prerequisites**: Linear algebra, convex optimization, Lagrange multipliers, Mercer's theorem.
- **Suggested Follow-up Papers**: *A Training Algorithm for Optimal Margin Classifiers* (Boser et al., 1992); *Sequential Minimal Optimization* (Platt, 1998).

---

## 2. Greedy Function Approximation: A Gradient Boosting Machine
- **Title**: Greedy Function Approximation: A Gradient Boosting Machine
- **Authors**: Jerome H. Friedman
- **Year**: 2001
- **Link**: https://doi.org/10.1214/aos/1013203451
- **Problem**: Ensembling weak learners under arbitrary differentiable loss functions beyond exponential loss (AdaBoost).
- **Main Idea**: Treat boosting as numerical optimization in function space, iteratively fitting new base-learners to the negative gradient (pseudo-residuals) of the loss function.
- **Key Contribution**: Formalized gradient boosting for regression, multiclass classification, and ranking with regularization via shrinkage (learning rate).
- **Important Architecture/Math**:
  $$\tilde{y}_i = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}, \quad F_m(x) = F_{m-1}(x) + \nu \rho_m h_m(x)$$
- **Why It Matters**: Underpins modern competitive tabular ML systems across global industries (XGBoost, LightGBM, CatBoost).
- **Prerequisites**: Multivariable calculus, gradient descent, decision tree induction.
- **Suggested Follow-up Papers**: *XGBoost: A Scalable Tree Boosting System* (Chen & Guestrin, 2016); *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* (Ke et al., 2017).
