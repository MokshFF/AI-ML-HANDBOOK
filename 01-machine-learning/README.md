# Machine Learning (`01-machine-learning`)

## Overview
Classical machine learning theory, supervised/unsupervised algorithms, ensemble methods, and practical workflow fundamentals.

## Subtopics & Navigation
| Directory | Topic | Scope |
| :--- | :--- | :--- |
| [`supervised/`](./supervised/) | **Supervised** | Linear and logistic regression, decision trees, support vector machines, k-nearest neighbors, and naive Bayes. |
| [`unsupervised/`](./unsupervised/) | **Unsupervised** | Clustering algorithms (K-Means, DBSCAN, GMM), dimensionality reduction (PCA, t-SNE, UMAP), and anomaly detection. |
| [`ensemble/`](./ensemble/) | **Ensemble** | Bagging, random forests, boosting (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost), and stacking. |
| [`feature-engineering/`](./feature-engineering/) | **Feature Engineering** | Handling missing data, encoding strategies, scaling, feature creation, selection techniques, and automated transformation. |
| [`model-evaluation/`](./model-evaluation/) | **Model Evaluation** | Cross-validation techniques, classification and regression metrics, ROC-AUC, calibration curves, and bias-variance diagnostics. |
| [`time-series/`](./time-series/) | **Time Series** | ARIMA, SARIMA, exponential smoothing, stationarity tests, lag features, and sequence forecasting techniques. |
| [`recommender-systems/`](./recommender-systems/) | **Recommender Systems** | Collaborative filtering, matrix factorization, content-based recommendation, and two-tower retrieval architectures. |
| [`reinforcement-learning/`](./reinforcement-learning/) | **Reinforcement Learning** | Markov Decision Processes (MDPs), Q-learning, policy gradients, actor-critic methods, and reward modeling. |

## Standard Directory Schema
Every topic directory in this module follows our standard five-component structure:
- `README.md` - Module introduction, learning objectives, and concept matrix
- `notebook.ipynb` - Reproducible, runnable interactive notebook
- `code/` - Clean, modular Python scripts and helper utilities
- `interview.md` - Technical screening questions, edge cases, and design discussions
- `references.md` - Research papers, textbooks, and documentation

## Prerequisites
Before beginning this module, review:
- Foundational math and coding prerequisites in [`../00-prerequisites/`](../00-prerequisites/)
- The end-to-end learning pathways defined in [`../ROADMAP.md`](../ROADMAP.md)
