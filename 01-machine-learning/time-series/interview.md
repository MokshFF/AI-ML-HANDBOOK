# Time Series Analysis: Technical Interview Question Bank

Technical screening questions, temporal validation dilemmas, stationarity mechanics, and time series machine learning architectures.

---

## 1. Temporal Validation & Stationarity

### Q1: Why does standard K-Fold cross-validation fail completely on time-series data, and what is the proper validation setup?
- **Answer Outline**:
  - Time series data violates the core i.i.d. assumption due to temporal autocorrelation ($y_t$ depends on $y_{t-1}$).
  - Random K-Fold randomly shuffles rows across folds. If row $t=100$ is in the training set and row $t=90$ is in the test set, the model uses future information to predict the past (**lookahead bias**), creating severe data leakage and artificially inflated metrics.
  - **Proper Setup (Walk-Forward / Expanding Window)**:
    - Folds must be chronologically ordered.
    - Fold $k$ trains strictly on history $[t_0 \dots t_k]$ and validates strictly on future unseen window $[t_{k+1} \dots t_{k+H}]$.
    - If training on rolling window, fixed horizon $[t_{k-W} \dots t_k]$ shifts forward in time.

### Q2: Why must a time series be made stationary before fitting an ARMA model, and how do you achieve stationarity?
- **Answer Outline**:
  - **Why Stationarity is Required**:
    - ARMA models assume the underlying statistical parameters (mean $\mu$, variance $\sigma^2$, and autocorrelation $\gamma(k)$) are constant over time. If a series has an upward trend, future values will have an entirely different mean than historical training data, causing model forecasts to diverge or under-predict systematically.
  - **How to Achieve Stationarity**:
    1. **Differencing**: $y'_t = y_t - y_{t-1}$ removes linear trends. Seasonal differencing $y_t - y_{t-s}$ removes periodic cycles.
    2. **Log / Box-Cox Transformation**: Stabilizes non-constant, expanding variance (heteroscedasticity).
    3. **Detrending**: Fitting a deterministic polynomial regression curve to time $t$ and subtracting it.

---

## 2. Statistical Frameworks vs. Tabular ML

### Q3: Why do Gradient Boosted Decision Trees (like LightGBM or XGBoost) fail to extrapolate linear trends in time-series forecasting, and how do you fix it?
- **Answer Outline**:
  - **Failure Mechanism**: Decision trees perform orthogonal axis-aligned splits on feature thresholds ($x_j \le \theta$). A leaf node outputs a constant scalar prediction (the mean target of training samples in that leaf). If a time feature $t$ enters values greater than the maximum training timestamp $t_{\max}$, the tree routes all future queries into the extreme right leaf, predicting a flat constant line into infinity rather than continuing the trend.
  - **Fixes**:
    1. **De-trend First**: Train a linear model on trend $t$ or difference the series to make it mean-stationary, and train the GBDT strictly on stationary residuals.
    2. **Lag Relative Features**: Train the model strictly on relative percentage changes ($\frac{y_{t-1} - y_{t-2}}{y_{t-2}}$) and rolling ratios rather than absolute timestamp indices.

### Q4: Explain the difference between MASE (Mean Absolute Scaled Error) and MAPE. Why is MASE preferred?
- **Answer Outline**:
  - **MAPE**: $\frac{100\%}{H} \sum \left| \frac{y_t - \hat{y}_t}{y_t} \right|$.
    - *Disadvantages*: Undefined when $y_t = 0$; penalizes over-forecasts vastly more heavily than under-forecasts (asymmetric).
  - **MASE**: Compares model MAE against the in-sample naive persistence forecast ($y_t = y_{t-1}$):
    $$\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\frac{1}{N-1} \sum_{i=2}^N |y_i - y_{i-1}|}$$
    - *Advantages*: Scale-free, defined for zeros, symmetric, and immediately interpretable: $\text{MASE} < 1$ indicates the model is genuinely learning meaningful patterns beyond simply repeating yesterday's value.

---

## 3. Coding Drill: Vectorized Cyclical Feature Encoding

### Task
Implement an encoder function converting timestamp hour (0-23) and day of week (0-6) into cyclical trigonometric features preserving modular continuity.

```python
import numpy as np

def encode_cyclical_features(hours: np.ndarray, days: np.ndarray) -> np.ndarray:
    # 24 hours per day, 7 days per week
    sin_hour = np.sin(2.0 * np.pi * hours / 24.0)
    cos_hour = np.cos(2.0 * np.pi * hours / 24.0)

    sin_day = np.sin(2.0 * np.pi * days / 7.0)
    cos_day = np.cos(2.0 * np.pi * days / 7.0)

    return np.column_stack([sin_hour, cos_hour, sin_day, cos_day])
```
