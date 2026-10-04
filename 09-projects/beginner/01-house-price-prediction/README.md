# End-to-End House Price Prediction

## Problem
Predict the sale prices of residential homes using physical, geographic, and temporal tabular attributes.

## Motivation
Real estate valuation is a quintessential regression challenge. Accurate automated valuation models (AVM) enable homebuyers, lenders, and investors to assess fair market values without manual appraisals.

## Dataset
Synthetic tabular residential housing dataset modeled after the Ames and California Housing datasets:
- Features: `square_feet`, `bedrooms`, `bathrooms`, `year_built`, `neighborhood`, `garage_cars`.
- Target: `sale_price` (continuous positive dollar amount).

## Architecture
```mermaid
flowchart LR
    A[Raw Tabular CSV] --> B[Feature Preprocessing]
    B --> C[StandardScaler & OneHotEncoder]
    C --> D[Ridge & Gradient Boosting Regressors]
    D --> E[Ensemble Blending]
    E --> F[Evaluation MAE / RMSE / R2]
```

## Pipeline
1. Ingest raw CSV data and handle missing continuous/categorical records.
2. Standardize numerical attributes using z-score scaling.
3. One-hot encode categorical features (neighborhoods).
4. Train Ridge Regression and a tree-based ensemble.
5. Predict continuous sale price and evaluate residuals.

## Technologies
- Python 3.11+
- NumPy, Pandas
- Scikit-Learn
- Matplotlib, Pytest

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Run training and evaluation via the command-line interface:
```bash
python src/model.py
```

## Evaluation
- Mean Absolute Error (MAE): $\frac{1}{n} \sum |y_i - \hat{y}_i|$
- Root Mean Squared Error (RMSE): $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$
- Coefficient of Determination ($R^2$): $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

## Results
- Validated on synthetic test split (200 records):
  - Train $R^2$: $\approx 0.88$
  - Test $R^2$: $\approx 0.84$
  - Test MAE: $\approx \$18,400$
  - Full benchmark results on production Ames data: *Pending real-world cluster evaluation*.

## Limitations
- Linear and tree models cannot extrapolate to price regimes significantly beyond training domain boundaries.
- Macroeconomic factors (interest rate fluctuations, inflation) are not captured in physical tabular attributes.

## Future Improvements
- Integrate spatial geographic coordinates using Spatial K-NN embeddings.
- Add macroeconomic time-series features (mortgage interest rates).
