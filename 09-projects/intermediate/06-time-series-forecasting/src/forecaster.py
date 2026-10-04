"""Autoregressive Time-Series Demand Forecaster."""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def generate_load_series(n_hours: int = 1000, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    t = np.arange(n_hours)
    trend = 0.05 * t
    daily = 15.0 * np.sin(2 * np.pi * t / 24.0)
    weekly = 8.0 * np.cos(2 * np.pi * t / 168.0)
    noise = np.random.normal(0, 3.0, n_hours)
    load = 100.0 + trend + daily + weekly + noise
    return pd.Series(load, name="demand")

class TimeSeriesForecaster:
    def __init__(self, lags: Tuple[int, ...] = (1, 2, 24)):
        self.lags = lags
        self.weights = None
        self.bias = 0.0

    def _create_features(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.DataFrame({"y": series})
        for l in self.lags:
            df[f"lag_{l}"] = df["y"].shift(l)
        df["hour_sin"] = np.sin(2 * np.pi * (df.index % 24) / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * (df.index % 24) / 24.0)
        
        df = df.dropna()
        X = df.drop(columns=["y"]).values
        y = df["y"].values
        return X, y

    def fit(self, series: pd.Series):
        X, y = self._create_features(series)
        # Ridge regression
        n_features = X.shape[1]
        A = X.T @ X + 1.0 * np.eye(n_features)
        self.weights = np.linalg.solve(A, X.T @ (y - np.mean(y)))
        self.bias = float(np.mean(y))

    def evaluate(self, series: pd.Series) -> Dict[str, float]:
        X, y = self._create_features(series)
        preds = X @ self.weights + self.bias
        mape = float(np.mean(np.abs((y - preds) / (y + 1e-8))) * 100)
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        return {"mape": mape, "rmse": rmse}

if __name__ == "__main__":
    series = generate_load_series(1000)
    train_s = series.iloc[:800]
    test_s = series.iloc[800:]
    
    forecaster = TimeSeriesForecaster()
    forecaster.fit(train_s)
    m = forecaster.evaluate(test_s)
    print(f"Test Forecast Metrics: MAPE={m['mape']:.2f}% | RMSE={m['rmse']:.2f} kW")
