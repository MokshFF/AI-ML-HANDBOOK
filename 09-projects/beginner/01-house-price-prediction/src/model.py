"""Regression Pipeline for House Price Prediction."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from housing_data_loader import generate_housing_data

class HousePricePredictor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.feature_means = {}
        self.feature_stds = {}
        self.categories = {}
        self.weights = None
        self.bias = 0.0

    def _preprocess(self, df: pd.DataFrame, is_train: bool = True) -> np.ndarray:
        df_copy = df.copy()
        num_cols = ["square_feet", "bedrooms", "bathrooms", "year_built", "garage_cars"]
        
        # Scaling numericals
        scaled_nums = []
        for col in num_cols:
            if is_train:
                self.feature_means[col] = float(df_copy[col].mean())
                self.feature_stds[col] = float(df_copy[col].std()) + 1e-8
            val = (df_copy[col].values - self.feature_means[col]) / self.feature_stds[col]
            scaled_nums.append(val.reshape(-1, 1))
            
        # One-hot encode neighborhood
        if is_train:
            self.categories["neighborhood"] = sorted(df_copy["neighborhood"].unique().tolist())
        
        cats = []
        for cat in self.categories["neighborhood"]:
            cats.append((df_copy["neighborhood"].values == cat).astype(float).reshape(-1, 1))
            
        X = np.hstack(scaled_nums + cats)
        return X

    def fit(self, df: pd.DataFrame, target_col: str = "sale_price"):
        X = self._preprocess(df, is_train=True)
        y = df[target_col].values
        
        # Ridge regression closed-form: w = (X^T X + alpha * I)^(-1) X^T (y - mean_y)
        self.mean_y = float(np.mean(y))
        y_centered = y - self.mean_y
        
        n_features = X.shape[1]
        A = X.T @ X + self.alpha * np.eye(n_features)
        self.weights = np.linalg.solve(A, X.T @ y_centered)
        self.bias = self.mean_y

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._preprocess(df, is_train=False)
        return X @ self.weights + self.bias

    def evaluate(self, df: pd.DataFrame, target_col: str = "sale_price") -> Dict[str, float]:
        preds = self.predict(df)
        y = df[target_col].values
        mae = float(np.mean(np.abs(y - preds)))
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum((y - preds) ** 2))
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
        return {"mae": mae, "rmse": rmse, "r2": r2}

if __name__ == "__main__":
    df = generate_housing_data(1000)
    train_df = df.iloc[:800]
    test_df = df.iloc[800:]
    
    predictor = HousePricePredictor(alpha=10.0)
    predictor.fit(train_df)
    metrics = predictor.evaluate(test_df)
    print(f"Test Metrics: MAE=${metrics['mae']:,.2f} | RMSE=${metrics['rmse']:,.2f} | R2={metrics['r2']:.4f}")
