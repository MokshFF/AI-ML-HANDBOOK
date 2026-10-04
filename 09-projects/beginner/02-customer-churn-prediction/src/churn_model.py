"""Customer Churn Classification Engine."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from churn_data_loader import generate_churn_data

class CustomerChurnModel:
    def __init__(self, lr: float = 0.05, n_epochs: int = 250):
        self.lr = lr
        self.n_epochs = n_epochs
        self.num_means = {}
        self.num_stds = {}
        self.categories = {}
        self.weights = None
        self.bias = 0.0

    def _preprocess(self, df: pd.DataFrame, is_train: bool = True) -> np.ndarray:
        df_copy = df.copy()
        num_cols = ["tenure_months", "monthly_charges", "total_charges", "tech_support_tickets"]
        
        scaled_nums = []
        for col in num_cols:
            if is_train:
                self.num_means[col] = float(df_copy[col].mean())
                self.num_stds[col] = float(df_copy[col].std()) + 1e-8
            val = (df_copy[col].values - self.num_means[col]) / self.num_stds[col]
            scaled_nums.append(val.reshape(-1, 1))
            
        cat_cols = ["contract_type", "payment_method"]
        encoded_cats = []
        for col in cat_cols:
            if is_train:
                self.categories[col] = sorted(df_copy[col].unique().tolist())
            for cat in self.categories[col]:
                encoded_cats.append((df_copy[col].values == cat).astype(float).reshape(-1, 1))
                
        return np.hstack(scaled_nums + encoded_cats)

    def fit(self, df: pd.DataFrame, target_col: str = "churn"):
        X = self._preprocess(df, is_train=True)
        y = df[target_col].values.astype(float)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.n_epochs):
            linear = X @ self.weights + self.bias
            preds = 1.0 / (1.0 + np.exp(-np.clip(linear, -15, 15)))
            
            error = preds - y
            grad_w = (X.T @ error) / n_samples
            grad_b = np.mean(error)
            
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._preprocess(df, is_train=False)
        linear = X @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -15, 15)))

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)

    def evaluate(self, df: pd.DataFrame, target_col: str = "churn", threshold: float = 0.5) -> Dict[str, float]:
        probs = self.predict_proba(df)
        preds = (probs >= threshold).astype(int)
        y = df[target_col].values
        
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        tn = np.sum((preds == 0) & (y == 0))
        
        precision = float(tp / (tp + fp + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))
        accuracy = float((tp + tn) / len(y))
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    df = generate_churn_data(1000)
    train_df = df.iloc[:800]
    test_df = df.iloc[800:]
    
    model = CustomerChurnModel(lr=0.1, n_epochs=300)
    model.fit(train_df)
    m = model.evaluate(test_df)
    print(f"Test Metrics -> Accuracy: {m['accuracy']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f}")
