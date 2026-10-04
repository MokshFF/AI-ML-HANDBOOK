"""TF-IDF and Linear Classifier NLP Sentiment Engine."""
import math
import numpy as np
from typing import List, Dict, Tuple
from sentiment_data_loader import generate_sentiment_data

class SentimentPipeline:
    def __init__(self, max_features: int = 500, lr: float = 0.2, epochs: int = 150):
        self.max_features = max_features
        self.lr = lr
        self.epochs = epochs
        self.vocab = {}
        self.idf = {}
        self.weights = None
        self.bias = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return [w.strip() for w in text.lower().split() if len(w.strip()) > 1]

    def fit(self, texts: List[str], labels: List[int]):
        # Build Vocab
        df_counts = {}
        n_docs = len(texts)
        for t in texts:
            words = set(self._tokenize(t))
            for w in words:
                df_counts[w] = df_counts.get(w, 0) + 1
                
        sorted_words = sorted(df_counts.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        self.vocab = {w: i for i, (w, _) in enumerate(sorted_words)}
        self.idf = {w: math.log((1 + n_docs) / (1 + count)) + 1.0 for w, count in sorted_words}
        
        X = self._transform(texts)
        y = np.array(labels, dtype=float)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            preds = 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
            err = preds - y
            self.weights -= self.lr * ((X.T @ err) / n_samples + 0.01 * self.weights)
            self.bias -= self.lr * np.mean(err)

    def _transform(self, texts: List[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=float)
        for i, t in enumerate(texts):
            tokens = self._tokenize(t)
            if not tokens:
                continue
            tf = {}
            for w in tokens:
                if w in self.vocab:
                    tf[w] = tf.get(w, 0) + 1
            for w, count in tf.items():
                idx = self.vocab[w]
                matrix[i, idx] = (count / len(tokens)) * self.idf[w]
                
        # L2 norm per row
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return matrix / norms

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = self._transform(texts)
        z = X @ self.weights + self.bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        preds, probs = self.predict(texts)
        y = np.array(labels)
        acc = float(np.mean(preds == y))
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        p = float(tp / (tp + fp + 1e-8))
        r = float(tp / (tp + fn + 1e-8))
        f1 = float(2 * p * r / (p + r + 1e-8))
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

if __name__ == "__main__":
    texts, labels = generate_sentiment_data(400)
    train_txt, test_txt = texts[:300], texts[300:]
    train_y, test_y = labels[:300], labels[300:]
    
    pipe = SentimentPipeline()
    pipe.fit(train_txt, train_y)
    m = pipe.evaluate(test_txt, test_y)
    print(f"Sentiment Evaluation: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")
