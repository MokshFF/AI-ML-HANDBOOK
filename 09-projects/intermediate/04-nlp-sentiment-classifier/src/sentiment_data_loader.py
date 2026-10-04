"""Synthetic NLP Sentiment Dataset."""
import numpy as np
from typing import List, Tuple

POSITIVE_PHRASES = [
    "great product loved it", "super fast shipping highly recommend", "excellent quality works well",
    "very pleased with this purchase", "fantastic customer support", "amazing value for money",
    "best purchase of the year", "exceeded my expectations completely", "reliable and sturdy build"
]

NEGATIVE_PHRASES = [
    "terrible quality broke on day one", "awful customer service", "waste of money do not buy",
    "slow shipping and damaged box", "defective item refund refused", "worst experience ever",
    "completely useless product", "poorly made very disappointed", "failed to work right away"
]

def generate_sentiment_data(n_samples: int = 400, seed: int = 42) -> Tuple[List[str], List[int]]:
    np.random.seed(seed)
    texts = []
    labels = []
    for _ in range(n_samples // 2):
        texts.append(np.random.choice(POSITIVE_PHRASES))
        labels.append(1)
        texts.append(np.random.choice(NEGATIVE_PHRASES))
        labels.append(0)
    return texts, labels
