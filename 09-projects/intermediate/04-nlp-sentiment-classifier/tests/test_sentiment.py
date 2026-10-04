"""Tests for NLP Sentiment Engine."""
from sentiment_data_loader import generate_sentiment_data
from sentiment_engine import SentimentPipeline

def test_sentiment_pipeline():
    texts, labels = generate_sentiment_data(200)
    pipe = SentimentPipeline(lr=0.3, epochs=100)
    pipe.fit(texts[:160], labels[:160])
    
    m = pipe.evaluate(texts[160:], labels[160:])
    assert m["accuracy"] > 0.75
    
    sample_preds, sample_probs = pipe.predict(["loved it amazing quality", "terrible waste of money"])
    assert sample_preds[0] == 1
    assert sample_preds[1] == 0
