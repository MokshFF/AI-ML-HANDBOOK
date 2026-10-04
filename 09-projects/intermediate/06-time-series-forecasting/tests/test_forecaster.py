"""Tests for Time-Series Forecaster."""
from forecaster import generate_load_series, TimeSeriesForecaster

def test_forecaster_pipeline():
    s = generate_load_series(500)
    f = TimeSeriesForecaster()
    f.fit(s.iloc[:400])
    m = f.evaluate(s.iloc[400:])
    assert m["mape"] < 10.0
    assert m["rmse"] < 20.0
