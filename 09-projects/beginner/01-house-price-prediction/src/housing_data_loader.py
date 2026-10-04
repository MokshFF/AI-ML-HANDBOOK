"""Synthetic Housing Data Generator and Loader."""
import numpy as np
import pandas as pd

def generate_housing_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    sqft = np.random.normal(2000, 500, n_samples).clip(600, 5000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.2, 0.45, 0.2, 0.1])
    bathrooms = np.maximum(1, (bedrooms * 0.75 + np.random.normal(0, 0.5, n_samples)).round(1))
    year_built = np.random.randint(1950, 2024, size=n_samples)
    neighborhoods = np.random.choice(["Suburban", "Downtown", "Rural"], size=n_samples, p=[0.5, 0.3, 0.2])
    garage_cars = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.1, 0.3, 0.45, 0.15])
    
    # Base price calculation with noise
    neigh_mult = {"Suburban": 1.2, "Downtown": 1.6, "Rural": 0.9}
    mult = np.array([neigh_mult[n] for n in neighborhoods])
    
    base_price = (
        sqft * 120.0 +
        bedrooms * 15000.0 +
        bathrooms * 18000.0 +
        (year_built - 1950) * 800.0 +
        garage_cars * 12000.0
    ) * mult
    
    noise = np.random.normal(0, 20000, n_samples)
    sale_price = np.maximum(50000, base_price + noise)
    
    return pd.DataFrame({
        "square_feet": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "neighborhood": neighborhoods,
        "garage_cars": garage_cars,
        "sale_price": sale_price
    })
