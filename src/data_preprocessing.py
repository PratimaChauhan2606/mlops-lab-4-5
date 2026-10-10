import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"


def generate_dataset(n_samples=500, random_state=42):
    """Generates a small synthetic house-price dataset and saves it as CSV."""
    rng = np.random.RandomState(random_state)
    area = rng.randint(500, 4000, n_samples)
    bedrooms = rng.randint(1, 6, n_samples)
    age = rng.randint(0, 40, n_samples)
    distance_to_city = rng.uniform(1, 30, n_samples)

    price = (
        area * 120
        + bedrooms * 50000
        - age * 800
        - distance_to_city * 1000
        + rng.normal(0, 20000, n_samples)
    )

    df = pd.DataFrame({
        "area": area,
        "bedrooms": bedrooms,
        "age": age,
        "distance_to_city": distance_to_city,
        "price": price,
    })
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_data():
    if not DATA_PATH.exists():
        generate_dataset()
    df = pd.read_csv(DATA_PATH)
    X = df[["area", "bedrooms", "age", "distance_to_city"]]
    y = df["price"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)