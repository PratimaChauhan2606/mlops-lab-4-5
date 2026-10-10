import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

from data_preprocessing import load_data, split_data

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("house-price-regression")

mlflow.sklearn.autolog()

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="autolog-demo"):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

print("Autolog run complete.")