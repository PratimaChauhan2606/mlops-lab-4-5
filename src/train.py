import argparse
import yaml
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from mlflow.models.signature import infer_signature

from data_preprocessing import load_data, split_data
from evaluate import compute_metrics, save_residual_plot, save_feature_importance_plot

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path, n_estimators=None, max_depth=None, run_name=None):
    config = load_config(config_path)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Allow command-line overrides so we can train several candidates easily
    n_estimators = n_estimators or config["model"]["n_estimators"]
    max_depth = max_depth or config["model"]["max_depth"]

    with mlflow.start_run(run_name=run_name):
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            test_size=config["training"]["test_size"],
            random_state=config["training"]["random_state"],
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=config["training"]["random_state"],
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        print(f"n_estimators={n_estimators}, max_depth={max_depth} -> Metrics:", metrics)

        save_residual_plot(y_test, y_pred, str(ROOT / "reports/figures/residual_plot.png"))
        save_feature_importance_plot(model, X.columns, str(ROOT / "reports/figures/feature_importance.png"))
        joblib.dump(model, str(ROOT / "models/model.pkl"))

        # Parameters
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", config["training"]["test_size"])
        mlflow.log_param("random_state", config["training"]["random_state"])

        # Metrics
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("r2", metrics["r2"])

        # Artifacts
        mlflow.log_artifact(str(ROOT / "reports/figures/residual_plot.png"))
        mlflow.log_artifact(str(ROOT / "reports/figures/feature_importance.png"))

        # Model signature + input example (Activity A4)
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]

        # Log AND register the model in one step (Activity A5 — creates a new version
        # each time this runs, under the same registered model name)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=config["mlflow"]["registered_model_name"],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print("Run complete. Model logged and registered to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args.n_estimators, args.max_depth, args.run_name)