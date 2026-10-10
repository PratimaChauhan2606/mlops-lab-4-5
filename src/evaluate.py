import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def save_residual_plot(y_true, y_pred, path):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    plt.title("Residual plot")
    plt.savefig(path)
    plt.close()


def save_feature_importance_plot(model, feature_names, path):
    importances = model.feature_importances_
    plt.figure()
    plt.bar(feature_names, importances)
    plt.ylabel("Importance")
    plt.title("Feature importance")
    plt.savefig(path)
    plt.close()