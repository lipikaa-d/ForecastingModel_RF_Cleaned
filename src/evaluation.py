# src/model_evaluation.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import prepare_features_and_target

def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.pkl')
    return joblib.load(model_path)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float('inf') 
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    mean_actual = np.mean(y_test)
    rmse_percent = (rmse / mean_actual) * 100 if mean_actual != 0 else float('inf')

    return {
        'r2_score': round(r2, 4),
        'rmse': round(rmse, 4),
        'rmse_percent': round(rmse_percent, 2),
        'mae': round(mae, 4),
        'mape': round(mape, 2),
        'y_test': y_test.tolist(),
        'y_pred': predictions.tolist()
    }


def get_evaluation_metrics():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'combinedddddd_dataset.xlsx')
    df = load_and_clean_data(data_path)
    X, y = prepare_features_and_target(df, target_col='LOAD', num_lags=5)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = load_model()
    return evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    metrics = get_evaluation_metrics()
    print("Model Evaluation Metrics:")
    print(f"R² Score     : {metrics['r2_score']}")
    print(f"RMSE         : {metrics['rmse']}")
    print(f"RMSE (%)     : {metrics['rmse_percent']}%")
    print(f"MAE          : {metrics['mae']}")
    print(f"MAPE         : {metrics['mape']}%")
