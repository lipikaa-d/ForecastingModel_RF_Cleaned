import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # ✅ Added
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import prepare_features_and_target


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float('inf')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_and_save_model(data_path, model_path=None, num_lags=5):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.pkl')

    # Load and prepare data
    df = load_and_clean_data(data_path)
    X, y = prepare_features_and_target(df, target_col='LOAD', num_lags=num_lags)

    # ✅ Random train-test split (10% test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mean_actual = np.mean(y_test)
    rmse_percent = (rmse / mean_actual) * 100 if mean_actual != 0 else float('inf')
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Save model
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    print("Evaluation Metrics:")
    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"RMSE (%) : {rmse_percent:.2f}%")
    print(f"MAE      : {mae:.4f}")
    print(f"MAPE     : {mape:.2f}%")

    return model, r2, rmse, mae, rmse_percent, mape


# Run manually
if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'combinedddddd_dataset.xlsx')
    train_and_save_model(data_path)
