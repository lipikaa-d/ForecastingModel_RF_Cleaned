# src/model_inference.py
import pandas as pd
import joblib
import os
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_lag_features


def load_model(model_path='model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_latest(df, model, target_col='LOAD', num_lags=5):
    # Generate lag features
    df_with_lags = create_lag_features(df, target_col=target_col, num_lags=num_lags)

    # Select the last row for prediction
    latest_row = df_with_lags.tail(1)

    # Prepare feature columns
    feature_cols = ['P_IN', 'T_IN', 'P_OUT', 'T_OUT'] + [f'{target_col}_t-{i}' for i in range(1, num_lags + 1)]
    X_latest = latest_row[feature_cols]

    # Predict
    predicted_load = model.predict(X_latest)[0]
    return predicted_load


if __name__ == '__main__':
    model = load_model('../model.pkl')
    df = load_and_clean_data('../data/combinedddddd_dataset.xlsx')
    prediction = predict_latest(df, model, target_col='LOAD', num_lags=5)
    print(f"Predicted next LOAD: {prediction:.4f}")
