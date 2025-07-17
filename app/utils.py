import pandas as pd
import joblib
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import prepare_features_and_target, create_lag_features

MODEL_PATH = 'model.pkl'
DATA_PATH = 'data/combinedddddd_dataset.xlsx'


def load_trained_model():
    return joblib.load(MODEL_PATH)


def get_latest_input_data():
    df = load_and_clean_data(DATA_PATH)
    latest_row = df.iloc[-1]
    return latest_row


def forecast_from_manual_input(model, manual_input_dict, steps):
    forecast = []

    current_input = pd.DataFrame([manual_input_dict])
    current_input = current_input[['P_IN', 'T_IN', 'P_OUT', 'T_OUT',
                                   'LOAD_t-1', 'LOAD_t-2', 'LOAD_t-3', 'LOAD_t-4', 'LOAD_t-5']]

    for _ in range(steps):
        next_load = model.predict(current_input)[0]
        forecast.append(next_load)

        for i in range(5, 1, -1):
            current_input[f'LOAD_t-{i}'] = current_input[f'LOAD_t-{i-1}']
        current_input['LOAD_t-1'] = next_load

    return forecast


def forecast_next_steps(model, steps):
    df = load_and_clean_data(DATA_PATH)
    df_with_lags = create_lag_features(df, target_col='LOAD', num_lags=5)
    input_df = df_with_lags.iloc[[-1]][['P_IN', 'T_IN', 'P_OUT', 'T_OUT',
                                        'LOAD_t-1', 'LOAD_t-2', 'LOAD_t-3', 'LOAD_t-4', 'LOAD_t-5']]

    forecast = []
    for _ in range(steps):
        pred = model.predict(input_df)[0]
        forecast.append(pred)
        
        new_row = input_df.copy()
        for i in range(5, 1, -1):
            new_row[f'LOAD_t-{i}'] = new_row[f'LOAD_t-{i-1}']
        new_row['LOAD_t-1'] = pred
        input_df = new_row

    return forecast

