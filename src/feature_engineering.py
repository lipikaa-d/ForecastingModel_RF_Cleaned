import pandas as pd
from src.data_preprocessing import load_and_clean_data


def prepare_features_and_target(df, target_col='LOAD', num_lags=5):
    df_with_lags = create_lag_features(df, target_col, num_lags)
    feature_cols = ['P_IN', 'T_IN', 'P_OUT', 'T_OUT'] + [f'{target_col}_t-{i}' for i in range(1, num_lags + 1)]
    X = df_with_lags[feature_cols]
    y = df_with_lags[target_col]
    return X, y


def create_lag_features(df, target_col='LOAD', num_lags=5):
    df = df.copy() # creating copy of data frame to avoid modifying orignal one
    for lag in range(1, num_lags + 1):
        df[f'{target_col}_t-{lag}'] = df[target_col].shift(lag)
    # Drop rows with NaNs resulting from lagging
    df = df.dropna().reset_index(drop=True)

    return df
if __name__ == '__main__':
    df = load_and_clean_data('../data/combinedddddd_dataset.xlsx')
    df_with_lags = create_lag_features(df, target_col='LOAD', num_lags=5)
    print(df_with_lags.head())

