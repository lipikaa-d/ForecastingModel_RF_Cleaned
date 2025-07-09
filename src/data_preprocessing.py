import pandas as pd
import os

def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_excel(filepath, engine='openpyxl')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    df = df.drop_duplicates(subset='DATE', keep='first')
    df = df.ffill()  # ✅ Forward fill only (do NOT dropna)
    return df

if __name__ == '__main__':
    df = load_and_clean_data('../data/combinedddddd_dataset.xlsx')
    print(df.head())


