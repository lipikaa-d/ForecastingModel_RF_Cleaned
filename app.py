from flask import Flask, render_template, request
from app.utils import (
    load_trained_model,
    forecast_next_steps,
    get_latest_input_data,
    forecast_from_manual_input
)
from src.evaluation import evaluate_model, load_model, get_evaluation_metrics
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import prepare_features_and_target
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


import os
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import io
from flask import send_file
import pandas as pd
import matplotlib.pyplot as plt
import io
from flask import send_file



app = Flask(
    __name__,
    template_folder='app/templates',
    static_folder='app/static'
)
app.secret_key = 'your-secret-key'

@app.route('/', methods=['GET', 'POST'])
def index():
    model = load_trained_model()
    latest_inputs = get_latest_input_data()

    if request.method == 'POST':
        steps = int(request.form.get('steps', 1))
        use_manual = request.form.get('input_type') == 'manual'

        if use_manual:
            try:
                manual_data = {
                    'P_IN': float(request.form.get('P_IN')),
                    'T_IN': float(request.form.get('T_IN')),
                    'P_OUT': float(request.form.get('P_OUT')),
                    'T_OUT': float(request.form.get('T_OUT')),
                    'LOAD': float(request.form.get('LOAD')),
                }

                input_data = {
                    "P_IN": manual_data['P_IN'],
                    "T_IN": manual_data['T_IN'],
                    "P_OUT": manual_data['P_OUT'],
                    "T_OUT": manual_data['T_OUT'],
                    "LOAD_t-1": manual_data['LOAD'],
                    "LOAD_t-2": manual_data['LOAD'],
                    "LOAD_t-3": manual_data['LOAD'],
                    "LOAD_t-4": manual_data['LOAD'],
                    "LOAD_t-5": manual_data['LOAD'],
                }

                forecast_values = forecast_from_manual_input(model, input_data, steps)

                return render_template(
                    'forecast_result.html',
                    forecast_values=forecast_values,
                    steps=steps,
                    latest_inputs=manual_data,
                    manual_used=True
                )

            except Exception as e:
                return f"<h3>Error in manual input: {e}</h3><br><a href='/'>Go back</a>"

        else:
            try:
                forecast_values = forecast_next_steps(model, steps)
                return render_template(
                    'forecast_result.html',
                    forecast_values=forecast_values,
                    steps=steps,
                    latest_inputs=latest_inputs,
                    manual_used=False
                )
            except Exception as e:
                return f"<h3>Error: {e}</h3><br><a href='/'>Go back</a>"

    return render_template('index.html', latest_inputs=latest_inputs)


@app.route('/metrics')
def metrics():
    try:
        from src.evaluation import get_evaluation_metrics  

        metrics = get_evaluation_metrics()

        return render_template(
            'metrics.html',
            r2=metrics['r2_score'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            rmse_percent=metrics['rmse_percent'],
            mape=metrics['mape']
        )
    except Exception as e:
        return f"<h3>Error loading evaluation metrics: {e}</h3><br><a href='/'>Go back</a>"

from flask import send_file
import matplotlib.pyplot as plt
import io

import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, send_file

@app.route('/load_plot.png')
def load_plot():
    try:
        df = load_and_clean_data('data/combinedddddd_dataset.xlsx')

        # Clean and convert date column
        df['DATE'] = df['DATE'].astype(str).str.strip()
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE', 'LOAD'])
        df = df[df['LOAD'] > 0].tail(500)

        plt.figure(figsize=(12, 4))
        plt.plot(df['DATE'], df['LOAD'], color='blue')
        plt.title('Time vs Load')
        plt.xlabel('Time')
        plt.ylabel('Load (kW)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return send_file(img, mimetype='image/png')

    except Exception as e:
        return f"<h3>Error generating plot: {e}</h3><br><a href='/'>Go back</a>"



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    message = None
    if request.method == 'POST':
        file = request.files.get('dataset')
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join('data', filename)  # Save to /data folder
            file.save(save_path)
            message = f"Dataset uploaded successfully as '{filename}'"
        else:
            message = "No file selected."

    return render_template('upload.html', message=message)

from src.model_training import train_and_save_model

@app.route('/train', methods=['GET', 'POST'])
def train():
    try:
        model, r2, rmse = train_and_save_model('data/combinedddddd_dataset.xlsx')
        return render_template('train_result.html', r2=r2, rmse=rmse)
    except Exception as e:
        return f"<h3>Error during model training: {e}</h3><br><a href='/'>Go back</a>"




if __name__ == '__main__':
    app.run(debug=True)
