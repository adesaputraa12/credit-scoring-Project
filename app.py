from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Completed"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Ambil semua input dari form
    fixed_acidity        = float(request.form.get('fixed_acidity'))
    volatile_acidity     = float(request.form.get('volatile_acidity'))
    citric_acid          = float(request.form.get('citric_acid'))
    residual_sugar       = float(request.form.get('residual_sugar'))
    chlorides            = float(request.form.get('chlorides'))
    free_sulfur_dioxide  = float(request.form.get('free_sulfur_dioxide'))
    total_sulfur_dioxide = float(request.form.get('total_sulfur_dioxide'))
    density              = float(request.form.get('density'))
    pH                   = float(request.form.get('pH'))
    sulphates            = float(request.form.get('sulphates'))
    alcohol              = float(request.form.get('alcohol'))

    # Susun jadi array
    data = pd.DataFrame({"fixed acidity": [fixed_acidity],
                         "volatile acidity": [volatile_acidity],
                         "citric acid": [citric_acid],
                         "residual sugar": [residual_sugar],
                         "chlorides": [chlorides],
                         "free sulfur dioxide": [free_sulfur_dioxide],
                         "total sulfur dioxide": [total_sulfur_dioxide],
                         "density": [density],
                         "pH": [pH],
                         "sulphates": [sulphates],
                         "alcohol": [alcohol]
                        })

    # Prediksi
    prediction = PredictionPipeline().predict(data)
    result = round(float(prediction[0]), 2)

    # ← Kirim 'prediction' ke template
    return render_template('index.html', prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080)