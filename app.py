from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": "http://localhost:5173"}
})
# Load preprocessing pipeline
pipeline = joblib.load('pipeline.pkl')

# Load trained model
model = joblib.load('model.pkl')


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        # data = {
        #     "Age": 42,
        #     "Sex": "M",
        #     "ChestPainType": "ASY",
        #     "RestingBP": 160.0,
        #     "Cholesterol": 193.0,
        #     "FastingBS": 0,
        #     "RestingECG": "Normal",
        #     "MaxHR": 102,
        #     "ExerciseAngina": "Y",
        #     "Oldpeak": 3.0,
        #     "ST_Slope": "Flat"
        # }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Apply preprocessing
        transformed_data = pipeline.transform(input_df)

        # Make prediction
        prediction = model.predict(transformed_data)
        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
