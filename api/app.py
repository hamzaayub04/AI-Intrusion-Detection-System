from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

MODEL_PATH = "../models/best_ids_model.pkl"
LOG_PATH = "../logs/alerts.log"

app = Flask(__name__)

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
label_encoder = model_data["label_encoder"]

feature_names = model.feature_names_in_

os.makedirs("../logs", exist_ok=True)

def log_alert(attack_type, confidence, data):
    with open(LOG_PATH, "a") as f:
        f.write(
            f"{datetime.now()} | {attack_type} | "
            f"{confidence:.4f} | {data}\n"
        )

@app.route("/predict", methods=["POST"])
def predict():

    input_json = request.get_json()

    try:
        input_df = pd.DataFrame([input_json])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        attack_type = label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))

        if attack_type != "normal":
            log_alert(attack_type, confidence, input_json)

        return jsonify({
            "attack_type": attack_type,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)