import joblib

MODEL_PATH = "models/ids_model.pkl"

def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"]