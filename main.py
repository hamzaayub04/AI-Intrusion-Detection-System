import os
import joblib

from src.data_loader import load_data
from src.data_validation import validate_dataset
from src.model_training import split_data
from src.model_comparison import compare_models
from src.hyperparameter_tuning import tune_xgboost
from src.evaluation import evaluate

MODEL_PATH = "models/ids_model.pkl"

def main():

    print("\nIDS TRAINING PIPELINE\n")

    X, y = load_data("data/processed/cic_ids.csv")

    validate_dataset(X, y)

    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = split_data(X, y)

    comparison_df = compare_models(X_train, X_val, y_train, y_val)

    best_model_name = comparison_df.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]["Model"]

    print(f"\n Best Model: {best_model_name}")

    # Tune XGBoost 
    print("\n Running Hyperparameter Tuning...")
    best_model = tune_xgboost(X_train, y_train)

    print("\n Final Test Evaluation:")
    best_model.fit(X_train, y_train)
    evaluate(best_model, X_test, y_test)

    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": best_model,
        "label_encoder": label_encoder
    }, MODEL_PATH)

    print(f"\n Final Model Saved: {MODEL_PATH}")

if __name__ == "__main__":
    main()