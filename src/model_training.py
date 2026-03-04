import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def split_data(X, y):

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=0.30,
        stratify=y_encoded,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, le

MODEL_PATH = "models/ids_model.pkl"


def train_model(X, y):

    print("\n Training Production IDS Model...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    train_preds = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)

    print(f"\nTraining Accuracy: {train_acc:.4f}")

    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": pipeline,
        "label_encoder": le
    }, MODEL_PATH)

    print(f"\n Model saved to {MODEL_PATH}")

    return pipeline, X_test, y_test