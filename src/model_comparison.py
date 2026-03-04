import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def run_cross_validation(model, X_train, y_train):

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    print("\nCross Validation F1 Scores:", scores)
    print("Mean CV Score:", scores.mean())


def compare_models(X_train, X_val, y_train, y_val):

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        
        "LogisticRegression": LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    solver="lbfgs"
        ),

        "XGBoost": XGBClassifier(
    n_estimators=200,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42
    )
    }

    results = []

    for name, model in models.items():

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="weighted")
        recall = recall_score(y_val, preds, average="weighted")

        results.append([name, acc, f1, recall])

        print(f"\n{name} Completed.")

    df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "F1 Score", "Recall"]
    )

    print("\n=== Model Comparison ===")
    print(df)

    return df