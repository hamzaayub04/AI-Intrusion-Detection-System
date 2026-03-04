import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

def tune_xgboost(X_train, y_train):

    print("⚡ Using 200k sample subset for tuning...")

    sample_size = 200000
    indices = np.random.choice(len(X_train), sample_size, replace=False)

    # Handle both pandas and numpy
    if hasattr(X_train, "iloc"):
        X_sample = X_train.iloc[indices]
    else:
        X_sample = X_train[indices]

    if hasattr(y_train, "iloc"):
        y_sample = y_train.iloc[indices]
    else:
        y_sample = y_train[indices]

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        use_label_encoder=False
    )

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1]
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=8,
        cv=3,
        scoring="f1_weighted",
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_sample, y_sample)

    print("Best Params:", search.best_params_)

    return search.best_estimator_