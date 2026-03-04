import pandas as pd

def validate_dataset(X, y):
    print("\n DATA VALIDATION REPORT")

    print(f"Total samples: {len(X)}")
    print(f"Total features: {X.shape[1]}")

    if X.isnull().sum().sum() > 0:
        print("Missing values detected:")
        print(X.isnull().sum())
    else:
        print("No missing values")

    class_counts = y.value_counts()
    print("\nClass Distribution:")
    print(class_counts)

    if class_counts.min() < 2:
        raise ValueError(
            "Dataset unsafe for stratified split: "
            "each class must have at least 2 samples"
        )
    elif class_counts.min() < 5:
        print("Warning: Some classes have very few samples (risk of overfitting)")

    print("Data validation completed\n")