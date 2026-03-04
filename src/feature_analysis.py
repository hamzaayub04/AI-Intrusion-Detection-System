import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, X):

    importances = model.feature_importances_
    features = X.columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    top10 = importance_df.head(10)

    print("\nTop 10 Important Features:\n")
    print(top10)

    plt.figure()
    plt.barh(top10["Feature"], top10["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importance")
    plt.show()