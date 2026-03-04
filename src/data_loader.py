import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    df.replace([float("inf"), -float("inf")], 0, inplace=True)

    df.fillna(0, inplace=True)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    return X, y