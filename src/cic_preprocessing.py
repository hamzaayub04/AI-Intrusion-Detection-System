import pandas as pd
import os

SELECTED_FEATURES = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Label"
]

LABEL_MAP = {
    "BENIGN": "normal",
    "DoS Hulk": "dos",
    "DoS GoldenEye": "dos",
    "PortScan": "scan",
    "SSH-Patator": "bruteforce",
    "FTP-Patator": "bruteforce"
}

def preprocess_all(raw_dir, output_file):
    dfs = []

    for file in os.listdir(raw_dir):
        if file.endswith(".csv"):
            path = os.path.join(raw_dir, file)
            print("Processing:", file)

            df = pd.read_csv(path)

            df.columns = df.columns.str.strip()

            df = df[SELECTED_FEATURES]
            df = df.dropna()

            df["Label"] = df["Label"].map(LABEL_MAP)
            df = df.dropna()

            dfs.append(df)

    final_df = pd.concat(dfs)
    final_df.to_csv(output_file, index=False)

    print("Processed dataset saved:", output_file)

if __name__ == "__main__":
    preprocess_all(
        raw_dir="data/Raw/MachineLearningCSV",
        output_file="data/Processed/cic_ids.csv"
    )