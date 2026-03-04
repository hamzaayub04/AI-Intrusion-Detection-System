# AI Based Intrusion Detection System

## Overview
Machine Learning-based Intrusion Detection System trained on CICIDS dataset and deployed using Flask API with a Streamlit SOC dashboard.

## Architecture
Data → Training → Model → Flask API → SOC Dashboard → Logging

## Features
- RandomForest / XGBoost models
- Hyperparameter tuning
- Real time prediction API
- Alert logging system
- SOC visualization dashboard

## Tech Stack
- Python
- Scikit-learn
- XGBoost
- Flask
- Streamlit

## How to Run

### 1. Train Model
```bash
python main.py
```

### 2. Run API
```bash
cd api
python app.py
```

### 3. Run Dashboard
```bash
cd dashboard
streamlit run soc_dashboard.py
```


Dataset: CIC-IDS2017 (not included due to size)