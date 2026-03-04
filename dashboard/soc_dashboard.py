import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

st.set_page_config(
    page_title="IDS SOC Dashboard",
    layout="wide"
)

st.title("IDS SOC Dashboard")
st.subheader("Live Threat Detection System")

API_URL = "http://127.0.0.1:5000/predict"

st.sidebar.header("Flow Feature Input")

flow_duration = st.sidebar.number_input("Flow Duration", value=100)
total_fwd_packets = st.sidebar.number_input("Total Fwd Packets", value=10)
total_backward_packets = st.sidebar.number_input("Total Backward Packets", value=5)
flow_bytes = st.sidebar.number_input("Total Length of Fwd Packets", value=500)

if st.sidebar.button("Analyze Traffic"):

    payload = {
        "Flow Duration": flow_duration,
        "Total Fwd Packets": total_fwd_packets,
        "Total Backward Packets": total_backward_packets,
        "Total Length of Fwd Packets": flow_bytes
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "Unknown")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.success(f"Prediction: {prediction}")

            # Save alert to session state
            if "alerts" not in st.session_state:
                st.session_state.alerts = []

            st.session_state.alerts.append({
                "Time": timestamp,
                "Prediction": prediction
            })

        else:
            st.error("API Error")

    except Exception as e:
        st.error(f"Connection Error: {e}")

st.subheader("Live Alerts Log")

if "alerts" in st.session_state and len(st.session_state.alerts) > 0:
    df = pd.DataFrame(st.session_state.alerts[::-1])
    st.dataframe(df, use_container_width=True)
else:
    st.info("No alerts generated yet.")

time.sleep(1)