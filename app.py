import BackEnd.scripts.util as su
import BackEnd.scripts.load as sl
import BackEnd.scripts.run as sr
import BackEnd.scripts.plot as sp
import BackEnd.model.model as mm
import numpy as np
import sys
import os
import pandas as pd
import streamlit as st # type: ignore

# Define paths
LOGS_PATH = os.path.join(os.path.dirname(__file__), 'BackEnd/model/logs')

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ("Lof model", "VQVAE model"))

# Initialize model name based on choice
modelname = 'Lof' if model_choice == "Lof model" else 'Vqvae'

# Define tabs at the top for "Chart" and "Log"
tabs = st.tabs(["Chart", "Log"])

# Run model with Run function and prepare for plotting
try:
    with st.spinner("Running the model..."):
        result_data, score, y_pred, key_change_indices = sr.Run(modelname=modelname)
except Exception as e:
    st.error(f"Error running model: {e}")

# Plotting and displaying chart
with tabs[0]:
    st.header("Chart")
    
    # Display the plot image generated
    plot_path = os.path.join(LOGS_PATH, f'anomaly_detection_plot_{modelname}.png')
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Anomaly Detection Chart", use_column_width=True)
    else:
        st.warning("Plot image not found. Please run the model to generate the plot.")

# Display logs in a table format
with tabs[1]:
    st.header("Logs")
    
    log_path = os.path.join(LOGS_PATH, f"{modelname}.csv")
    if os.path.exists(log_path):
        logs_df = sl.load_data(log_path)
        st.dataframe(logs_df)  # Display as a table
    else:
        st.warning("No logs found. Please run the model to generate logs.")