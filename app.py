import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import shap

title = "Clinical Decision Support for Post-Stroke Cognitive Impairment Based on EEG"

model_path = "Random Forest.pkl"
model = joblib.load(model_path)
explainer = shap.TreeExplainer(model)

selected_col = ['A-MMD', 'DTABR(global)', 'D-MFO', 'DTABR(central)', 'B-MMD', 'DTABR(frontal)', 'A-MC']
selected_col1 = ['A-MMD', 'D-MFO', 'B-MMD', 'A-MC', 'DTABR(global)', 'DTABR(central)', 'DTABR(frontal)']
origin_data = [57.71, 2.59, 3.6, 2.81, 49.7, 4.27, 35.35]

st.set_page_config(page_title=f"{title}", layout="wide", page_icon="🖥️")

st.markdown(f'''
    <h1 style="text-align: center; font-size: 26px; font-weight: bold; color: white; background: #3478CE; border-radius: 0.5rem; margin-bottom: 15px;">
        {title}
    </h1>''', unsafe_allow_html=True)

data = {}
with st.form("inputform"):
    col = st.columns(4)
    for i, j in enumerate(selected_col1):
        data[j] = col[i%4].number_input(j, step=0.01, min_value=0.00, max_value=100.00, value=origin_data[selected_col.index(j)])

    c1 = st.columns(3)
    bt = c1[1].form_submit_button("**Start prediction**", use_container_width=True, type="primary")
    
if "predata" not in st.session_state:
    st.session_state.predata = data
else:
    pass


def prefun():
    pred_data = pd.DataFrame([st.session_state.predata])
    
    with st.expander("**Current input**", True):
        st.dataframe(pred_data, hide_index=True, use_container_width=True)
    
    pred_data = pred_data[selected_col]
    proba = round(float(model.predict_proba(pred_data).flatten()[1])*100, 2)
    
    with st.expander("**Predict result**", True):
        st.info(f"Predict probability segmentation threshold is **{round(float(explainer.expected_value[1])*100, 2)}%**.")
        st.markdown(f'''
             <div style="text-align: center; font-size: 26px; color: black; margin-bottom: 5px; font-family: Times New Roman; border-bottom: 1px solid black;">
             Probability of disease: {proba}%
             </div>''', unsafe_allow_html=True)
            
        shap_values = explainer.shap_values(pred_data)
        
        plt.figure()
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][0, :],
            pred_data.iloc[0, :],
            matplotlib=True
        )
        
        col = st.columns([1, 6, 1])
        col[1].pyplot(plt.gcf(), use_container_width=True)
        
if bt:
    st.session_state.predata = data
    prefun()
else:
    prefun()