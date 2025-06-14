import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import shap

df1 = pd.read_csv("train.csv")
X_train = df1.iloc[:, :-1]
y_train = df1.iloc[:, -1]

df2 = pd.read_csv("test.csv")
X_test = df2.iloc[:, :-1]
y_test = df2.iloc[:, -1]

title = "Clinical Decision Support for Post-Stroke Cognitive Impairment Based on EEG"

model_path = "XGBoost.pkl"
model = joblib.load(model_path)
explainer = shap.TreeExplainer(model)

def plot():
    col = st.columns(3)
    
    expected_value = explainer.expected_value
    features = X_train
    features.columns=X.columns
    shap_values = explainer.shap_values(features)
    
    plt.figure()
    shap.decision_plot(expected_value, shap_values, features, show=False)
    col[0].pyplot(plt.gcf(), use_container_width=True)

    plt.figure()
    shap_values = explainer.shap_values(features)
    shap.summary_plot(shap_values, features, show=False)
    col[1].pyplot(plt.gcf(), use_container_width=True)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist(), plot_type='bar', show=False)
    col[2].pyplot(plt.gcf(), use_container_width=True)


selected_col = ['B-MMD', 'D-MFO', 'DTABR(global)', 'DTABR(frontal)', 'A-MMD']
selected_col1 = ['A-MMD', 'D-MFO', 'B-MMD', 'DTABR(global)', 'DTABR(frontal)']
origin_data = [50.23, 3.93, 1.45, 2.02, 47.64]

st.set_page_config(page_title=f"{title}", layout="wide", page_icon="🖥️")

st.markdown(f'''
    <h1 style="text-align: center; font-size: 26px; font-weight: bold; color: white; background: #3478CE; border-radius: 0.5rem; margin-bottom: 15px;">
        {title}
    </h1>''', unsafe_allow_html=True)

data = {}
with st.form("inputform"):
    col = st.columns(3)
    for i, j in enumerate(selected_col1):
        data[j] = col[i%3].number_input(j, step=0.01, min_value=0.00, max_value=100.00, value=origin_data[selected_col.index(j)])

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
    res = model.predict(pred_data, output_margin=True)
    proba = round(float(model.predict_proba(pred_data)[0][1])*100, 2)
    
    with st.expander("**Predict result**", True):
        st.info("Predict output_margin value is: {res},  $Probability=\\frac {1} {1 + e^{(-output\_margin)}}$, output_margin segmentation threshold is **-0.0429**, probability segmentation threshold is **48.93%**.".replace("{res}", str(round(float(res), 4))))
        st.markdown(f'''
             <div style="text-align: center; font-size: 26px; color: black; margin-bottom: 5px; font-family: Times New Roman; border-bottom: 1px solid black;">
             Probability of disease: {proba}%
             </div>''', unsafe_allow_html=True)
            
        shap_values = explainer.shap_values(pred_data)
        
        plt.figure()
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X_test.iloc[0],
            matplotlib=True
        )
        
        col = st.columns([1, 6, 1])
        col[1].pyplot(plt.gcf(), use_container_width=True)
        
        #plot()
        
if bt:
    st.session_state.predata = data
    prefun()
else:
    prefun()