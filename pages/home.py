import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from time import sleep
import os
import json

st.set_page_config(page_title='Welcome', layout='wide', initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: none
}
[data-testid="collapsedControl"] {
    display: none
}
</style>
""", unsafe_allow_html=True)
 
reduce_header_height_style = """
    <style>
        div.block-container {padding-top:4rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>ML Ops: Model Reliability Analysis at Production Level</h1>", unsafe_allow_html=True)
mark1 = st.markdown("<p style='font-size: 20px;text-align: center;'>An interactive dashboard for automating the process of monitoring the model health</p>", unsafe_allow_html=True)

st.write("---")
    
with open("pages/models.json", "r") as m:
    model_dict = json.load(m)

with open("pages/output.json", "r") as o:
    output_dict = json.load(o)
    
model_type = st.selectbox("Select Model Type:", options=list(model_dict.keys()))
model_name = st.selectbox("Select Model:", options=model_dict[model_type])
production_runs_ = os.listdir(f"pages/models/{model_name}/Production Runs")
production_run = st.selectbox("Select Production Run:", options=sorted([i[:-4] for  i in production_runs_]))
if output_dict[model_name] == "Text":
        analysis = ["Performance Drift Analysis", "Data Drift Analysis", "Data Quality Analysis", "Prediction Drift Analysis"]
elif model_type == "CV" or (model_type == "NLP" and output_dict[model_name] != "Text"):
    analysis = ["Performance Drift Analysis", "Data Drift Analysis", "Data Quality Analysis", "Prediction Drift Analysis", "LIME Analysis"]
else:
    analysis = ["Performance Drift Analysis", "Data Drift Analysis", "Data Quality Analysis", "Prediction Drift Analysis", "SHAP Analysis"]
analysis_type = st.selectbox("Select Analysis Type:",options=analysis)

st.session_state['model_type'] = model_type
st.session_state['model_name'] = model_name
st.session_state['production_run'] = production_run
st.session_state['analysis_type'] = analysis_type

first, second, _,_,_,_,_,_,_,_ = st.columns(10)
with first:
    if st.button("Run Analysis"):
        switch_page('analysis')
with second:
    if st.button("Add Model"):
        switch_page('add_model')

hide_st_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """

st.markdown(hide_st_style, unsafe_allow_html=True)