import networkx as nx
import streamlit as st
from lime.lime_text import LimeTextExplainer
from lime import lime_image
from streamlit_extras.switch_page_button import switch_page
from utils import EncodeLabels, classification_metrics, plot_bar, plot_confusion_matrix, plot_roc, plot_PR, regression_metrics, plot_reg_target, data_completeness, donut_for_completeness, plot_missing_bar, data_uniqueness, donut_for_uniqueness, donut_for_dtype, plot_unique_bar, validity_check, determine_dtype_ft, ks_test, chi_test , pred_drift_plots, one_hot, syntax_drift, semantic_drift, disp_invalid_rows, clean_data, converted_data, BR_validation, donut_for_BR, get_words, nlp_quality_metrics, check_spelling, donut_for_spell_errors,CV_images_for_data_quality, seq2seq_metrics, text_score_figures, BR_for_categorical, BR_for_numerical_categorical
import pandas as pd
import numpy as np
import os
import shap
import joblib
import matplotlib.pyplot as plt
import random
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import json
 
st.set_page_config(page_title='Model Monitoring', layout='wide')
 
no_sidebar_style = """
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
"""
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(no_sidebar_style, unsafe_allow_html=True)

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)
        
try:
    st.sidebar.markdown(f"<h2>Welcome {st.session_state['username']}</h2>",unsafe_allow_html=True)
    st.sidebar.write(f"ID: {st.session_state['id']+1}")
    st.sidebar.write(f"Occupation: {st.session_state['occupation']}")
    st.sidebar.write(f"Email: {st.session_state['email']}")
    logout = st.sidebar.button('Logout')
    if logout and len(st.session_state) != 0:
        st.session_state.clear()
        switch_page("login")
    elif logout and len(st.session_state) == 0:
        st.sidebar.error("Please login first")

except:
    switch_page("login") 

model_type = model_name = production_run = analysis_type = None
vars = st.session_state
  
analysis = []

with open("pages/benchmark.json", "r") as af:
    benchmark_metrics = json.load(af)
    
with open("pages/models.json", "r") as m:
    model_dict = json.load(m)
    
with open("pages/output.json", "r") as o:
    output_dict = json.load(o)

model_type = st.sidebar.selectbox("Select Model Type:", options=list(model_dict.keys()), index=None)

if model_type != None:
    model_name = st.sidebar.selectbox("Select Model:", options=model_dict[model_type], index=None)

if model_name != None and model_name in os.listdir(f"pages/models/"):
    production_runs_ = os.listdir(f"pages/models/{model_name}/Production Runs")
    production_run = st.sidebar.selectbox("Select Production Run:", options=sorted([i[:-4] for  i in production_runs_]), index=None)
    
    if output_dict[model_name] == "Text":
        analysis = ["Performance Drift Analysis", "Data Drift Analysis", "Data Quality Analysis", "Prediction Drift Analysis"]
    elif model_type == "CV" or (model_type == "NLP" and output_dict[model_name] != "Text"):
        analysis = ["Performance Drift Analysis", "Data Drift Analysis", "Data Quality Analysis", "Prediction Drift Analysis", "LIME Analysis"]
    else:
        analysis = ["Performance Drift Analysis", "Data Drift Analysis", "Data Quality Analysis", "Prediction Drift Analysis", "SHAP Analysis"]

    analysis_type = st.sidebar.selectbox("Select Analysis Type:",options=analysis, index=None)
elif model_name!= None and model_name not in os.listdir(f"pages/models/"):
    st.write("###")
    st.write("No such model")
    exit(0)

class Helper:
    def display(model_type, model_name, production_run, analysis_type):
        st.markdown(f"<p><h2>{analysis_type} for {model_name} {model_type} model for {production_run} run</h2></p>", unsafe_allow_html=True)
        st.write("---")
        
    def get_labels(baseline_data, model_name):
        return list(baseline_data['target'].unique())

    def read_data(model_name, production_run):
    
        data_csv = f"pages/models/{model_name}/Ground Truths/{production_run}.csv"
        production_csv = f"pages/models/{model_name}/Production Runs/{production_run}.csv"
        baseline = f"pages/models/{model_name}/baseline.csv"
    
        actual_data = pd.read_csv(data_csv)
        production_data = pd.read_csv(production_csv)
        baseline_data = pd.read_csv(baseline)
        actual_data = actual_data.loc[:, ~actual_data.columns.str.startswith('Unnamed')]
        production_data = production_data.loc[:, ~production_data.columns.str.startswith('Unnamed')]
        baseline_data = baseline_data.loc[:, ~baseline_data.columns.str.startswith('Unnamed')]
        
        
        return actual_data, production_data, baseline_data

    def know_about_metrics(model_type):
        if model_type == "Regression":
            st.markdown("<p style='font-size: 15px;'>1) <b>R2 Score</b>: The R2 score (pronounced R-Squared Score) is a statistical measure that tells us how well our model is making all its predictions on a scale of 0 to 1, we can use the R2 score to determine the accuracy of our model in terms of distance or residual. It is also called as `Coefficient of Determination` </p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'><b>Formula</b>: 1 -  sum of squares of the residual errors / total sum of the errors", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'>2) <b>Mean Absolute Error</b>: The MAE is simply defined as the sum of all the distances/residuals (the difference between the actual and predicted value) divided by the total number of points in the dataset.If you want to know the model's average absolute distance when making a prediction, you can use MAE. In other words, you want to know how close the predictions are to the actual model on average.</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'><b>Formula</b>: (1/Total number of observations) * ∑|Actual value for the ith observation – Calculated value for the ith observation|", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'>3) <b>Root Mean Squared Error</b>: It is the square root of the average squared distance (difference between actual and predicted value). You use the RMSE to determine whether there are any large errors or distances that could be caused if the model overestimated the prediction (that is the model predicted values that were significantly higher than the actual value) or underestimated the predictions (that is, predicted values less than actual prediction).</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'><b>Formula</b>: √(Σ(Predicted value for data point i - Actual (observed) value for data point i)² / Number of data points)", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'>4) <b>Mean Absolute Percentage Error</b>: Mean absolute percentage error (MAPE) is a metric that defines the accuracy of a forecasting method. It represents the average of the absolute percentage errors of each entry in a dataset to calculate how accurate the forecasted quantities were in comparison with the actual quantities. MAPE is often effective for analyzing large sets of data and requires the use of dataset values other than zero.</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'><b>Formula</b>: (1/Number of data points) * Σ(|Predicted value for data point i - Actual (observed) value for data point i| / Actual (observed) value for data point i) * 100", unsafe_allow_html=True)

        elif model_type == "Classification":
            st.markdown("<p style='font-size: 15px;'>1) <b>F1</b>: It gives a combined idea about Precision and Recall metrics. It is maximum when Precision is equal to Recall. The F1 score punishes extreme values more. F1 Score could be an effective evaluation metric in the following cases:<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>i)</b> When FP and FN are equally costly<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>ii)</b> Adding more data doesn’t effectively change the outcome<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>iii)</b> True Negative is high</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'><b>Formula</b>: 2 * (Precision * Recall) / (Precision + Recall)", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'>2) <b>Precision</b>: Precision explains how many of the correctly predicted cases actually turned out to be positive. Precision is useful in the cases where False Positive is a higher concern than False Negatives. Precision is also called as `Positive Predicted Value`.</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'><b>Formula</b>: True Positives (correctly predicted positive instances) / (True Positives (correctly predicted positive instances) + False Positives (incorrectly predicted positive instances))", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'>3) <b>Recall</b>: Recall explains how many of the actual positive cases we were able to predict correctly with our model. It is a useful metric in cases where False Negative is of higher concern than False Positive. Recall is also called as `Sensitivity`.</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'><b>Formula</b>: True Positives (correctly predicted positive instances) / (True Positives (correctly predicted positive instances) + False Negatives (missed positive instances))", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'>4) <b>ROC AUC</b>: The Receiver Operator Characteristic (ROC) is a probability curve that plots the TPR(True Positive Rate) against the FPR(False Positive Rate) at various threshold values and separates the `signal` from the `noise`. The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes.</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'>5) <b>Accuracy</b>: Accuracy simply measures how often the classifier correctly predicts. We can define accuracy as the ratio of the number of correct predictions and the total number of predictions. Accuracy is useful when the target class is well balanced but is not a good choice for the unbalanced classes</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'><b>Formula</b>: (True Positives (correctly predicted positive instances) + True Negatives (correctly predicted negative instances)) / (True Positives (correctly predicted positive instances) + True Negatives (correctly predicted negative instances) + False Positives (incorrectly predicted positive instances) + False Negatives (missed positive instances))", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'>6) <b>False Positive Rate</b>: The false positive rate is calculated as FP/FP+TN, where FP is the number of false positives and TN is the number of true negatives (FP+TN being the total number of negatives). It's the probability that a false alarm will be raised: that a positive result will be given when the true value is negative.</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 15px;'><b>Formula</b>: False Positives (incorrectly predicted positive instances) / (False Positives (incorrectly predicted positive instances) + True Negatives (correctly predicted negative instances))", unsafe_allow_html=True) 
        return ''
    
    def Data_Drift_Analysis(baseline, production):

        if "probs" in production:
            production.drop("probs", axis=1, inplace=True)
        
        if model_type == "CV":
            baseline.drop('paths', axis=1, inplace=True)
            production.drop('paths', axis=1, inplace=True)

        categorical_ft, numerical_ft = determine_dtype_ft(baseline)
        total, cat, num = st.columns(3)
        total.metric(label="Total Features", value=len(baseline.columns))
        cat.metric(label="Categorical Features", value=len(categorical_ft))
        num.metric(label="Numerical Features", value=len(numerical_ft))
        st.write("---")
        
        drift_cat_val, drift_num_val, drift_df_cat, drift_df_num = 0, 0, pd.DataFrame(), pd.DataFrame()
        if len(numerical_ft) != 0:
            result_num, fig_num = ks_test(baseline, production, numerical_ft)
            drift_num_val = len([result_num[i]['status'] for i in result_num if result_num[i]['status']==True])
            drift_df_num = pd.DataFrame({'Feature Name': [i for i in result_num if result_num[i]['status'] == True],
                                    'Type of Test': 'KS Test',
                                    'Statistic': [result_num[i]['ks-stat'] for i in result_num if result_num[i]['status'] == True],
                                    'P-Value': [result_num[i]['p-value'] for i in result_num if result_num[i]['status'] == True]})
            
        if len(categorical_ft) != 0:
            result_cat, fig_cat = chi_test(baseline, production, categorical_ft)
            drift_cat_val = len([result_cat[i]['status'] for i in result_cat if result_cat[i]['status']==True])
            drift_df_cat = pd.DataFrame({'Feature Name': [i for i in result_cat if result_cat[i]['status'] == True],
                                    'Type of Test': 'Chi Square Test',
                                    'Statistic': [result_cat[i]['chi'] for i in result_cat if result_cat[i]['status'] == True],
                                    'P-Value': [result_cat[i]['p-value'] for i in result_cat if result_cat[i]['status'] == True]})
            
        drift_total, drift_cat, drift_num = st.columns(3)
        drift_total.metric(label="Drifting Features", value=drift_num_val + drift_cat_val)
        drift_cat.metric(label="Drifting Categorical Features", value=drift_cat_val)
        drift_num.metric(label="Drifting Numerical Features", value=drift_num_val)
        st.write("---")
        
        st.subheader("Information regarding Features with Drift")
        if model_type != "CV":
            st.info("🛈 Note: To check which feature importance of the drifting features please go to SHAP Analysis section.")
        

        final_df = pd.concat([drift_df_num, drift_df_cat], axis=0)
        st.dataframe(final_df, use_container_width=True)

        st.write("---")
        feature_name = st.selectbox("Select feature to analyse drift", options=baseline.columns)

        if feature_name in numerical_ft:
            st.markdown(f"<h4>KS Test with KS statistic: {result_num[feature_name]['ks-stat']} and p-value: {result_num[feature_name]['p-value']}</h4>", unsafe_allow_html=True)
            if result_num[feature_name]['status'] == True:
                st.error(f"⚠️ Drift experienced, Consider looking into data or retrain the model with new data")
            
            st.plotly_chart(fig_num[feature_name])
        
        elif feature_name in categorical_ft:
            st.markdown(f"<h4>Chi Square Test with Chi statistic: {result_cat[feature_name]['chi']} and p-value: {result_cat[feature_name]['p-value']}</h4>", unsafe_allow_html=True)
            if result_cat[feature_name]['status'] == True:
                st.error(f"⚠️ Drift experienced, Consider looking into data or retrain the model with new data")
            st.plotly_chart(fig_cat[feature_name])
        
        st.write("---")
        with st.expander("Glossary"):
            st.markdown("<p style='font-size: 17px;'><b>Data Drift Analysis</b>: Data drift analysis, with respect to production run data and baseline data, is a crucial process in machine learning model monitoring. It involves assessing and detecting changes in the distribution and characteristics of incoming data in a production environment (production run data) in comparison to the data that the model was initially trained on (baseline data). Data drift can affect the model's performance and accuracy, making it important to monitor and address.</p>",unsafe_allow_html=True)
            st.markdown("<p style='font-size: 17px;'><b>KS-Test</b>: The Kolmogorov-Smirnov (KS) test is a statistical method used to compare the similarity between two probability distributions of numerical data. It measures the maximum vertical distance between the cumulative distribution functions (CDFs) of the two datasets being compared. The KS test helps determine whether the two distributions are significantly different from each other, indicating a potential shift or change in the underlying data. The choice of the alpha level (often set to 0.05) determines the threshold for significance – if the calculated p-value is lower than alpha, we reject the null hypothesis and conclude that the two distributions are indeed different. This test is valuable for detecting data drift or changes in feature distributions.</p>",unsafe_allow_html=True)
            if model_type != "CV":
                st.markdown("<p style='font-size: 17px;'><b>Chi-Square Test</b>: The Chi-Square test is a statistical method frequently used to analyze the association between categorical variables. It can also be applied to assess data drift between two datasets with categorical data. In this context, the Chi-Square test evaluates whether the observed distribution of categories within a variable significantly differs from the expected distribution. The test involves creating a contingency table that displays the frequencies of categories for each dataset. By comparing the observed and expected frequencies using a calculated test statistic and the chosen alpha level (typically 0.05), we can determine whether the distributions have changed over time, indicating data drift. This approach helps identify shifts in categorical relationships, patterns, or proportions between datasets, aiding in the detection of potential changes or anomalies.</p>",unsafe_allow_html=True)
            if model_type == "NLP":
                st.markdown("<p style='font-size: 17px;'><b>Syntax Drift</b>: Syntax drift in the context of data drift in NLP refers to changes or discrepancies in the syntactic structure of text data between different datasets, such as a baseline dataset and a production dataset. It encompasses shifts in vocabulary usage, alterations in the distribution of linguistic features, and variations in syntactic patterns. <br> <b>Vocabulary drift</b> refers to changes in the set of words used in text data over time or between different datasets. In data drift analysis, vocabulary drift is detected by comparing the vocabulary (set of unique words) between the baseline and production datasets. New words appearing in the production dataset that were not present in the baseline indicate vocabulary drift. <br> <b>Type-Token Ratio</b> is a measure of lexical richness and diversity, calculated as the ratio of the number of unique words (types) to the total number of words (tokens) in the text. A change in TTR between the baseline and production datasets indicates a shift in vocabulary usage or lexical diversity. An increase in TTR suggests greater lexical richness or variety in the production dataset, while a decrease may indicate repetitive or limited vocabulary usage. <br> <b>Word clouds</b> visualize the frequency distribution of words in a text corpus, with more frequent words displayed in larger font sizes. Comparing word clouds between the baseline and production datasets helps identify prominent words and visualize changes in word frequency or distribution.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Semantic Drift</b>: Semantic drift refers to the phenomenon where the meaning or semantics of words, phrases, or concepts change over time or across different contexts. Words or phrases may acquire new meanings, nuances, or connotations over time, leading to semantic drift <br> In the graph plotted, we take 15 random samples from both production and baseline, using these we calculate the cosine similarity based on embeddings formed using Universal Sentence Encoder. Isolated nodes in the graph denote that they do not relate to any instance in the sampled baseline data.</p>",unsafe_allow_html=True)
            if model_type == "CV":
                st.markdown("<p style='font-size: 17px;'><b>GLCM</b>: GLCM (Gray-Level Co-occurrence Matrix) is a statistical method used in image processing to capture spatial relationships between pixels by analyzing the frequency of pixel pairs at various spatial offsets.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Contrast</b>: Measures the local variations in pixel intensity, indicating the difference between bright and dark regions in an image.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Energy</b>: Represents the uniformity of pixel intensity distribution, with higher values indicating more uniform textures.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Homogeneity</b>: Quantifies the closeness of pixel pairs to the diagonal of the GLCM, reflecting the similarity of neighboring pixel values.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Correlation</b>: Describes the linear dependency between pixel pairs in the GLCM, indicating how well pixel intensities are correlated across different directions.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Dissimilarity</b>: Measures the average difference in intensity between neighboring pixels, providing insights into the texture heterogeneity of an image.</p>",unsafe_allow_html=True)
        
            
    def Data_Quality_Analysis(baseline_data, production_data):
        if model_type == "NLP":
            quality_type = st.sidebar.selectbox("Select a Data Quality metric",options=['Completeness','Uniqueness','NLP Metrics','Profile Report'])
        elif model_type != 'CV':
            quality_type = st.sidebar.selectbox("Select a Data Quality metric",options=['Validity', 'Completeness','Uniqueness', 'Profile Report'])
        else:
            quality_type = None
        st.write('###')
        
        if model_type == "CV":
            df = pd.read_csv(f"pages/models/{model_name}/Production/{production_run}/production_paths_df.csv")
            df.drop("Unnamed: 0", axis=1, inplace=True)

            figures_CV_quality = CV_images_for_data_quality(df)
            st.markdown(f"<h3><u>Resolution and Size:</u></h3>", unsafe_allow_html=True)
            st.markdown(f"<h4>Intended Resolution: {df['resolution'].iloc[0]} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Intended Size: {df['size'].iloc[0]}</h4></h4>", unsafe_allow_html=True)
            
            size_df = pd.DataFrame({"Unique Sizes": df['size'].unique(),"Count": [df['size'].value_counts()[i] for i in df['size'].unique()]})
            resolution_df = pd.DataFrame({"Unique Resolutions": df['resolution'].unique(),"Count": [df['resolution'].value_counts()[i] for i in df['resolution'].unique()]})
            resolution, size = st.columns(2)
            with resolution:
                st.markdown(f"<h4>Resolutions found:</h4>", unsafe_allow_html=True)
                st.dataframe(resolution_df, use_container_width=True)
            with size:
                st.markdown(f"<h4>Sizes found:</h4>", unsafe_allow_html=True)
                st.dataframe(size_df, use_container_width=True)
            
            st.write("---")

            st.markdown(f"<h3><u>Sharpness:</u></h3>", unsafe_allow_html=True)
            st.write("###")
            min_sharp, avg_sharp, max_sharp = st.columns(3)

            min_sharp.metric(label="Minimum", value=round(np.min(df['sharpness']), 2))
            avg_sharp.metric(label="Average", value=round(np.mean(df['sharpness']), 2))
            max_sharp.metric(label="Maximum", value=round(np.max(df['sharpness']), 2))
            st.write("###")

            max_image_sharp, min_image_sharp = st.columns(2)
            with max_image_sharp:
                st.markdown(f"<h4>Image with Maximum Sharpness:</h4>", unsafe_allow_html=True)
                st.image(figures_CV_quality['max sharpness'], width=300)
            with min_image_sharp:
                st.markdown(f"<h4>Image with Minimum Sharpness:</h4>", unsafe_allow_html=True)
                st.image(figures_CV_quality['min sharpness'], width=300)
            
            st.write("---")

            st.markdown(f"<h3><u>Brightness:</u></h3>", unsafe_allow_html=True)
            st.write("###")
            min_bright, avg_bright, max_bright = st.columns(3)

            min_bright.metric(label="Minimum", value=round(np.min(df['brightness']), 2))
            avg_bright.metric(label="Average", value=round(np.mean(df['brightness']), 2))
            max_bright.metric(label="Maximum", value=round(np.max(df['brightness']), 2))
            st.write("###")

            max_image_bright, min_image_bright = st.columns(2)
            with max_image_bright:
                st.markdown(f"<h4>Image with Maximum Brightness:</h4>", unsafe_allow_html=True)
                st.image(figures_CV_quality['max brightness'], width=300)
            with min_image_bright:
                st.markdown(f"<h4>Image with Minimum Brightness:</h4>", unsafe_allow_html=True)
                st.image(figures_CV_quality['min brightness'], width=300)

            
            st.write("---")

            st.markdown(f"<h3><u>Noise:</u></h3>", unsafe_allow_html=True)
            st.write("###")
            min_noise, avg_noise, max_noise = st.columns(3)

            min_noise.metric(label="Minimum", value=round(np.min(df['noise']), 2))
            avg_noise.metric(label="Average", value=round(np.mean(df['noise']), 2))
            max_noise.metric(label="Maximum", value=round(np.max(df['noise']), 2))
            st.write("###")

            max_image_noise, min_image_noise = st.columns(2)
            with max_image_noise:
                st.markdown(f"<h4>Image with Maximum Noise:</h4>", unsafe_allow_html=True)
                st.image(figures_CV_quality['max noise'], width=300)
            with min_image_noise:
                st.markdown(f"<h4>Image with Minimum Noise:</h4>", unsafe_allow_html=True)
                st.image(figures_CV_quality['min noise'], width=300)
            
            st.write("---")
            st.markdown(f"<h3><u>Anomalies:</u></h3>", unsafe_allow_html=True)
            st.markdown(f"<h4>Number of Anomalies in {production_run}: <u>{np.sum(df['Anomaly'] == True)}</u></h4>", unsafe_allow_html=True)

            figure_hist = figures_CV_quality['Histogram']
            figure_hist.update_layout(width=1100,
                                height=500,
                                title_font={"size": 20},
                                yaxis_title='Count',
                                xaxis_title_font={"size":16, "color":"black"},
                                yaxis_title_font={"size":16, "color":"black"})
            figure_hist.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_hist.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_hist)

            st.write("---")
            with st.expander("Glossary"):
                st.markdown("<p style='font-size: 17px;'><b>Resolution: </b>Resolution refers to the number of pixels in an image and is typically expressed as width x height (e.g., 1920x1080). Higher resolution images have more pixels, resulting in finer details and higher image quality.",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Size: </b>Image size refers to the physical dimensions of an image in terms of file size (e.g., 2 MB). It represents the amount of disk space required to store the image. Image size is influenced by factors such as resolution, color depth, and compression. Smaller image sizes are desirable for efficient storage and faster processing.",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Sharpness: </b>Sharpness measures the clarity and crispness of details in an image. Sharp images have well-defined edges and fine details, while blurry images lack clarity. Factors affecting sharpness include focus accuracy, lens quality, and camera shake. Image processing techniques like sharpening filters can enhance sharpness in post-processing. This is necessary to build an accurate model. Sharpness was calculated using Laplacian filter.",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Brightness: </b>Brightness refers to the overall lightness or darkness of an image. It is influenced by factors such as exposure settings, lighting conditions, and dynamic range. Bright images have higher pixel values, while dark images have lower pixel values. Adjusting brightness can improve image visibility and enhance visual appeal. Brightness here is same as the mean of the image.",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Noise: </b>Noise refers to random variations in pixel values that degrade image quality. Noise can be caused by factors such as high ISO settings, low light conditions, and sensor limitations. Image denoising techniques, such as filtering and smoothing, are used to reduce noise and improve image quality. Noise here is same as standard deviation of the image.",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Anomaly: </b>Anomalies are unexpected or abnormal features in images that deviate from the norm. They can include artifacts, distortions, or objects that do not belong to the scene. Anomaly detection techniques aim to identify these irregularities and flag them for further inspection. Anomalies may indicate errors in image acquisition, processing, or analysis, or they could signify the presence of interesting or unusual phenomena. Anomalies were calculated using Pixel level deviation.",unsafe_allow_html=True)
        
        if quality_type == "Validity":
            values, score, tot_rows, v_rows, iv_rows, indices, cat_err = validity_check(baseline_data, production_data)
            st.markdown(f"<h3>Data Validity Check</h3>", unsafe_allow_html=True)
            if len(values['column']) == 0:
                st.success(f"Datatypes match perfectly", icon="👍")
            else:
                st.error(f"Contains Invalid Data. Please refer below information for further insights", icon="⚠️")
                st.error(f"Consider Checking the Production data", icon="⚠️")
            
            st.markdown(f"<b>Columns with Invalid Data</b>", unsafe_allow_html=True)
            df = pd.DataFrame(values)
            st.write(df)
            st.write("---")
                
            donut, _, a, total, valid, invalid = st.columns(6)
            with donut:
                fig_donut = donut_for_dtype(score)
                fig_donut.update_layout(title=f'Validity score: {score}%',
                                        width=400,
                                        height=400)
                st.plotly_chart(fig_donut)
            total.metric(label="Total Rows", value=tot_rows)
            valid.metric(label="Valid Rows",value=v_rows)
            invalid.metric(label='Invalid Rows', value=iv_rows)
            
            st.markdown(f"<b>Rows present with Invalid Data</b>", unsafe_allow_html=True)
            
            iv_df1 = disp_invalid_rows(production_data, indices)
            iv_df2 = disp_invalid_rows(production_data, cat_err)
            iv_df = pd.concat([iv_df1, iv_df2])
            iv_df.drop_duplicates(keep='first', inplace=True)

            st.write(iv_df.style.apply(lambda row: ['background-color: orange' if any(row.name == index and col_name == column for index, column in zip(indices['indices'], indices['column'])) or any(row.name == index and col_name == column for index, column in zip(cat_err['indices'], cat_err['column'])) else '' for col_name in row.index], axis=1))
            st.write('---')
            
            st.markdown(f"<h3>Business Rule Check</h3>", unsafe_allow_html=True)
            
            if len(indices["column"]) != 0 or len(cat_err['column']) != 0:
                st.warning("Analysis are performed on automatically cleaned data for Business Rules check.", icon="🚨")
            spare_prod = production_data.copy()
            spare_base = baseline_data.copy()
            spare_prod = converted_data(baseline_data, clean_data(production_data, indices))
            spare_base = clean_data(baseline_data, indices)
            column = st.selectbox("Select a Data Column", options=spare_prod.columns, index=None)
            args, ret = dict(), dict()
            int_cols = {col for col in spare_prod.columns if spare_prod[col].dtype in ['int64', 'float64']}
            obj_cols = {col for col in spare_prod.columns if spare_prod[col].dtype in ['object']}
            if column is not None:
                if spare_prod[column].dtype == 'int64' or spare_prod[column].dtype == 'float64':
                    rule = st.selectbox("Select an operation", options=['Greater Than', 'Less Than', 'Greater Than or Equal To', 'Less Than or Equal To', 'Equal To', 'Not Equal To', 'In range of'], index=None)
                
                    if rule is not None:
                        args['column'] = column
                        args['rule'] = rule
                        if rule == 'In range of':
                            min_value = st.number_input('Enter Minimum Value of the range: ', value=0)
                            max_value = st.number_input('Enter Maximum Value of the range: ', value=0)
                            args['min_value'] = min_value
                            args['max_value'] = max_value
                        else:
                            value = st.number_input('Enter Value to compare: ', value=0)
                            args['value'] = value
        
                        add_column = st.selectbox("Select additional column to narrow down if any:", options = int_cols.union(obj_cols)-{column}, index=None)
                        args['add_column'] = add_column
                        if add_column:
                            sub_class = st.selectbox("Select sub class to filter:", options = spare_prod[args['add_column']].unique())
                            args['sub_class'] = sub_class
                            _, _, ret1, _ = BR_validation(spare_base, spare_prod, args)
                            ret = BR_for_numerical_categorical(spare_base, spare_prod, args, ret1)
                        else:
                            _, _, ret, _ = BR_validation(spare_base, spare_prod, args)
                    
                        donut, _, a, total, satisfy, not_satisfy = st.columns(6)
                        satis_score = np.round((len(spare_prod)-(len(ret['indices']))) / len(spare_prod) * 100, 2)
                        with donut:
                            fig_donut = donut_for_BR(satis_score)
                            fig_donut.update_layout(title=f'Satisfaction score: {np.round(100-satis_score, 2)}%',
                                                    width=400,
                                                    height=400)
                            st.plotly_chart(fig_donut)
                        total.metric(label="Total Rows", value=len(spare_prod))
                        not_satisfy.metric(label="No of rows not satisfying",value=len(spare_prod)-len(ret['indices']))
                        satisfy.metric(label='No of rows satisfying', value=len(ret['indices']))  
                        st.markdown("<b>Rows satisfying the conditions</b>", unsafe_allow_html=True)
                        if len(ret) != 0:  
                            fdf = disp_invalid_rows(spare_prod, ret)
                            st.write(fdf.style.apply(lambda row: ['background-color: orange' if any(row.name == index and col_name == column for index, column in zip(ret['indices'], ret['column'])) else '' for col_name in row.index], axis=1))
                                    
                if spare_prod[column].dtype == 'object':
                    args['column'] = column
                    class_select = st.selectbox("Select class:", options=set(spare_prod[column].unique()), index=None)
                    if class_select is not None:
                        add_column = st.selectbox("Select additional column to narrow down if any:", options = int_cols.union(obj_cols)-{column}, index=None)
                        if add_column is not None and class_select is not None:
                            args['class_select'] = class_select
                            args['add_column'] = add_column
                            if args['add_column'] in int_cols:
                                rule = st.selectbox("Select an operation", options=['Greater Than', 'Less Than', 'Greater Than or Equal To', 'Less Than or Equal To', 'Equal To', 'Not Equal To', 'In range of'], index=None)
                
                                if rule is not None:
                                    args['rule'] = rule
                                    if rule == 'In range of':
                                        min_value = st.number_input('Enter Minimum Value of the range: ', value=0)
                                        max_value = st.number_input('Enter Maximum Value of the range: ', value=0)
                                        args['min_value'] = min_value
                                        args['max_value'] = max_value
                                    else:
                                        value = st.number_input('Enter Value to compare: ', value=0)
                                        args['value'] = value
                        
                                    accepted_vals, prod_vals, ret, ret1 = BR_validation(spare_base, spare_prod, args)
                                    
                            elif args['add_column'] in obj_cols:
                                sub_class = st.selectbox('Select sub class to filter:', options= spare_prod[add_column].unique())
                                args['sub_class'] = sub_class
                                ret1 = BR_for_categorical(spare_base, spare_prod, args)
                                
                            donut, _, a, total, satisfy, not_satisfy = st.columns(6)
                            try:
                                satis_score = np.round((len(spare_prod)-len(ret1['indices'])) / len(spare_prod) * 100, 2)
                                with donut:
                                    fig_donut = donut_for_BR(satis_score)
                                    fig_donut.update_layout(title=f'Satisfaction score: {np.round(100-satis_score, 2)}%',
                                                            width=400,
                                                            height=400)
                                    st.plotly_chart(fig_donut)
                                total.metric(label="Total Rows", value=len(spare_prod))
                                not_satisfy.metric(label="No of rows not satisfying",value=len(spare_prod)-len(ret1['indices']))
                                satisfy.metric(label='No of rows satisfying', value=len(ret1['indices']))
                                
                                if len(ret1) != 0:  
                                    fdf = disp_invalid_rows(spare_prod, ret1)
                                    st.write(fdf.style.apply(lambda row: ['background-color: orange' if any(row.name == index and col_name == column for index, column in zip(ret1['indices'], ret1['column'])) else '' for col_name in row.index], axis=1))
                            except:
                                pass
                            
                    else:
                        accepted_vals, prod_vals, ret, ret1 = BR_validation(spare_base, spare_prod, args)
                        st.markdown("<b>Rows with Invalid values</b>", unsafe_allow_html=True)
                        donut, _, a, total, satisfy, not_satisfy = st.columns(6)
                        satis_score = np.round((len(spare_prod)-len(ret['indices'])) / len(spare_prod) * 100, 2)
                        with donut:
                            fig_donut = donut_for_BR(satis_score)
                            fig_donut.update_layout(title=f'Invalidity score: {np.round(100-satis_score, 2)}%',
                                                    width=400,
                                                    height=400)
                            st.plotly_chart(fig_donut)
                        total.metric(label="Total Rows", value=len(spare_prod))
                        not_satisfy.metric(label="No of valid rows",value=len(spare_prod)-len(ret['indices']))
                        satisfy.metric(label='No of invalid rows', value=len(ret['indices'])) 
                        
                        if len(ret) != 0:  
                            fdf = disp_invalid_rows(spare_prod, ret)
                            st.write(fdf.style.apply(lambda row: ['background-color: orange' if any(row.name == index and col_name == column for index, column in zip(ret['indices'], ret['column'])) else '' for col_name in row.index], axis=1))
    
            st.write("---")
            with st.expander("Glossary"):
                st.markdown("<p style='font-size: 17px;'><b>Validity: </b>Validity focuses on whether data adheres to predefined rules, constraints, or acceptable value ranges. By defining and enforcing business rules, you can verify data integrity and identify discrepancies. Invalid data, such as unrealistic values or outliers, can skew analysis results, emphasizing the need for thorough validation. Validating and transforming data to meet required standards enhances its suitability for analysis and reporting. Valid data supports accurate insights, reliable reporting, and confident decision-making, fostering a data-driven culture.",unsafe_allow_html=True)
            
        elif quality_type == "Completeness":
            completeness, data_len, complete_len, missing_len, missing_data = data_completeness(production_data)
            donut, _, a, total, complete, missing = st.columns(6)
            with donut:
                fig_donut = donut_for_completeness(completeness)
                fig_donut.update_layout(title=f'Completeness Score: {completeness}%',
                                        width=400,
                                        height=400)
                st.plotly_chart(fig_donut)
            total.metric(label="Total rows", value=data_len)
            complete.metric(label="Rows with complete data",value=complete_len)
            missing.metric(label='Rows with missing data', value=missing_len)

            st.write("---")

            figure_missing_bar = plot_missing_bar(missing_data, complete_len)
            figure_missing_bar.update_layout(title='Feature Wise Missing Data',
                                            width=1100,
                                            height=500,
                                            title_font={"size": 20},
                                            xaxis_title='Features',
                                            yaxis_title='Values',
                                            xaxis_title_font={"size":16, "color":"black"},
                                            yaxis_title_font={"size":16, "color":"black"})
            figure_missing_bar.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_missing_bar.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_missing_bar)

            st.write("---")
            with st.expander("Glossary"):
                st.markdown("<p style='font-size: 17px;'><b>Completeness: </b>Ensures all required fields have non-missing values, enhancing the reliability of analysis and decision-making. Incomplete data can lead to biased conclusions and inaccurate predictions, making it essential to address and impute missing values. By calculating the percentage of missing values for each attribute, you can identify areas with high data gaps and take appropriate actions. Accurate and complete data empowers organizations to make informed decisions, improve customer experiences, and drive successful outcomes.",unsafe_allow_html=True)

        elif quality_type == "Uniqueness":
            score, unique_score, unique_value, total_len, unique_len = data_uniqueness(production_data)
            donut, _, a, total, unique, duplicate = st.columns(6)
            with donut:
                fig_donut = donut_for_uniqueness(score)
                fig_donut.update_layout(title=f'Uniqueness Score: {score}%',
                                        width=400,
                                        height=400)
                st.plotly_chart(fig_donut)
            total.metric(label="Total rows", value=total_len)
            unique.metric(label="Complete Rows",value=unique_len)
            duplicate.metric(label='Duplicate Rows', value=total_len - unique_len)

            st.write("---")

            figure_unique_score = plot_unique_bar(unique_score)
            figure_unique_score.update_layout(title='Feature Wise Uniqueness Score',
                                            width=1100,
                                            height=500,
                                            title_font={"size": 20},
                                            xaxis_title='Features',
                                            yaxis_title='Score',
                                            xaxis_title_font={"size":16, "color":"black"},
                                            yaxis_title_font={"size":16, "color":"black"})
            figure_unique_score.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_unique_score.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_unique_score)

            st.write("---")

            figure_unique_value = plot_unique_bar(unique_value)
            figure_unique_value.update_layout(title='Number of Unique Values Feature Wise',
                                            width=1100,
                                            height=500,
                                            title_font={"size": 20},
                                            xaxis_title='Features',
                                            yaxis_title='Number of Unique Values',
                                            xaxis_title_font={"size":16, "color":"black"},
                                            yaxis_title_font={"size":16, "color":"black"})
            figure_unique_value.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_unique_value.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_unique_value)

            st.write("---")
            with st.expander("Glossary"):
                st.markdown("<p style='font-size: 17px;'><b>Uniqueness: </b>Uniqueness assesses the presence of duplicate records within a dataset, highlighting potential inaccuracies or data entry errors. Duplicate data can distort analysis results, waste computational resources, and lead to redundant insights. Ensuring data uniqueness is crucial for generating reliable reports, minimizing errors, and maintaining trust in the data-driven decision-making process.",unsafe_allow_html=True)

        elif quality_type == "Profile Report":
            report = ProfileReport(production_data)
            st_profile_report(report)
            st.write("---")
            with st.expander("Glossary"):
                st.markdown("<p style='font-size: 17px;'><b>Profile Report: </b>Profile reports provide a comprehensive summary of dataset characteristics, including basic statistics, distributions, and data quality metrics. Profile reports help data analysts quickly grasp the nature of the dataset, its key attributes, and potential data quality issues. Visualizations such as histograms, scatter plots, and correlation matrices in profile reports facilitate better understanding and analysis. Profile reports highlight missing values, duplicates, unique values, and other key metrics critical for data quality assessment. Profile reports serve as a starting point for data exploration, guiding data preprocessing, validation, and feature engineering efforts.",unsafe_allow_html=True)
        
        elif quality_type == "NLP Metrics":
        
            baseline_helper = pd.read_csv(f"pages/models/{model_name}/helper/baseline.csv")
            production_helper = pd.read_csv(f"pages/models/{model_name}/helper/{production_run}.csv")
            
            st.markdown(f"<h2>Readability:</h2>", unsafe_allow_html=True)
            st.write("###")
            read_max_b, read_min_b, read_avg_b, max_text_read_b, min_text_read_b, _, _ = nlp_quality_metrics(baseline_helper, 'readability')
            b_name, min_b, avg_b, max_b = st.columns(4)
            with b_name:
                st.markdown(f"<h3>Baseline:</h3>", unsafe_allow_html=True)
            min_b.metric(label="Minimum", value=read_min_b)
            avg_b.metric(label="Average", value=read_avg_b)
            max_b.metric(label="Maximum", value=read_max_b)
            
            st.markdown(f"<h4>Text with minimum readability score</h4>", unsafe_allow_html=True)
            st.write(min_text_read_b)
            st.markdown(f"<h4>Text with maximum readability score</h4>", unsafe_allow_html=True)
            st.write(max_text_read_b)
            
            st.write("---")
            
            read_max_p, read_min_p, read_avg_p, max_text_read_p, min_text_read_p, _, _ = nlp_quality_metrics(production_helper, 'readability')
            p_name, min_p, avg_p, max_p = st.columns(4)
            with p_name:
                st.markdown(f"<h3>Production Run:</h3>", unsafe_allow_html=True)
            min_p.metric(label="Minimum", value=read_min_p)
            avg_p.metric(label="Average", value=read_avg_p)
            max_p.metric(label="Maximum", value=read_max_p)
            
            st.markdown(f"<h4>Text with minimum readability score</h4>", unsafe_allow_html=True)
            st.write(min_text_read_p)
            st.markdown(f"<h4>Text with maximum readability score</h4>", unsafe_allow_html=True)
            st.write(max_text_read_p)
            
            st.write("---")
            st.markdown(f"<h2>Spelling Errors in {production_run}</h2>", unsafe_allow_html=True)
            
            prod_words = get_words(production_data)
            number, spell_score, example = check_spelling(prod_words)
            donut_spell, _, a, total_words, error_words, crt_words = st.columns(6)
            with donut_spell:
                fig_donut_spell = donut_for_spell_errors(spell_score)
                fig_donut_spell.update_layout(title=f'Spelling Error Score: {round(spell_score,2)}%',
                                        width=400,
                                        height=400)
                st.plotly_chart(fig_donut_spell)
            total_words.metric(label="Vocabulary", value=len(set(prod_words)))
            error_words.metric(label="Misspelled words",value=number)
            crt_words.metric(label='Correct words', value=len(set(prod_words)) - number)
        
            st.markdown(f"<h4>Example words are:</h4>", unsafe_allow_html=True)
            st.text(example)
            st.write("---")
            
            st.markdown(f"<h2>Lengths</h2>", unsafe_allow_html=True)

            length_max_b, length_min_b, length_avg_b, max_text_length_b, min_text_length_b, _, _ = nlp_quality_metrics(baseline_helper, 'length')
            st.write("###")
            b_name_l, min_b_l, avg_b_l, max_b_l = st.columns(4)
            with b_name_l:
                st.markdown(f"<h3>Baseline:</h3>", unsafe_allow_html=True)
            min_b_l.metric(label="Minimum", value=length_min_b)
            avg_b_l.metric(label="Average", value=length_avg_b)
            max_b_l.metric(label="Maximum", value=length_max_b)
            
            st.markdown(f"<h4>Text with minimum length</h4>", unsafe_allow_html=True)
            st.write(min_text_length_b)
            st.markdown(f"<h4>Text with maximum length</h4>", unsafe_allow_html=True)
            st.write(max_text_length_b)
            
            st.write("---")
            
            length_max_p, length_min_p, length_avg_p, max_text_length_p, min_text_length_p, _, _ = nlp_quality_metrics(production_helper, 'length')
            p_name_l, min_p_l, avg_p_l, max_p_l = st.columns(4)
            with p_name_l:
                st.markdown(f"<h3>Production Run:</h3>", unsafe_allow_html=True)
            min_p_l.metric(label="Minimum", value=length_min_p)
            avg_p_l.metric(label="Average", value=length_avg_p)
            max_p_l.metric(label="Maximum", value=length_max_p)
            
            st.markdown(f"<h4>Text with minimum length</h4>", unsafe_allow_html=True)
            st.write(min_text_length_p)
            st.markdown(f"<h4>Text with maximum length</h4>", unsafe_allow_html=True)
            st.write(max_text_length_p)
            
            st.write("---")
            with st.expander("Glossary"):
                st.markdown("<p style='font-size: 17px;'><b>Readability</b>, in the context of text quality, refers to the ease with which a piece of text can be read and understood. <br> The Flesch–Kincaid readability tests are a set of readability formulas designed to assess the readability of English texts. They provide quantitative measures of readability based on factors such as sentence length and average syllables per word, with lower scores saying that the text is easy to read.",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Spelling errors</b> refer to mistakes or inconsistencies in the orthography of words within text data. These errors can manifest as incorrect spellings, typographical errors, or variations in spelling conventions. Detecting and correcting these spelling errors is essential for ensuring the accuracy and reliability of text processing tasks. ",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Sentence length</b> refers to the number of words or tokens contained within a sentence. Analyzing sentence length is an important aspect of text preprocessing and analysis. <br> <br> Sentence length can influence the readability and comprehension of text. Very long sentences may be difficult to parse and understand, while very short sentences may lack necessary context or detail. Analyzing and understanding the distribution of sentence lengths helps in designing models and algorithms that can handle sentences of varying complexities effectively.<br> <br> Sentence length can impact the performance of NLP models. Many NLP tasks, such as text classification or sentiment analysis, rely on fixed-length input sequences. Long sentences may exceed the model's input length limit, leading to truncation or loss of information. Conversely, short sentences may not provide sufficient context for accurate prediction. Understanding the distribution of sentence lengths in the dataset helps in selecting appropriate model architectures and hyperparameters. ",unsafe_allow_html=True)
                
            
    def Prediction_Drift_Analysis(model_name, runs, col_name, production, model_type, production_run):
        
        targets = dict()
        try:
            targets[production_run] = production[col_name]
            prev_runs = runs[:runs.index(production_run)]
            for i in prev_runs:
                _, prods,_ = Helper.read_data(i)
                targets[i] = prods[col_name]
            stats, figures = pred_drift_plots(targets, model_type)
            ma, mi = st.columns(2)
            st.subheader(f"Statistics of Target's Probability in each Production Run: ")
            
            with ma:
                st.plotly_chart(figures['Max'])
            with mi:
                st.plotly_chart(figures['Min'])
            mean, median = st.columns(2)
            with mean:
                st.plotly_chart(figures['Mean'])
            with median:
                st.plotly_chart(figures['Median'])
            st.write("---")
        except:
            pass
        
        with st.expander("Glossary"):
            st.markdown("<p style='font-size: 17px;'><b>Prediction drift analysis</b>: Prediction drift analysis is a crucial aspect of monitoring and maintaining the accuracy and reliability of machine learning models deployed in real-world applications. Prediction drift occurs when the model's predictions of new data (production data) differ significantly from its predictions of the data used during training (baseline data).",unsafe_allow_html=True)
            
            if model_type == "NLP":
                st.markdown("<p style='font-size: 17px;'><b>Semantic Drift</b>: Semantic drift refers to the phenomenon where the meaning or semantics of words, phrases, or concepts change over time or across different contexts. Words or phrases may acquire new meanings, nuances, or connotations over time, leading to semantic drift <br> In the graph plotted, we take 15 random samples from both production and baseline, using these we calculate the cosine similarity based on embeddings formed using Universal Sentence Encoder. Isolated nodes in the graph denote that they do not relate to any instance in the sampled baseline data.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>KS-Test</b>: The Kolmogorov-Smirnov (KS) test is a statistical method used to compare the similarity between two probability distributions of numerical data. It measures the maximum vertical distance between the cumulative distribution functions (CDFs) of the two datasets being compared. The KS test helps determine whether the two distributions are significantly different from each other, indicating a potential shift or change in the underlying data. The choice of the alpha level (often set to 0.05) determines the threshold for significance – if the calculated p-value is lower than alpha, we reject the null hypothesis and conclude that the two distributions are indeed different. This test is valuable for detecting data drift or changes in feature distributions.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Readability</b>, in the context of text quality, refers to the ease with which a piece of text can be read and understood. <br> The Flesch–Kincaid readability tests are a set of readability formulas designed to assess the readability of English texts. They provide quantitative measures of readability based on factors such as sentence length and average syllables per word, with lower scores saying that the text is easy to read.",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Spelling errors</b> refer to mistakes or inconsistencies in the orthography of words within text data. These errors can manifest as incorrect spellings, typographical errors, or variations in spelling conventions. Detecting and correcting these spelling errors is essential for ensuring the accuracy and reliability of text processing tasks. ",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Sentence length</b> refers to the number of words or tokens contained within a sentence. Analyzing sentence length is an important aspect of text preprocessing and analysis. <br> <br> Sentence length can influence the readability and comprehension of text. Very long sentences may be difficult to parse and understand, while very short sentences may lack necessary context or detail. Analyzing and understanding the distribution of sentence lengths helps in designing models and algorithms that can handle sentences of varying complexities effectively.<br> <br> Sentence length can impact the performance of NLP models. Many NLP tasks, such as text classification or sentiment analysis, rely on fixed-length input sequences. Long sentences may exceed the model's input length limit, leading to truncation or loss of information. Conversely, short sentences may not provide sufficient context for accurate prediction. Understanding the distribution of sentence lengths in the dataset helps in selecting appropriate model architectures and hyperparameters. ",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Similarity</b>: Cosine similarity is a metric used to measure the similarity between two vectors, often in the context of text similarity. In natural language processing (NLP), cosine similarity is commonly used to compare the similarity of documents, sentences, or word embeddings.</p>",unsafe_allow_html=True)
                
            if model_type == "Classification":
                st.markdown("<p style='font-size: 17px;'><b>Chi-Square Test</b>: The Chi-Square test is a statistical method frequently used to analyze the association between categorical variables. It can also be applied to assess data drift between two datasets with categorical data. In this context, the Chi-Square test evaluates whether the observed distribution of categories within a variable significantly differs from the expected distribution. The test involves creating a contingency table that displays the frequencies of categories for each dataset. By comparing the observed and expected frequencies using a calculated test statistic and the chosen alpha level (typically 0.05), we can determine whether the distributions have changed over time, indicating data drift. This approach helps identify shifts in categorical relationships, patterns, or proportions between datasets, aiding in the detection of potential changes or anomalies.</p>",unsafe_allow_html=True)
            
            if model_type == "Regression":
                st.markdown("<p style='font-size: 17px;'><b>KS-Test</b>: The Kolmogorov-Smirnov (KS) test is a statistical method used to compare the similarity between two probability distributions of numerical data. It measures the maximum vertical distance between the cumulative distribution functions (CDFs) of the two datasets being compared. The KS test helps determine whether the two distributions are significantly different from each other, indicating a potential shift or change in the underlying data. The choice of the alpha level (often set to 0.05) determines the threshold for significance – if the calculated p-value is lower than alpha, we reject the null hypothesis and conclude that the two distributions are indeed different. This test is valuable for detecting data drift or changes in feature distributions.</p>",unsafe_allow_html=True)

    def SHAP_Analysis(baseline, production, model_type, model_name, scaler=None):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.initjs()
        baseline = baseline.loc[:, ~baseline.columns.str.startswith('Unnamed')]
        production = production.loc[:, ~production.columns.str.startswith('Unnamed')]
        if "probs" in production.columns:
            production.drop("probs",axis=1, inplace=True)
            
        baseline_one = one_hot(baseline)
        production_one = one_hot(production)
        
        return baseline_one, production_one
    
    
class Regression:
    def Performance_Analysis(model_name, true_labels, predicted_labels):
        metrics = regression_metrics(true_labels, predicted_labels)
        st.subheader(f"Production Run's Metrics: ")
        r2, rmse, mae, mape = st.columns(4)
        r2.metric(label="R2 Score", value=metrics['R2 Score'], delta=round(metrics['R2 Score']-benchmark_metrics[model_name]['R2 Score'], 2))
        rmse.metric(label="Root Mean Squared Error", value=metrics['Root Mean Squared Error'], delta=round(benchmark_metrics[model_name]['Root Mean Squared Error'] - metrics['Root Mean Squared Error'], 2))
        mae.metric(label="Mean Absolute Error", value=metrics['Mean Absolute Error'], delta=round(benchmark_metrics[model_name]['Mean Absolute Error'] - metrics['Mean Absolute Error'], 2))
        mape.metric(label="Mean Absolute Percentage Error", value=metrics['Mean Absolute Percentage Error'], delta=round(benchmark_metrics[model_name]['Mean Absolute Percentage Error'] - metrics['Mean Absolute Percentage Error'], 2))

        st.write("---")
        st.subheader(f"Benchmark Metrics: ")
        b_r2, b_rmse, b_mae, b_mape = st.columns(4)
        b_r2.metric(label="R2 Score", value=benchmark_metrics[model_name]['R2 Score'])
        b_rmse.metric(label="Root Mean Squared Error", value=benchmark_metrics[model_name]['Root Mean Squared Error'])
        b_mae.metric(label="Mean Absolute Error", value=benchmark_metrics[model_name]['Mean Absolute Error'])
        b_mape.metric(label="Mean Absolute Percentage Error", value=benchmark_metrics[model_name]['Mean Absolute Percentage Error'])

        st.write("---")
        st.subheader(f"Ground Truth vs Predicted Targets Comparison: ")
        figure_reg = plot_reg_target(true_labels, predicted_labels)
        figure_reg.update_layout(        width=1110,
                                         height=450,
                                         xaxis_title='Number of Records in Production Data',
                                         yaxis_title='Ground Truth vs Predicted Values',
                                         xaxis_title_font={"size":16, "color":"black"},
                                        yaxis_title_font={"size":16, "color":"black"})
        figure_reg.update_xaxes(tickfont={"size":14, "color":"black"})
        figure_reg.update_yaxes(tickfont={"size":14, "color":"black"})
        st.plotly_chart(figure_reg)
        
        with st.expander("Glossary"):
            Helper.know_about_metrics('Regression')
            
    def Data_Drift_Analysis(baseline, production):
        Helper.Data_Drift_Analysis(baseline, production)
        
    def Data_Quality_Analysis(baseline_data, production_data):
        Helper.Data_Quality_Analysis(baseline_data, production_data)
        
    def Prediction_Drift_Analysis(model_name, baseline, production, model_type, production_run):
        
        runs = sorted([i[:-4] for  i in production_runs_])
        baseline_, production_ = baseline['target'], production['target']
        
        result_num, fig_num = ks_test(baseline_, production_)
        st.markdown(f"<h4>KS Test with KS statistic: {result_num['ks-stat']} and p-value: {result_num['p-value']}</h4>", unsafe_allow_html=True)
        if result_num['status'] == True:
            st.error(f"⚠️ Drift experienced, Consider looking into data or retrain the model with new data")
        
        st.plotly_chart(fig_num)
        st.write("---")
        col_name = 'target'
        
        Helper.Prediction_Drift_Analysis(model_name, runs, col_name, production, model_type, production_run)
    
    def SHAP_Analysis(baseline, production, model_type, model_name):
        baseline_one, production_one = Helper.SHAP_Analysis(baseline, production, model_type, model_name)
        
        explainer = shap.Explainer(joblib.load(f"pages/models/{model_name}/model.joblib"))
        shap_values_baseline = explainer(baseline_one)
        shap_values_production = explainer(production_one)
        row = 0
        st.markdown("<h3 style='text-align: center;'>Baseline and Production SHAP Analysis Plots</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>Waterfall Plot</h4>", unsafe_allow_html=True)
        b_waterfall, p_waterfall = st.columns(2)
        with b_waterfall:
            st.pyplot(shap.plots.waterfall(shap_values_baseline[row]))
        with p_waterfall:
            st.pyplot(shap.plots.waterfall(shap_values_production[row]))

        st.write("---")
        st.markdown(f"<h4>Bar Plot</h4>", unsafe_allow_html=True)
        b_bar, p_bar = st.columns(2)
        with b_bar:
            st.pyplot(shap.plots.bar(shap_values_baseline))
        with p_bar:
            st.pyplot(shap.plots.bar(shap_values_production))

        st.write("---")
        st.markdown(f"<h4>Violin Plot</h4>", unsafe_allow_html=True)
        b_violin, p_violin = st.columns(2)
        with b_violin:
            st.pyplot(shap.plots.violin(shap_values_baseline[:100]))
        with p_violin:
            st.pyplot(shap.plots.violin(shap_values_production[:100]))
    
        st.write("---")
        st.markdown(f"<h4>Decision Plot</h4>", unsafe_allow_html=True)
        b_decision, p_decision = st.columns(2)
        with b_decision:
            st.pyplot(shap.plots.decision(explainer.expected_value, explainer.shap_values(baseline_one)[:5], feature_names=list(baseline_one.columns)))
        with p_decision:
            st.pyplot(shap.plots.decision(explainer.expected_value, explainer.shap_values(production_one)[:5], feature_names=list(baseline_one.columns)))
        
        st.write("---")
        st.markdown(f"<h4>Force Plot for Baseline Run</h4>", unsafe_allow_html=True)
        st.pyplot(shap.plots.force(shap_values_baseline[row], matplotlib=True, figsize=(30, 5)))
        st.write("---")
        st.markdown(f"<h4>Force Plot for Production Run</h4>", unsafe_allow_html=True)
        st.pyplot(shap.plots.force(shap_values_production[row], matplotlib=True, figsize=(30, 5)))
        st.write("---")
        with st.expander("Glossary"):
            st.markdown("<p style='font-size: 17px;'><b>SHAP (SHapley Additive exPlanations)</b> is used for explaining the output of machine learning models. It originates from cooperative game theory and assigns each feature an importance value for a particular prediction.<br> SHAP aims to make complex machine learning models interpretable by providing explanations for individual predictions. This is crucial for understanding how models make decisions, especially in high-stakes or regulated domains. <br>It is based on the concept of Shapley values from cooperative game theory. Shapley values determine the contribution of each feature to the difference between the actual prediction and the average prediction. They guarantee several desirable properties, such as fairness, consistency, and accuracy.<br> These SHAP values provide feature importance scores, allowing users to identify which features have the most significant impact on model predictions. This helps in feature selection, model debugging, and understanding the underlying relationships in the data.<br> Such explanations or interpretability can be visualized through various plots, such as summary plots, force plots, dependence plots, and waterfall plots. These visualizations enhance understanding and facilitate communication of model explanations.",unsafe_allow_html=True) 

        
class Classification:
    def Performance_Analysis(model_name, true_labels, predicted_labels):
        if any(isinstance(label, str) for label in true_labels) and any(isinstance(label, str) for label in predicted_labels):
            true_labels, predicted_labels = EncodeLabels(true_labels, predicted_labels)

        metrics = classification_metrics(true_labels, predicted_labels)
        st.subheader(f"Production Run's Metrics: ")
        f1, precision, recall, roc, accuracy, fpr = st.columns(6)
        f1.metric(label="F1", value=round(metrics["F1"],2), delta= str(round(metrics["F1"] - benchmark_metrics[model_name]["F1"],2)))
        precision.metric(label="Precision", value=round(metrics["Precision"],2), delta= str(round(metrics["Precision"] - benchmark_metrics[model_name]["Precision"],2)))
        recall.metric(label="Recall", value=round(metrics["Recall"],2), delta= str(round(metrics["Recall"] - benchmark_metrics[model_name]["Recall"],2)))
        roc.metric(label="ROC_AUC", value=round(metrics["ROC_AUC"],2), delta= str(round(metrics["ROC_AUC"] - benchmark_metrics[model_name]["ROC_AUC"],2)))
        accuracy.metric(label="Accuracy", value=round(metrics["Accuracy"],2), delta= str(round(metrics["Accuracy"] - benchmark_metrics[model_name]["Accuracy"],2)))
        fpr.metric(label="False Positive Rate", value=round(metrics["False Positive Rate"],2), delta= str(round(benchmark_metrics[model_name]["False Positive Rate"] - metrics["False Positive Rate"], 2)))
        st.write("---")
        
        st.subheader("Benchmark Metrics: ")
        f1_bench, precision_bench, recall_bench, roc_bench, accuracy_bench, fpr_bench = st.columns(6)
        f1_bench.metric(label="F1", value=round(benchmark_metrics[model_name]["F1"],2))
        precision_bench.metric(label="Precision", value=round(benchmark_metrics[model_name]["Precision"],2))
        recall_bench.metric(label="Recall", value=round(benchmark_metrics[model_name]["Recall"],2))
        roc_bench.metric(label="ROC_AUC", value=round(benchmark_metrics[model_name]["ROC_AUC"],2))
        accuracy_bench.metric(label="Accuracy", value=round(benchmark_metrics[model_name]["Accuracy"],2))
        # tpr_bench.metric(label="True Positive Rate", value=round(benchmark_metrics[model]["True Positive Rate"],2))
        fpr_bench.metric(label="False Positive Rate", value=round(benchmark_metrics[model_name]["False Positive Rate"],2))
        st.write("---")
       
        prod_roc, bench_roc = st.columns(2)
        with prod_roc:
            st.subheader("AUC-ROC Curve")
            figure_roc, roc_prod = plot_roc(true_labels,predicted_labels)
            figure_roc.update_layout(
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=400,
                        height=400)
            figure_roc.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_roc.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_roc)
 
        with bench_roc:
            st.subheader("Confusion Matrix")
            figure_conf = plot_confusion_matrix(true_labels,predicted_labels, Helper.get_labels(actual_data, model_name))
            figure_conf.update_layout(
                        #title=f'AUC: {round(roc_bench, 2)}',
                        xaxis_title='True Class',
                        yaxis_title='Predicted Class',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=400,
                        height=400)
            figure_conf.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_conf.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_conf)
        
        pr_curve, bench_roc = st.columns(2)
        with pr_curve:
            st.subheader("Precision-Recall Curve")
            figure_pr, precision, recall = plot_PR(true_labels,predicted_labels)
            figure_pr.update_layout(
                        title=f"Precision: {precision}, Recall: {recall}",
                        xaxis_title='Recall',
                        yaxis_title='Precision',
                        title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=400,
                        height=400)
            figure_pr.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_pr.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_pr)
 
        with bench_roc:
            st.subheader("Metrics' Bar Chart")
            figure_conf = plot_bar(metrics, benchmark_metrics[model_name])
            figure_conf.update_layout(
                        #title=f'AUC: {round(roc_bench, 2)}',
                        xaxis_title='Metric',
                        yaxis_title='Value',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=600,
                        height=400)
            figure_conf.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_conf.update_yaxes(tickfont={"size":14, "color":"black"})
            figure_conf.update_layout(barmode='group')
            st.plotly_chart(figure_conf)
        st.write("---")
        
        with st.expander("Glossary"):
            Helper.know_about_metrics('Classification')
    
    def Data_Drift_Analysis(baseline, production):
        Helper.Data_Drift_Analysis(baseline, production)
        
    def Data_Quality_Analysis(baseline_data, production_data):
        Helper.Data_Quality_Analysis(baseline_data, production_data)
        
    def Prediction_Drift_Analysis(model_name, baseline, production, model_type, production_run):
        
        runs = sorted([i[:-4] for  i in production_runs_])
        baseline_, production_ = baseline['target'], production['target']
        
        result_cat, fig_cat = chi_test(baseline_, production_)
        st.markdown(f"<h4>Chi Square Test with Chi statistic: {result_cat['chi']} and p-value: {result_cat['p-value']}</h4>", unsafe_allow_html=True)
        if result_cat['status'] == True:
            st.error(f"⚠️ Drift experienced, Consider looking into data or retrain the model with new data")
        st.plotly_chart(fig_cat)
        st.write("---")

        col_name = 'probs'
        
        Helper.Prediction_Drift_Analysis(model_name, runs, col_name, production, model_type, production_run)
    
    def SHAP_Analysis(baseline, production, model_type, model_name):
        baseline_one, production_one = Helper.SHAP_Analysis(baseline, production, model_type, model_name)
        
        model = joblib.load(f"pages/models/{model_name}/model.joblib")
        
        clas = random.randint(0,1)
        def f(X):
            return model.predict_proba(X)[:, clas]
    
        explainer = shap.Explainer(f, baseline_one)
        shap_values_baseline = explainer(baseline_one[:100])
        shap_values_production = explainer(production_one[:100])
        
        row = 0
        
        st.markdown("<h3 style='text-align: center;'>Baseline and Production SHAP Analysis Plots</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>Waterfall Plot</h4>", unsafe_allow_html=True)
        b_waterfall, p_waterfall = st.columns(2)
        with b_waterfall:
            st.pyplot(shap.plots.waterfall(shap_values_baseline[row]))
        with p_waterfall:
            st.pyplot(shap.plots.waterfall(shap_values_production[row]))

        st.write("---")
        st.markdown(f"<h4>Bar Plot</h4>", unsafe_allow_html=True)
        b_bar, p_bar = st.columns(2)
        with b_bar:
            st.pyplot(shap.plots.bar(shap_values_baseline))
        with p_bar:
            st.pyplot(shap.plots.bar(shap_values_production))

        st.write("---")
        st.markdown(f"<h4>Violin Plot</h4>", unsafe_allow_html=True)
        b_violin, p_violin = st.columns(2)
        with b_violin:
            st.pyplot(shap.plots.violin(shap_values_baseline[:100]))
        with p_violin:
            st.pyplot(shap.plots.violin(shap_values_production[:100]))
    
        st.write("---")
        st.markdown(f"<h4>Scatter Plot</h4>", unsafe_allow_html=True)
        col = st.selectbox("Select column to understand dependence", options=list(baseline_one.columns))
        b_decision, p_decision = st.columns(2)
        
        with b_decision:
            st.pyplot(shap.plots.scatter(shap_values_baseline[:, col], dot_size=20, color=shap_values_baseline))
        with p_decision:
            st.pyplot(shap.plots.scatter(shap_values_production[:, col], dot_size=20, color=shap_values_production))
        
        st.write("---")
        st.markdown(f"<h4>Force Plot for Baseline Run</h4>", unsafe_allow_html=True)
        st.pyplot(shap.plots.force(shap_values_baseline[row], matplotlib=True, figsize=(30, 5)))
        st.write("---")
        st.markdown(f"<h4>Force Plot for Production Run</h4>", unsafe_allow_html=True)
        st.pyplot(shap.plots.force(shap_values_production[row], matplotlib=True, figsize=(30, 5)))
        st.write("---")
        with st.expander("Glossary"):
            st.markdown("<p style='font-size: 17px;'><b>SHAP (SHapley Additive exPlanations)</b> is used for explaining the output of machine learning models. It originates from cooperative game theory and assigns each feature an importance value for a particular prediction.<br> SHAP aims to make complex machine learning models interpretable by providing explanations for individual predictions. This is crucial for understanding how models make decisions, especially in high-stakes or regulated domains. <br>It is based on the concept of Shapley values from cooperative game theory. Shapley values determine the contribution of each feature to the difference between the actual prediction and the average prediction. They guarantee several desirable properties, such as fairness, consistency, and accuracy.<br> These SHAP values provide feature importance scores, allowing users to identify which features have the most significant impact on model predictions. This helps in feature selection, model debugging, and understanding the underlying relationships in the data.<br> Such explanations or interpretability can be visualized through various plots, such as summary plots, force plots, dependence plots, and waterfall plots. These visualizations enhance understanding and facilitate communication of model explanations.",unsafe_allow_html=True) 

        
    
class NLP:
    def Performance_Analysis(model_name, true_labels, predicted_labels):
        outputs = output_dict[model_name]
        if outputs == 'Classification':
            Classification.Performance_Analysis(model_name, true_labels, predicted_labels)
        elif outputs == 'Text':
            metrics = seq2seq_metrics(true_labels, predicted_labels)

            gt_helper = pd.read_csv(f'pages/models/{model_name}/helper_target/Ground Truths/{production_run}.csv')
            prod_helper = pd.read_csv(f'pages/models/{model_name}/helper_target/Production Runs/{production_run}.csv')
            
            figures = text_score_figures(prod_helper)
            st.subheader(f"Production Run's Metrics: ")
            f1, precision, recall, bleu_s, simi = st.columns(5)
            f1.metric(label="F1", value=round(metrics["F1"],2), delta= str(round(metrics["F1"] - benchmark_metrics[model_name]["F1"],2)))
            precision.metric(label="Precision", value=round(metrics["Precision"],2), delta= str(round(metrics["Precision"] - benchmark_metrics[model_name]["Precision"],2)))
            recall.metric(label="Recall", value=round(metrics["Recall"],2), delta= str(round(metrics["Recall"] - benchmark_metrics[model_name]["Recall"],2)))
            bleu_s.metric(label="BLEU", value=round(metrics["BLEU"],2), delta= str(round(metrics["BLEU"] - benchmark_metrics[model_name]["BLEU"],2)))
            simi.metric(label="Similarity", value=round(np.mean(prod_helper['similarity_with_gt']),2), delta= str(round(np.mean(prod_helper['similarity_with_gt']) - benchmark_metrics[model_name]["Similarity"],2)))
            
            st.write("---")
            st.subheader("Benchmark Metrics: ")
            
            f1_bench, precision_bench, recall_bench, bleu_bench, simi_bench = st.columns(5)
            f1_bench.metric(label="F1", value=round(benchmark_metrics[model_name]["F1"],2))
            precision_bench.metric(label="Precision", value=round(benchmark_metrics[model_name]["Precision"],2))
            recall_bench.metric(label="Recall", value=round(benchmark_metrics[model_name]["Recall"],2))
            bleu_bench.metric(label="BLEU", value=round(benchmark_metrics[model_name]["BLEU"],2))
            simi_bench.metric(label="Similarity", value=round(benchmark_metrics[model_name]["Similarity"],2))
            
            st.write("---")
            st.subheader("ROUGE")
            figure_rouge_hist = figures['ROUGE Hist']
            figure_rouge_hist.update_layout(width=1000,
                                height=500,
                                title="Distribution of Metrics",
                                title_font={"size": 20},
                                yaxis_title='Count',
                                xaxis_title_font={"size":16, "color":"black"},
                                yaxis_title_font={"size":16, "color":"black"})
            figure_rouge_hist.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_rouge_hist.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_rouge_hist)
            
            figure_rouge_box = figures['ROUGE Box']
            figure_rouge_box.update_layout(width=1000,
                                height=500,
                                title='Box Plot',
                                title_font={"size": 20},
                                xaxis_title="Score",
                                yaxis_title='Metric',
                                xaxis_title_font={"size":16, "color":"black"},
                                yaxis_title_font={"size":16, "color":"black"})
            figure_rouge_box.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_rouge_box.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_rouge_box)
            
            st.write("---")
            st.subheader("BLEU")
            figure_BLEU_hist = figures['BLEU Hist']
            figure_BLEU_hist.update_layout(width=1000,
                                height=500,
                                title="Distribution of Metrics",
                                title_font={"size": 20},
                                yaxis_title='Count',
                                xaxis_title_font={"size":16, "color":"black"},
                                yaxis_title_font={"size":16, "color":"black"})
            figure_BLEU_hist.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_BLEU_hist.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_BLEU_hist)
            
            figure_BLEU_box = figures['BLEU Box']
            figure_BLEU_box.update_layout(width=1000,
                                height=500,
                                title='Box Plot',
                                title_font={"size": 20},
                                xaxis_title="Score",
                                yaxis_title='Metric',
                                xaxis_title_font={"size":16, "color":"black"},
                                yaxis_title_font={"size":16, "color":"black"})
            figure_BLEU_box.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_BLEU_box.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_BLEU_box)
            
            st.write("---")
            st.subheader("Similarity")
            figure_sim = plot_reg_target(gt_helper['similarity'], prod_helper['similarity_wth_text'])
            figure_sim.update_layout( title="Similarity of predicted and ground truth with text",
                                    width=1000,
                                    height=450,
                                    xaxis_title='Similarity',
                                    yaxis_title='Ground Truth vs Predicted Similarity',
                                    title_font={"size": 20},
                                    xaxis_title_font={"size":16, "color":"black"},
                                    yaxis_title_font={"size":16, "color":"black"})
            figure_sim.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_sim.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_sim)
            
            figure_Similarity_hist = figures['Similarity Hist']
            figure_Similarity_hist.update_layout(width=1000,
                                height=500,
                                title="Distribution of Metrics",
                                title_font={"size": 20},
                                yaxis_title='Count',
                                xaxis_title_font={"size":16, "color":"black"},
                                yaxis_title_font={"size":16, "color":"black"})
            figure_Similarity_hist.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_Similarity_hist.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_Similarity_hist)
            
            figure_Similarity_box = figures['Similarity Box']
            figure_Similarity_box.update_layout(width=1000,
                                height=500,
                                title='Box Plot',
                                title_font={"size": 20},
                                xaxis_title="Score",
                                yaxis_title='Metric',
                                xaxis_title_font={"size":16, "color":"black"},
                                yaxis_title_font={"size":16, "color":"black"})
            figure_Similarity_box.update_xaxes(tickfont={"size":14, "color":"black"})
            figure_Similarity_box.update_yaxes(tickfont={"size":14, "color":"black"})
            st.plotly_chart(figure_Similarity_box)
            
            
            result_num, fig_num = ks_test(prod_helper['similarity_with_gt'], prod_helper['similarity_wth_text'], b_name="Ground Truth")
            st.markdown(f"<h4>KS Test with KS statistic: {result_num['ks-stat']} and p-value: {result_num['p-value']}</h4>", unsafe_allow_html=True)
            if result_num['status'] == True:
                st.error(f"⚠️ Drift experienced, Consider looking into data or retrain the model with new data")
            st.plotly_chart(fig_num)
            
            st.write("---")
            with st.expander("Glossary"):
                st.markdown("<p style='font-size: 17px;'><b>F1</b>: F1-score is the harmonic mean of precision and recall and provides a balanced measure of both metrics. It takes into account both false positives (incorrectly included information) and false negatives (important information that is missed). F1-score is often used as a single summary measure of the overall quality of the generated summary.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Precision</b>: Precision measures the ability of the summarization system to avoid including irrelevant or redundant information in the generated summary. A high precision indicates that the system generates summaries with a high proportion of relevant information and few irrelevant or redundant details. However, high precision alone does not necessarily mean that the summary is comprehensive or covers all the important information from the reference summary.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Recall</b>: Recall measures the ability of the summarization system to capture all the important information from the reference summary.A high recall indicates that the system is good at including important information from the reference summary in the generated summary. However, high recall alone does not necessarily mean that the summary is of high quality, as it may include irrelevant or redundant information.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>BLEU</b>: BLEU is designed to measure the similarity between a machine-generated translation and one or more human-generated reference translations. It provides an automatic evaluation method for machine translation systems, allowing researchers to compare the performance of different systems objectively.</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size: 17px;'><b>Similarity</b>: Cosine similarity is a metric used to measure the similarity between two vectors, often in the context of text similarity. In natural language processing (NLP), cosine similarity is commonly used to compare the similarity of documents, sentences, or word embeddings.</p>",unsafe_allow_html=True)
            
    def Data_Drift_Analysis(baseline, production):
        st.markdown(f"<h3>Syntax Drift</h3>", unsafe_allow_html=True)
        
        words_b = get_words(baseline)
        words_p = get_words(production)
        
        syntax_figures, base_freq, prod_freq, uncommon, uncommon_freq, top5_uncommon, top5_b, top5_p = syntax_drift(words_b, words_p)
        score = round((len(uncommon) / len(set(words_b))) * 100, 2)
        
        st.markdown(f"<h4>Vocabulary Drift: {score}%</h4>", unsafe_allow_html=True)
        if score > 30:
            st.error(f"{score}% of the vocabulary doesn't appear in baseline data", icon="⚠️")
        else:
            st.success(f"Not much drift in vocabulary", icon="👍")
        
        st.write("---")
        ttr = round(len(set(words_p)) / len(words_p), 2) * 100
        st.markdown(f"<h4>Type-Token Ratio: {ttr}%</h4>", unsafe_allow_html=True)
        if (ttr >= 40) and (ttr<=60):
            st.error(f"Lexical Variation Present", icon="⚠️")
        else:
            st.success(f"Not much Lexical Diversity", icon="👍")
        
        st.write("---")
        st.markdown(f"<h4>Top 5 Uncommon Words that appeared in Production</h4>", unsafe_allow_html=True)
        uncommon_df = pd.DataFrame({"Word": list(top5_uncommon.keys()),
                                    "Frequency": list(top5_uncommon.values())})
        st.dataframe(uncommon_df, use_container_width=True)
        
        st.markdown(f"<h4>Word Clouds of Baseline and Production</h4>", unsafe_allow_html=True)
        cloud_b, cloud_p = st.columns(2)
        with cloud_b:
            plt.imshow(syntax_figures["baseline cloud"])
            plt.axis("off")
            st.pyplot()
        with cloud_p:
            plt.imshow(syntax_figures["production cloud"])
            plt.axis("off")
            st.pyplot()
        
        st.markdown(f"<h4>Frequencies of Top 5 Baseline words in Production</h4>", unsafe_allow_html=True) 
        st.plotly_chart(syntax_figures["top 5 baseline bar"])
        st.markdown(f"<h4>Frequencies of Top 5 Production words in Baseline</h4>", unsafe_allow_html=True)
        st.plotly_chart(syntax_figures["top 5 production bar"])

        sorted_words_baseline = dict(sorted(base_freq.items(), key=lambda a:a[1], reverse=True))
        sorted_words_production = dict(sorted(prod_freq.items(), key=lambda a:a[1], reverse=True))
        freq_result, freq_fig = ks_test(list(sorted_words_baseline.values())[:1000], list(sorted_words_production.values())[:1000], nlp=True)
        st.markdown(f"<h4>KS Test with KS statistic: {freq_result['ks-stat']} and p-value: {freq_result['p-value']}</h4>", unsafe_allow_html=True)
        if freq_result['status'] == True:
            st.error(f"⚠️ Drift experienced, Consider looking into data or retrain the model with new data")
        st.plotly_chart(freq_fig)

        st.write("---")
        st.markdown(f"<h3>Semantic Drift</h3>", unsafe_allow_html=True)
        semantic_figure, isolated_score, isolated, edges, edge_alpha, pr_node_sizes, bs_node_sizes = semantic_drift(production_run, model_name)
        st.plotly_chart(semantic_figure)
        
        st.write("---")
        G = nx.Graph()
        for i in list(bs_node_sizes.keys()):
            G.add_node(i, size=(bs_node_sizes[i]+1)*25, label="Baseline", node_type="Baseline", alpha=(bs_node_sizes[i])/max(bs_node_sizes.values()), color="red")

        for i in list(pr_node_sizes.keys()):
            G.add_node(i, size=(pr_node_sizes[i]+1)*25, label="Production", node_type="Production", alpha=(pr_node_sizes[i])/max(pr_node_sizes.values()), color="blue")

        G.add_edges_from(edges)

        node_sizes = [G.nodes[node]['size'] for node in G.nodes]
        node_colors = [G.nodes[node]['color'] for node in G.nodes]
        node_alphas = [G.nodes[node]['alpha'] for node in G.nodes]

        st.markdown(f"<h3>{round(isolated_score,2)}% nodes are isolated, Number of Isolated nodes are: {len(isolated)}</h3>", unsafe_allow_html=True)
        plt.figure(figsize=(10,7))
        plt.legend(["Red: Baseline", "Blue: Production"])
        nx.draw_networkx_nodes(G, nx.shell_layout(G), node_size=node_sizes, node_color=node_colors, alpha=node_alphas)
        nx.draw_networkx_edges(G, nx.shell_layout(G), edgelist=edges)
        nx.draw_networkx_labels(G, nx.shell_layout(G), labels={n:n for n in G})
        
        st.pyplot()
        
        st.write("---")
        st.markdown(f"<h3>Other Features</h3>", unsafe_allow_html=True)
        
        
        if "cleaned_text" in baseline.columns:
            baseline.drop("cleaned_text", axis=1, inplace=True)
        baseline.drop("text", axis=1, inplace=True)
    
        Helper.Data_Drift_Analysis(baseline, production)
    
    def Data_Quality_Analysis(baseline_data, production_data):
        Helper.Data_Quality_Analysis(baseline_data, production_data)
        
    def Prediction_Drift_Analysis(model_name, baseline, production, model_type, production_run):
        outputs = output_dict[model_name]
        if outputs == 'Classification':
            Classification.Prediction_Drift_Analysis(model_name, baseline, production, model_type, production_run)
        elif outputs == 'Text':
            gt_helper = pd.read_csv(f'pages/models/{model_name}/helper_target/Ground Truths/{production_run}.csv')
            prod_helper = pd.read_csv(f'pages/models/{model_name}/helper_target/Production Runs/{production_run}.csv')
            
            st.markdown(f"<h3><u>Readability:</u></h3>", unsafe_allow_html=True)
            result_read, fig_read = ks_test(gt_helper['readability'], prod_helper['readability'], b_name="Ground Truth")
            st.markdown(f"<h4>KS Test with KS statistic: {result_read['ks-stat']} and p-value: {result_read['p-value']}</h4>", unsafe_allow_html=True)
            if result_read['status'] == True:
                st.error(f"⚠️ Drift experienced, Consider looking into data or retrain the model with new data")
            st.plotly_chart(fig_read)
            
            read_max, read_min, read_avg, max_text_read, min_text_read, max_read_id, min_read_id = nlp_quality_metrics(prod_helper, 'readability', col='target')
            p_name, min_read, avg_read, max_read = st.columns(4)
            with p_name:
                st.markdown(f"<h3>{production_run}:</h3>", unsafe_allow_html=True)
            min_read.metric(label="Minimum", value=read_min)
            avg_read.metric(label="Average", value=read_avg)
            max_read.metric(label="Maximum", value=read_max)
            
            st.markdown(f"<h4>Text with minimum readability score in production run</h4>", unsafe_allow_html=True)
            min_p_read, min_gt_read = st.columns(2)
            with min_p_read:
                st.markdown(f"<h4>Production Run</h4>", unsafe_allow_html=True)
                st.write(min_text_read)
            with min_gt_read:
                st.markdown(f"<h4>Ground Truth</h4>", unsafe_allow_html=True)
                
                text = gt_helper['target'].iloc[min_read_id]
                st.write(text)
            
            st.markdown(f"<h4>Text with maximum readability score in production run</h4>", unsafe_allow_html=True)
            max_p_read, max_gt_read = st.columns(2)
            with max_p_read:
                st.markdown(f"<h4>Production Run</h4>", unsafe_allow_html=True)
                st.write(max_text_read)
            with max_gt_read:
                st.markdown(f"<h4>Ground Truth</h4>", unsafe_allow_html=True)
                
                text = gt_helper['target'].iloc[max_read_id]
                st.write(text)
            
            st.write("---")
            
            st.markdown(f"<h3><u>Length:</u></h3>", unsafe_allow_html=True)
            result_length, fig_length = ks_test(gt_helper['length'], prod_helper['length'], b_name="Ground Truth")
            st.markdown(f"<h4>KS Test with KS statistic: {result_length['ks-stat']} and p-value: {result_length['p-value']}</h4>", unsafe_allow_html=True)
            if result_length['status'] == True:
                st.error(f"⚠️ Drift experienced, Consider looking into data or retrain the model with new data")
            st.plotly_chart(fig_length)
            
            length_max, length_min, length_avg, max_text_length, min_text_length, max_length_id, min_length_id = nlp_quality_metrics(prod_helper, 'readability', col='target')
            p_name_l, min_length, avg_length, max_length = st.columns(4)
            with p_name_l:
                st.markdown(f"<h3>{production_run}:</h3>", unsafe_allow_html=True)
            min_length.metric(label="Minimum", value=length_min)
            avg_length.metric(label="Average", value=length_avg)
            max_length.metric(label="Maximum", value=length_max)
            
            st.markdown(f"<h4>Text with minimum length in production run</h4>", unsafe_allow_html=True)
            min_p_length, min_gt_length = st.columns(2)
            with min_p_length:
                st.markdown(f"<h4>Production Run</h4>", unsafe_allow_html=True)
                st.write(min_text_length)
            with min_gt_length:
                st.markdown(f"<h4>Ground Truth</h4>", unsafe_allow_html=True)
                text = gt_helper['target'].iloc[min_length_id]
                st.write(text)
            
            st.markdown(f"<h4>Text with maximum length in production run</h4>", unsafe_allow_html=True)
            
            max_p_length, max_gt_length = st.columns(2)
            
            with max_p_length:
                st.markdown(f"<h4>Production Run</h4>", unsafe_allow_html=True)
                st.write(max_text_length)
            
            with max_gt_length:
                st.markdown(f"<h4>Ground Truth</h4>", unsafe_allow_html=True)
                text = gt_helper['target'].iloc[max_length_id]
                st.write(text)
            
            st.write("---")
            
            st.markdown(f"<h3><u>Spelling Errors in {production_run}'s Predictions</u></h3>", unsafe_allow_html=True)
                
            prod_words = get_words(prod_helper, col='target')
            number, spell_score, example = check_spelling(prod_words)
            donut_spell, _, a, total_words, error_words, crt_words = st.columns(6)
            with donut_spell:
                fig_donut_spell = donut_for_spell_errors(spell_score)
                fig_donut_spell.update_layout(title=f'Spelling Error Score: {round(spell_score,2)}%',
                                        width=400,
                                        height=400)
                st.plotly_chart(fig_donut_spell)
            total_words.metric(label="Vocabulary", value=len(set(prod_words)))
            error_words.metric(label="Misspelled words",value=number)
            crt_words.metric(label='Correct words', value=len(set(prod_words)) - number)
            
            st.markdown(f"<h4>Example words are:</h4>", unsafe_allow_html=True)
            st.text(example)
            
            st.write("---")
            st.markdown(f"<h3><u>Semantic Drift in {production_run}'s Predictions</u></h3>", unsafe_allow_html=True)
            semantic_figure  = semantic_drift(production_run, model_name, col='target', b_name="Ground Truth")
            st.plotly_chart(semantic_figure)
            
            result_pred, fig_pred = ks_test(gt_helper['similarity'], prod_helper['similarity_wth_text'], b_name="Ground Truth")
            st.markdown(f"<h4>KS Test with KS statistic: {result_pred['ks-stat']} and p-value: {result_pred['p-value']}</h4>", unsafe_allow_html=True)
            if result_pred['status'] == True:
                st.error(f"⚠️ Drift experienced, Consider looking into data or retrain the model with new data")
            st.plotly_chart(fig_pred)
            
            st.write("---")
            with st.expander("Glossary"):
                    st.markdown("<p style='font-size: 17px;'><b>Prediction drift analysis</b>: Prediction drift analysis is a crucial aspect of monitoring and maintaining the accuracy and reliability of machine learning models deployed in real-world applications. Prediction drift occurs when the model's predictions of new data (production data) differ significantly from its predictions of the data used during training (baseline data).",unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 17px;'><b>Semantic Drift</b>: Semantic drift refers to the phenomenon where the meaning or semantics of words, phrases, or concepts change over time or across different contexts. Words or phrases may acquire new meanings, nuances, or connotations over time, leading to semantic drift <br> In the graph plotted, we take 15 random samples from both production and baseline, using these we calculate the cosine similarity based on embeddings formed using Universal Sentence Encoder. Isolated nodes in the graph denote that they do not relate to any instance in the sampled baseline data.</p>",unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 17px;'><b>KS-Test</b>: The Kolmogorov-Smirnov (KS) test is a statistical method used to compare the similarity between two probability distributions of numerical data. It measures the maximum vertical distance between the cumulative distribution functions (CDFs) of the two datasets being compared. The KS test helps determine whether the two distributions are significantly different from each other, indicating a potential shift or change in the underlying data. The choice of the alpha level (often set to 0.05) determines the threshold for significance – if the calculated p-value is lower than alpha, we reject the null hypothesis and conclude that the two distributions are indeed different. This test is valuable for detecting data drift or changes in feature distributions.</p>",unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 17px;'><b>Readability</b>, in the context of text quality, refers to the ease with which a piece of text can be read and understood. <br> The Flesch–Kincaid readability tests are a set of readability formulas designed to assess the readability of English texts. They provide quantitative measures of readability based on factors such as sentence length and average syllables per word, with lower scores saying that the text is easy to read.",unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 17px;'><b>Spelling errors</b> refer to mistakes or inconsistencies in the orthography of words within text data. These errors can manifest as incorrect spellings, typographical errors, or variations in spelling conventions. Detecting and correcting these spelling errors is essential for ensuring the accuracy and reliability of text processing tasks. ",unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 17px;'><b>Sentence length</b> refers to the number of words or tokens contained within a sentence. Analyzing sentence length is an important aspect of text preprocessing and analysis. <br> <br> Sentence length can influence the readability and comprehension of text. Very long sentences may be difficult to parse and understand, while very short sentences may lack necessary context or detail. Analyzing and understanding the distribution of sentence lengths helps in designing models and algorithms that can handle sentences of varying complexities effectively.<br> <br> Sentence length can impact the performance of NLP models. Many NLP tasks, such as text classification or sentiment analysis, rely on fixed-length input sequences. Long sentences may exceed the model's input length limit, leading to truncation or loss of information. Conversely, short sentences may not provide sufficient context for accurate prediction. Understanding the distribution of sentence lengths in the dataset helps in selecting appropriate model architectures and hyperparameters. ",unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 17px;'><b>Similarity</b>: Cosine similarity is a metric used to measure the similarity between two vectors, often in the context of text similarity. In natural language processing (NLP), cosine similarity is commonly used to compare the similarity of documents, sentences, or word embeddings.</p>",unsafe_allow_html=True)
            
    
    def LIME_Analysis(model_name, production_run):
        path = f"pages/models/{model_name}"
        vect = joblib.load(f"{path}/vectorizer.joblib")
        model = joblib.load(f"{path}/model.joblib")
        df = pd.read_csv(f"{path}/Production Runs/{production_run}.csv")

        def pipeline_nlp(text):
            text = vect.transform(text)
            return model.predict_proba(text)  
        
        
        class_names = df['target'].unique()
        explainer = LimeTextExplainer(class_names=class_names)

        for i in class_names:
            st.subheader(f"Class name: {i.title()}")
            explain = explainer.explain_instance(df[df['target'] == i]['text'].sample(1).iloc[0], pipeline_nlp, num_features=10)
            st.components.v1.html(explain.as_html(), height=400)
            st.write("---")
        
        with st.expander("Glossary"):
                st.markdown("<p style='font-size: 17px;'><b>LIME for Text: </b>LIME (Local Interpretable Model-agnostic Explanations) is a model interpretation technique designed to provide insights into the predictions of complex machine learning models, particularly in the context of natural language processing (NLP). LIME starts by generating perturbations or variations of the input text data. <br><br> These perturbations may involve adding or removing words, changing word order, or replacing words with synonyms. Each perturbation results in a modified version of the original text. <br><br>The perturbed text data are then passed through the black-box model to obtain predictions for each modified version of the input. This step effectively creates a dataset of perturbed instances and their corresponding model predictions.LIME fits an interpretable local linear model, such as Ridge regression or Lasso regression, to the perturbed data. This model aims to approximate the behavior of the black-box model in the vicinity of the input instance.<br><br>The coefficients of the local linear model represent the importance or contribution of each feature (e.g., words) to the model's prediction for the input instance. These coefficients serve as explanations for why the model made a particular prediction. Finally, LIME provides explanations for the model's predictions by highlighting the most influential features identified by the local linear model. These explanations help users understand which aspects of the input text data influenced the model's decision-making process.",unsafe_allow_html=True)



class CV():
    def Performance_Analysis(model_name, true_labels, predicted_labels):
        Classification.Performance_Analysis(model_name, true_labels, predicted_labels)
    
    def Data_Drift_Analysis(baseline, production):
        Helper.Data_Drift_Analysis(baseline, production)
    
    def Data_Quality_Analysis(baseline_data, production_data):
        Helper.Data_Quality_Analysis(baseline_data, production_data)
    
    def Prediction_Drift_Analysis(model_name, baseline, production, model_type, production_run):
        Classification.Prediction_Drift_Analysis(model_name, baseline, production, model_type, production_run)
    
    def LIME_Analysis(model_name, production_run):
        path = f"pages/models/{model_name}/Production"
        class_names = os.listdir(f"{path}/{production_run}")[:-1]
        random_images = dict()
        for i in class_names:
            random_images[i] = f"{path}/{production_run}/{i}/" + random.sample(os.listdir(f"{path}/{production_run}/{i}"), 1)[0]
        
        model = load_model(f"pages/models/{model_name}/model.h5")
        for i in class_names:
    
            to_display = cv.imread(random_images[i])
            image = tf.io.read_file(random_images[i])
            image = tf.image.decode_image(image,channels=3)
            image = tf.image.resize(image,size=(224, 224))
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(image.numpy().astype('double'), model, hide_color=0, num_samples=500)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)

            if i == 'fractured': st.subheader("no fracture".title())
            else: st.subheader("fractured".title())
            display_image, temp_= st.columns(2)
            with display_image:
                st.markdown(f"<h4>Actual Image:</h4>", unsafe_allow_html=True)
                st.image(to_display, width=450)
            with temp_:
                st.markdown(f"<h4>Image locations that influenced the prediction</h4>", unsafe_allow_html=True)
                plt.imshow(temp)
                plt.axis('off')
                st.pyplot()
            
            st.write("---")
        with st.expander("Glossary"):
            st.markdown("<p style='font-size: 17px;'><b>LIME for Images: </b>LIME (Local Interpretable Model-agnostic Explanations) can also be applied to interpret the predictions of image classification models. It provides insights into why a model makes a certain prediction by generating local explanations for individual images. LIME generates perturbed versions of the input image by applying small, localized changes. These perturbations may involve masking parts of the image, adding noise, or altering pixel values. <br><br> Each perturbed image is passed through the black-box image classification model to obtain predicted probabilities for each class. LIME extracts interpretable features from the perturbed images, such as superpixels or regions of interest. These features represent the localized areas of the image that contribute most to the model's prediction. <br><br> LIME fits a simple, interpretable model (e.g., linear regression or decision trees) to the extracted features and corresponding predicted probabilities. This local model approximates the behavior of the black-box model in the vicinity of the input image. The coefficients or importance scores of the local interpretable model indicate the contribution of each feature to the model's prediction. By visualizing these contributions, LIME provides explanations for why the model made a particular prediction.",unsafe_allow_html=True)
        

if len(st.session_state) > 0:   
    if st.session_state['model_type'] != None and st.session_state['model_name'] != None and st.session_state['production_run'] != None and st.session_state['analysis_type'] != None:
        model_type = vars['model_type']
        model_name = vars['model_name']
        production_run = vars['production_run']
        analysis_type = vars['analysis_type']    
        st.session_state['model_type']=st.session_state['model_name']=st.session_state['production_run']=st.session_state['analysis_type'] = None

if model_type and model_name and production_run and analysis_type is not None:
    Helper.display(model_type, model_name, production_run, analysis_type)
    actual_data, production_data, baseline = Helper.read_data(model_name, production_run)
    spare_production = production_data.copy()
    spare_actual = actual_data.copy()
    
    if model_type == 'Regression' or model_type == 'Classification':
        _, _, _, _, _, indices, cat_err = validity_check(baseline.drop('target',axis=1), production_data[actual_data.drop('target',axis=1).columns])
    
        if len(indices['column']) != 0 or len(cat_err['column']) != 0:
            if analysis_type != 'Data Quality Analysis':
                st.warning(f"Invalid Data found for the production run {production_run}. Please go to Data Quality Check for more detailed information.", icon="🚨")
                st.warning("Analysis are performed on automatically cleaned data.", icon="🚨")

            spare_production = converted_data(baseline, clean_data(production_data, indices))
            spare_actual = clean_data(spare_actual, indices)
            _, _, _, _, _, _, cat_err1 = validity_check(baseline.drop('target',axis=1), spare_production[actual_data.drop('target',axis=1).columns])
            spare_production = converted_data(baseline, clean_data(spare_production, cat_err1))
            spare_actual = clean_data(spare_actual, cat_err1)
    
    if analysis_type == "Performance Drift Analysis":
        eval(model_type).Performance_Analysis(model_name, spare_actual['target'], spare_production['target'])
    elif  analysis_type == "Data Drift Analysis":
        eval(model_type).Data_Drift_Analysis(baseline.drop('target',axis=1), spare_production.drop('target',axis=1))
    elif analysis_type == "Data Quality Analysis":
        eval(model_type).Data_Quality_Analysis(baseline.drop('target',axis=1), production_data[actual_data.drop('target',axis=1).columns]) 
    elif analysis_type == "Prediction Drift Analysis":
        eval(model_type).Prediction_Drift_Analysis(model_name, spare_actual, spare_production, model_type, production_run) 
    elif analysis_type == "SHAP Analysis":
        eval(model_type).SHAP_Analysis(baseline.drop('target',axis=1), spare_production.drop('target',axis=1), model_type, model_name)
    elif analysis_type == "LIME Analysis":
        eval(model_type).LIME_Analysis(model_name, production_run)    

if st.sidebar.button("Add Model"):
    switch_page("add_model") 


hide_st_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """

st.markdown(hide_st_style, unsafe_allow_html=True)