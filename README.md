# ML Ops: Model Reliability Analysis at Production Level 

Hello Tech peeps!
Now that the world turned into an environment for ML and DL models, has anyone thought about the performance monitoring of the models after the production stages? Well, we did, and have a blast using our tool! 

This is ML Ops, a model reliability analysis tool at post production stages. Our tool provides 5 types of analyses, namely ***Data Drift Analysis, Prediction Drift Analysis, Performance Drift Analysis, Data Quality Analysis, and SHAP/LIME Analysis for Regression, Classification, NLP, and CV Models***. 

Using this application, one can monitor their ML and DL models in post production stages on how they are working, what can be improved, is the model and the data consistent or not, etc.
For people who do not know what these analyses mean, here's a short explanation for it:
1. Data Drift Analysis -- Data drift analysis involves monitoring and detecting changes in data distributions over time to ensure the continued accuracy and reliability of machine learning models. It helps identify when a model's input data has shifted, potentially degrading its performance.

2. Prediction Drift Analysis -- Prediction drift analysis involves monitoring changes in the output predictions of a machine learning model over time. It helps identify when a model's predictions deviate significantly from expected results, indicating potential issues with the model's performance or data inputs.

3. Performance Drift Analysis -- Performance drift analysis involves monitoring the decline in a machine learning model's performance over time. It helps identify when a model's accuracy, precision, recall, or other performance metrics degrade, indicating a need for model retraining or adjustment.

4. Data Quality Analysis -- Data quality analysis involves evaluating the accuracy, completeness, consistency, and reliability of data used in a machine learning model. It ensures that the data is of high quality, which is crucial for generating accurate and reliable model predictions.

5. SHAP/LIME Analysis -- SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) are techniques used to explain the predictions of machine learning models. SHAP provides consistent and globally interpretable feature importance values based on cooperative game theory, while LIME explains individual predictions by approximating the model locally with an interpretable model.

How to run it? Well, that's pretty easy.

  **PREREQUISITES**
1. Clone or download the repository
2. ENSURE PYTHON IS INSTALLED ON YOUR SYSTEM! If not the app will not open no matter how many times you click on it.
3. Double click on the install_requirements file to install all the dependencies. An alternate would be to open the command prompt in the directory where you have this repository, and execute 'pip install -r requirements.txt', and your dependencies are installed!
   
  **USING THE APPLICATION**
1.  To use the application, simply double click on the 'app' file to open the website. The initial load time is around 1 to 2 minutes so please excuse the app for the delay.
2. To use the app, enter username as "Exampler" and Password as "123456"
3. Select any of the pre-defined models that we provided analysis for, or analyse your own model by clicking on ADD MODEL

  **ADDING A CUSTOM MODEL**
1. Click on ADD MODEL button to add your own custom model.
2. Upon landing at the ADD MODEL page, chose the type of the model (regression, classification, NLP, CV) you want to add.
3. Enter the model name, baseline data (training data), target label, scaler if any used, model file, benchmark metrics and click on submit. This will add your model to the database.
4. In the same page, after clicking on submit, upload the production data, ground truths for the production data, date of production run, and click on upload run.
5. Go back to the login page and viola! You have your model ready for analysis.   

And there's that. This is our takeModel Reliability Analysis in post production stages. For people who are interested in this domain, feel free to use our work for reference or even develop on it. Thank you.

**ALSO**, this works on local PC and not on website level. We made it a local application to preserve the security and safety of the training data, which can get exposed on the internet.
Thanks for reading!