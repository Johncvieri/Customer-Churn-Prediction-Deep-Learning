import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title = 'Customer Churn Prediction Application',
                  layout = "wide",
                  initial_sidebar_state = "expanded",
                  menu_items = {
                      'About' : 'Customer Churn Predicton '
                  })


# load model
class columnDropperTransformer():
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self

pickles = open('preprocessing.pkl', 'rb')
preprocessing = pickle.load(pickles)
saved_model=load_model('Model_seq.h5')

def predict(inputs):
    df = pd.DataFrame(inputs, index=[0])
    df = preprocessing.transform(df)
    y_pred = saved_model.predict(df)
    y_pred = np.where(y_pred < 0.5, 0, 1).squeeze()
    print(y_pred)
    return y_pred.item()

columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
label = ['0', '1']

st.title("Customer Churn Prediction")


customerID = 'afsafas'
gender = 'afawfaw'
SeniorCitizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
Partner = st.selectbox("Marriage Status", ['Married', 'Not Married'])
Dependents = 'awfawnfinaw'
tenure = st.slider("Tenure Length", min_value=0.0, max_value=72.0, value=24.0, step=1.0, help='Tenure Length Default 24 Months')
PhoneService = 'eniwfniwfwef'
MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No'])
InternetService = st.selectbox("Which internet service do you use?", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Do you need online security?", ['No', 'Yes', 'No internet service'])
OnlineBackup = st.selectbox("Do you need online backup?", ['No', 'Yes', 'No internet service'])
DeviceProtection = st.selectbox("Do you need device protection?", ['No', 'Yes', 'No internet service'])
TechSupport = st.selectbox("Do you need Tech Support?", ['No', 'Yes', 'No internet service'])
StreamingTV = 'dssdbjfhbjsd'
StreamingMovies = 'hdsjbjbsfy'
Contract = st.selectbox("Which contract will you use?", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = ' adbaibdq'
PaymentMethod = 'nadibfqf'
MonthlyCharges = st.number_input("Monthly Charges", min_value=19.0, max_value=119.0, value=75.0, step=0.1, help='Customers Monthly Charges Default is $75')
TotalCharges = st.number_input("Total Charges", min_value=19.0, max_value=8685.0, value=500.0, step=0.1, help='Customers Total Charges Default is $500')

#inference
new_data = [customerID, gender, SeniorCitizen, Partner, Dependents, tenure, 
PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
DeviceProtection, TechSupport, StreamingTV, StreamingMovies, 
Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
new_data = pd.DataFrame([new_data], columns = columns)
new_data = preprocessing.transform(new_data).tolist()
res = saved_model.predict(new_data)

res = 0 if res < 0.5 else 1

press = st.button('Predict')
if press:
   st.title(label[res])
   '' '''Description :
-  0: Not Churn
-  1: Churn'''