import streamlit as st
from streamlit import runtime
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title('Customer Churn Predictor')

model = "model.pkl"
scaler = "scaler.pkl"
label_encoder = "encoder.pkl"

md = pickle.load(open(model, 'rb'))
sc = pickle.load(open(scaler,'rb' ))
lb = pickle.load(open(label_encoder,'rb' ))

gender = st.selectbox('Gender', ('Female', 'Male'))
seniorCitizen = st.selectbox('Senior Citizen', ('0', '1'))
partner = st.selectbox('Partner', ('Yes', 'No'))
dependents = st.selectbox('Dependents', ('Yes', 'No'))
phoneService = st.selectbox('Phone Service', ('Yes', 'No'))
multipleLines = st.selectbox('Multiple Lines', ('Yes', 'No', 'No phone service'))
internetService = st.selectbox('Internet Service', ('DSL', 'Fiber Optic', 'No'))
onlineSecurity = st.selectbox('Online Security', ('Yes', 'No'))
onlineBackup = st.selectbox('Online Backup', ('Yes', 'No'))
deviceProtection = st.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
techSupport = st.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
streamingMovies = st.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
contract = st.selectbox('Contract', ('Month-to-Month', 'Two year', 'One year'))
paperlessBilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
paymentMethod = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
tenure = st.number_input('Tenure')
monthlyCharges = st.number_input('Monthly charges')
totalCharges = st.number_input('Total charges')

if st.button('Predict'):
    user_input_dataframe = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [seniorCitizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'PhoneService': [phoneService],
        'MultipleLines': [multipleLines],
        'InternetService': [internetService],
        'OnlineSecurity': [onlineSecurity],
        'OnlineBackup': [onlineBackup],
        'DeviceProtection': [deviceProtection],
        'TechSupport': [techSupport],
        'StreamingMovies': [streamingMovies],
        'Contract': [contract],
        'PaperlessBilling': [paperlessBilling],
        'PaymentMethod': [paymentMethod],
        'tenure': [tenure],
        'MonthlyCharges': [monthlyCharges],
        'TotalCharges': [totalCharges]})

    label = LabelEncoder()
    cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']
    for i in cat_features:
        if i in user_input_dataframe:
            user_input_dataframe[i] = lb.fit_transform(user_input_dataframe[i])
        

    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    user_input_dataframe[num_features] = sc.fit_transform(user_input_dataframe[num_features])

    prediction = md.predict(user_input_dataframe)
    probability = prediction[0]*100
    st.write("Churn probability is: " + str(probability) + "%")
    if probability < 50:
        st.success("The customer will not churn")
    else:
        st.error("The customer may churn")
