import streamlit as st
from streamlit import runtime
import pandas as pd
import numpy as np
import pickle

st.title('Customer Churn Predictor')

model = "model.pkl"
scaler = "scaler.pkl"
label_encoder = "label.pkl"

md = pickle.load(open(model, 'rb'))
sc = pickle.load(open(scaler,'rb' ))
lb = pickle.load(open(label_encoder,'rb' ))

user_inputs = {}

cat_features = ['gender', 'seniorCitizen', 'partner', 'dependents', 'phoneService',
            'multipleLines', 'internetService', 'onlineSecurity', 'onlineBackup',
            'deviceProtection', 'techSupport', 'streamingMovies', 'contract',
            'paperlessBilling', 'paymentMethod']
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

user_inputs[gender] = st.selectbox('Gender', ('Female', 'Male'))
user_inputs[seniorCitizen] = st.selectbox('Senior Citizen', ('0', '1'))
user_inputs[partner] = st.selectbox('Gender', ('Yes', 'No'))
user_inputs[dependents] = st.selectbox('Gender', ('Yes', 'No'))
user_inputs[phoneService] = st.selectbox('Gender', ('Yes', 'No'))
user_inputs[multipleLines] = st.selectbox('Gender', ('Yes', 'No', 'No phone service'))
user_inputs[internetService] = st.selectbox('Gender', ('DSL', 'Fiber Optic', 'No'))
user_inputs[onlineSecurity] = st.selectbox('Gender', ('Yes', 'No'))
user_inputs[onlineBackup] = st.selectbox('Gender', ('Yes', 'No'))
user_inputs[deviceProtection] = st.selectbox('Gender', ('Yes', 'No', 'No internet service'))
user_inputs[techSupport] = st.selectbox('Gender', ('Yes', 'No', 'No internet service'))
user_inputs[streamingMovies] = st.selectbox('Gender', ('Yes', 'No', 'No internet service'))
user_inputs[contract] = st.selectbox('Gender', ('Month-to-Month', 'Two year', 'One year'))
user_inputs[paperlessBilling] = st.selectbox('Gender', ('Yes', 'No'))
user_inputs[paymentMethod] = st.selectbox('Gender', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
user_inputs[tenure] = st.number_input('Tenure')
user_inputs[monthlyCharges] = st.number_input('Monthly charges')
user_inputs[totalCharges] = st.number_input('Total charges')

user_input_dataframe = pd.DataFrame([user_inputs])

for i in cat_features:
    user_input_dataframe[i] = lb[i].transform(user_input_dataframe[i])

for j in num_features:
    user_input_dataframe[j] = sc[j].transform(np.array(user_input_dataframe[[j]]))

prediction = md.predict(user_input_dataframe)

if st.button('Predict'):
    st.write("Churn rate is: ", prediction[0])


