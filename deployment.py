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

gender = st.selectbox('Gender', ('Female', 'Male'))
seniorCitizen = st.selectbox('Gender', ('Female', 'Male'))
partner = st.selectbox('Gender', ('Female', 'Male'))
dependents = st.selectbox('Gender', ('Female', 'Male'))
phoneService = st.selectbox('Gender', ('Female', 'Male'))
multipleLines = st.selectbox('Gender', ('Female', 'Male'))
internetService = st.selectbox('Gender', ('Female', 'Male'))
onlineSecurity = st.selectbox('Gender', ('Female', 'Male'))
onlineBackup = st.selectbox('Gender', ('Female', 'Male'))
deviceProtection = st.selectbox('Gender', ('Female', 'Male'))
techSupport = st.selectbox('Gender', ('Female', 'Male'))
streamingMovies = st.selectbox('Gender', ('Female', 'Male'))
contract = st.selectbox('Gender', ('Female', 'Male'))
paperlessBilling = st.selectbox('Gender', ('Female', 'Male'))
paymentMethod = st.selectbox('Gender', ('Female', 'Male'))

tenure = st.number_input('Tenure')
monthlyCharges = st.number_input('Monthly charges')
totalCharges = st.number_input('Total charges')









prediction = md.predict([[potential,
value,
wage,
release_clause,
shot_power,
passing,
dribbling,
movement,
mentality]])


if st.button('Predict'):
    st.write("The predicted overall for your player is ", prediction[0])


