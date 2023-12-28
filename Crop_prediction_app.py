import streamlit as st
import pickle
import pandas as pd
import numpy as np


# Load your trained Naive Bayes model using pickle
with open('NaiveBayes.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('CROP RECOMMENDATION SYSTEM')    

# Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # Assuming your model takes an array of features
    features = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Streamlit App


# Input form for user to enter values
st.sidebar.header("Input Parameters")
nitrogen = st.sidebar.slider("Nitrogen", 0.0, 100.0, 50.0)
phosphorus = st.sidebar.slider("Phosphorus", 0.0, 100.0, 50.0)
potassium = st.sidebar.slider("Potassium", 0.0, 100.0, 50.0)
temperature = st.sidebar.slider("Temperature", 0.0, 100.0, 25.0)
humidity = st.sidebar.slider("Humidity", 0.0, 100.0, 50.0)
ph = st.sidebar.slider("ph", 0.0, 10.0, 5.0)
rainfall = st.sidebar.slider("Rainfall", 0.0, 500.0, 100.0)

# Predict button
if st.sidebar.button("Predict"):
    result = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
    st.success(f"The predicted crop is: {result}")