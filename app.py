# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open('beer_servings_model.pkl', 'rb'))

# Streamlit app title and description
st.title("Beer Servings Prediction App")
st.write("Predict the number of beer servings based on user inputs.")

# Input fields for user input (features required by the model)
spirit_servings = st.number_input("Spirit Servings", min_value=0, max_value=500, value=50)
wine_servings = st.number_input("Wine Servings", min_value=0, max_value=500, value=50)
total_litres_of_pure_alcohol = st.number_input("Total Litres of Pure Alcohol", min_value=0.0, max_value=20.0, value=5.0)
continent = st.selectbox("Continent", ['Asia', 'Europe', 'Africa', 'Americas', 'Oceania'])

# Convert continent to one-hot encoding
continent_mapping = {
    'Asia': [1, 0, 0, 0, 0],
    'Europe': [0, 1, 0, 0, 0],
    'Africa': [0, 0, 1, 0, 0],
    'Americas': [0, 0, 0, 1, 0],
    'Oceania': [0, 0, 0, 0, 1]
}
continent_encoded = continent_mapping[continent]

# Prepare the input data for prediction
user_input = np.array([spirit_servings, wine_servings, total_litres_of_pure_alcohol] + continent_encoded).reshape(1, -1)

# When the "Predict" button is clicked, make the prediction
if st.button("Predict"):
    prediction = model.predict(user_input)
    st.write(f"The predicted number of beer servings is: {prediction[0]:.2f}")
