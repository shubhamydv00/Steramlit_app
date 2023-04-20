import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
model = pickle.load(open('banglore_home_prices_model.pkl', 'rb'))

# Define a function to predict the price
def predict_price(area, bhk, bathrooms):
    input_data = np.array([[area, bhk, bathrooms]])
    price = model.predict(input_data)[0]
    return price

# Define the app
def app():
    st.set_page_config(page_title='Bangalore House Price Prediction', page_icon=':money_with_wings:')
    st.title('Bangalore House Price Prediction')

    # Define the input form
    st.sidebar.title('Enter House Details')
    area = st.sidebar.slider('Area (in square feet)', 500, 10000, 1000)
    bhk = st.sidebar.slider('BHK', 1, 10, 2)
    bathrooms = st.sidebar.slider('Bathrooms', 1, 5, 2)

    # Predict the price
    price = predict_price(area, bhk, bathrooms)

    # Show the predicted price to the user
    st.write(f"Estimated house price is ₹{price:.2f} lakhs")

if __name__ == '__main__':
    app()
