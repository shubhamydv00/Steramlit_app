# import streamlit as st



import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pickled model
with open('banglore_home_prices_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('House Price Prediction')

sqft = st.slider("Area (in square feet)", 500, 10000)
bath = st.slider("Number of bedrooms", 1, 5)
bhk = st.slider("Number of bathrooms", 1, 5)
location = st.text_input("Location")

def predict():
    float_features = [float(x) for x in [sqft, bath, bhk]]
    location_encoded = location_mapper[location] # You can encode the location using a dictionary or other encoding technique
    final_features = [np.array(float_features + [location_encoded])]
    prediction = model.predict(final_features)
    label = prediction[0]
    st.write('The predicted price is:', label)

trigger = st.button('Predict', on_click=predict)



# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # Load the pickled model
# with open('banglore_home_prices_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# st.title('house price prediction ')

# sqft = st.slider("Area ", 500, 10000)
# bath = st.slider("Bedroom", 1, 5)
# bhk = st.slider("Bathroom", 1, 5)

# def predict():
#     float_features = [float(x) for x in [sqft, bath, bhk]]
#     final_features = [np.array(float_features)]
#     prediction = model.predict(final_features)
#     label = prediction[0]
#     st.write('The predicted price is:', label)

# trigger = st.button('Predict', on_click=predict)

# import pandas as pd
# import numpy as np
# import pickle

# # Load the pickled model
# with open('banglore_home_prices_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Define a function to predict the price
# def predict_price(area, bhk, bathrooms):
#     input_data = np.array([[area, bhk, bathrooms]])
#     price = model.predict(input_data)[0]
#     return price

# # Define the app
# def app():
#     st.set_page_config(page_title='Bangalore House Price Prediction', page_icon=':money_with_wings:')
#     st.title('Bangalore House Price Prediction')

#     # Define the input form
#     st.sidebar.title('Enter House Details')
#     area = st.sidebar.slider('Area (in square feet)', 500, 10000, 1000)
#     bhk = st.sidebar.slider('BHK', 1, 10, 2)
#     bathrooms = st.sidebar.slider('Bathrooms', 1, 5, 2)

#     # Predict the price
#     price = predict_price(area, bhk, bathrooms)

#     # Show the predicted price to the user
#     st.write(f"Estimated house price is ₹{price:.2f} lakhs")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # Load the pickled model
# with open('banglore_home_prices_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# st.title('What is the type IRIS? :blossom:')

# sqft= st.slider("Area ",500,10000)
# bath= st.slider("Bedroom",1,5)
# bhk = st.slider("Bathroom",1,5)
# # petal_width = st.slider("Petal Width ",0.1,5.8)

# def predict():
#     float_features = [float(x) for x in [sqft,bath ,bhk ]]
#     final_features = [np.array(float_features)]
#     prediction = model.predict(final_features)
#     label = prediction[0]
    
#     print(type(label))
#     print(label)

#     st.success('The Price is  : ' + str(label) + ' :thumbsup:')
    
# trigger = st.button('Predict', on_click=predict)



#     # Predict the price
#     price = predict_price(area, bhk, bathrooms)

#     # Show the predicted price to the user
#     st.write(f"Estimated house price is ₹{price:.2f} lakhs")

# if __name__ == '__main__':
#     app()

