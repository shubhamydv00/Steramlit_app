# import streamlit as st


# import streamlit as st
# from predict_cost import predict
# import numpy as np
 
# st.title('Home price prediction')
 
# st.write('---')
 
# # area of the house
# area = st.slider('Area of the house', 1000, 5000, 1500)
 
# # no. of bedrooms in the house
# bedrooms = st.number_input('No. of bedrooms', min_value=0, step=1)
 
# # no. of balconies in the house
# balconies = st.radio('No. of balconies', (0, 1, 2 , 3))
 
# # how old is the house? (age)
# location = st.number_input('Location', min_value=0, step=1)
 
# if st.button('Predict House Price'):
#     cost = predict(np.array([[area, bedrooms, balconies, age]]))
#     st.text(cost[0])



# import streamlit as st
# import pandas as pd
# from sklearn import datasets
# from sklearn.ensemble import RandomForestClassifier
# import pickle
# import numpy as np

# from PIL import Image

# pickle_in = open('model_pickel','rb')
# classifier = pickle.load(pickle_in)

# def Welcome():
#     return 'WELCOME ALL!'

# def predict_price(location,sqft,bath,bhk):    
#     """Let's Authenticate the Banks Note 
#     This is using docstrings for specifications.
#     ---
#     parameters:  
#       - name: location
#         in: query
#         type: text
#         required: true
#       - name: sqft
#         in: query
#         type: number
#         required: true
#       - name: bath
#         in: query
#         type: number
#         required: true
#       - name: bhk
#         in: query
#         type: number
#         required: true
#     responses:
#         200:
#             description: The output values
        
#     """
#     #loc_index = np.where(X.columns==location)[0][0]

#     x = np.zeros(243)
#     x[0] = sqft
#     x[1] = bath
#     x[2] = bhk
#     #if loc_index >= 0:
#         #   x[loc_index] = 1

#     return classifier.predict([x])[0]

# def main():
#     st.title("Bangalore House Rate Prediction")
#     html_temp = """
#     <h2 style="color:black;text-align:left;"> Streamlit House prediction ML App </h2>
#     """

#     st.markdown(html_temp,unsafe_allow_html=True)
#     st.subheader('Please enter the required details:')
#     location = st.text_input("Location","")
#     sqft = st.text_input("Sq-ft area","")
#     bath = st.text_input("Number of Bathroom","")
#     bhk = st.text_input("Number of BHK","")

#     result=""

#     if st.button("House Price in Lakhs"):
#         result=predict_price(location,sqft,bath,bhk)
#     st.success('The output is {}'.format(result))
#     # if st.button("About"):
#         # st.text("Please find the code at")
#         # st.text("https://github.com/Lokeshrathi/Bangalore-house-s-rate")

# if __name__=='__main__':
#     main()



# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # Load the pickled model
# with open('banglore_home_prices_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Define the location encoding dictionary
# # location_mapper = {'Location1': 0, 'Location2': 1, 'Location3': 2}  # Update with your own encoding

# st.title('House Price Prediction')

# location = st.text_input("Location")
# sqft = st.slider("Area (in square feet)", 500, 10000)
# bath = st.slider("Number of bedrooms", 1, 5)
# bhk = st.slider("Number of bathrooms", 1, 5)


# def predict():
#     location_encoded = location_mapper[location]
#     float_features = [float(x) for x in [sqft, bath, bhk]]
#     predict_price = [np.array([location_encoded]+float_features)]
#     prediction = model.predict(predict_price)
#     label = prediction[0]
#     st.write('The predicted price is:', label)

# trigger = st.button('Predict', on_click=predict)



# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # Load the pickled model
# with open('banglore_home_prices_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Define the location encoding dictionary
# location_mapper = {'Location1': 0, 'Location2': 1, 'Location3': 2}  # Update with your own encoding

# st.title('House Price Prediction')

# location = st.text_input("Location")
# sqft = st.slider("Area (in square feet)", 500, 10000)
# bath = st.slider("Number of bedrooms", 1, 5)
# bhk = st.slider("Number of bathrooms", 1, 5)


# def predict():
#     location_encoded = location_mapper[location]
#     float_features = [float(x) for x in [sqft, bath, bhk]]
#     predict_price = [np.array([location_encoded]+float_features)]
#     prediction = model.predict(predict_price)
#     label = prediction[0]
#     st.write('The predicted price is:', label)

# trigger = st.button('Predict', on_click=predict)

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pickled model
with open('banglore_home_prices_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('House Price Prediction')

# Example location_mapper
location_mapper = {'location1': 1, 'location2': 2, 'location3': 3}

location = st.text_input("Location")
sqft = st.slider("Area (in square feet)", 500, 10000)
bath = st.slider("Number of bedrooms", 1, 5)
bhk = st.slider("Number of bathrooms", 1, 5)
X = pd.read_csv("X.csv", index_col=0)

def predict_price():    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    print(x)
    st.success(model.predict([x])[0])

trigger = st.button('Predict', on_click=predict_price)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # Load the pickled model
# with open('banglore_home_prices_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# st.title('House Price Prediction')


# location = st.text_input("Location")
# sqft = st.slider("Area (in square feet)", 500, 10000)
# bath = st.slider("Number of bedrooms", 1, 5)
# bhk = st.slider("Number of bathrooms", 1, 5)


# def predict():
#     location_encoded = location_mapper[location]
#     float_features = [float(x) for x in [sqft, bath, bhk]]
#      # You can encode the location using a dictionary or other encoding technique
#     predict_price = [np.array([location_encoded]+float_features )]
#     prediction = model.predict(predict_price)
#     label = prediction[0]
#     st.write('The predicted price is:', label)

# trigger = st.button('Predict', on_click=predict)



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

