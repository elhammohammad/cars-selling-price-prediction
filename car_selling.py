import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
import joblib
from sklearn.linear_model import LinearRegression


st.title('Car prices prediction App')
st.image('car_prices.webp' , width = 350)
st.text('This application is designed to predict the car price')
st.text('based on the features you will select')
year = st.slider('year' , 1994 , 2020)
a1 , a2 = st.columns(2)
km_driven = a1.number_input("KM driven" , step = 1)
mileage = a1.number_input("mileage" , step = 5)
engine = a1.number_input("engine" , step = 100)
max_power = a1.number_input("max_power" , step = 10)
torque = a1.number_input("torque" , step = 5)
seats = a2.number_input("seats" , step= 1)
RPM = a2.number_input("RPM" , step = 100)
fuel = a2.selectbox("Fuel Type" , ['Diesel', 'Petrol', 'LPG', 'CNG'])
seller_type = a2.selectbox("Seller" , ['Individual', 'Dealer', 'Trustmark Dealer'])
owner = a2.selectbox("Owner Type" , ['First Owner', 'Second Owner', 'Third Owner','Fourth & Above Owner', 'Test Drive Car'])
transmission = a1.radio('Transmission Type' , ['Manual', 'Automatic'])
btn = st.button('Predict Price')
if btn == True :
    scaler = joblib.load('scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    model = joblib.load('model.pkl')
    
    #encoding cat. data
    fuel_map = {'Diesel' : 1, 'Petrol' : 3 , 'LPG' : 2 , 'CNG' : 0}
    seller_map = {'Individual': 1, 'Dealer' : 0, 'Trustmark Dealer' : 2}
    owner_map = {'First Owner' : 0, 'Second Owner' : 2, 'Third Owner' : 4,'Fourth & Above Owner' : 1, 'Test Drive Car' :3}
    transmission_map = {'Manual': 1, 'Automatic' : 0}

    fuel_encoded = fuel_map[fuel]
    seller_encoded = seller_map[seller_type]
    owner_encoded = owner_map[owner]
    transmission_encoded = transmission_map[transmission]

    input_data = np.array([[year , km_driven , mileage , engine , max_power ,torque , seats , RPM , fuel_encoded , seller_encoded , owner_encoded , transmission_encoded ]])
    input_data_scaled = scaler.transform(input_data)
    pred_scaled = model.predict(input_data_scaled)
    pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1 , 1))
    
    st.success('The predicted price is : ' + str(pred_original))

# Feature comparison section
st.title("Interactive Feature-Price Comparison")
#importing data
df = pd.read_csv('processed_dataset.csv')

# List of features to compare with price
features = ['year', 'fuel', 'seller_type', 'transmission', 'owner','km_driven', 'mileage', 'engine', 'max_power', 'torque', 'RPM', 'seats']
# User selects a feature
selected_feature = st.radio("Choose a feature to compare with Selling Price:", features)
# Dynamically generate a chart based on the selected feature
st.write(f"Comparing Selling Price with **{selected_feature}**:")

if selected_feature in ['year', 'fuel', 'seller_type', 'transmission', 'owner']:
    # Bar chart for categorical features (average selling_price)
    avg_prices = df.groupby(selected_feature)['selling_price'].mean().reset_index()
    fig = px.bar(
        avg_prices,
        x=selected_feature,
        y='selling_price',
        color=selected_feature,
        title=f"Average Selling Price by {selected_feature}",
        labels={'selling_price': 'Average Selling Price'},
        text='selling_price'
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')


elif selected_feature in ['km_driven', 'mileage', 'engine', 'max_power', 'torque', 'RPM', 'seats']:
    # Create a correlation matrix for the selected feature and selling price
    correlation_data = df[[selected_feature, 'selling_price']].corr()

    # Plot heatmap
    fig = px.imshow(
        correlation_data,
        labels={'x': 'Features', 'y': 'Features', 'color': 'Correlation'},
        title=f"Correlation Heatmap for {selected_feature.capitalize()} and Selling Price",
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1
    )

else:
    st.error("Feature not recognized.")

# Display the chart
st.plotly_chart(fig, use_container_width=True)


