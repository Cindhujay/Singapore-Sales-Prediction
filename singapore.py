import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import re
import json
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title= 'Singapore Sales Prediction',page_icon='🏛️',layout="wide")


st.write("""
<div style='text-align:center'>
    <h1 style='color:#7200F9;'> Singapore Flats Sale Prediction</h1>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(['Home',"Prediction with DT RF & LR"])
with tab1:
    col1,col2 = st.columns([2,2])
    with col1:
        st.image(r'D:\Singapore\sing images.jpg')

    with col2:
        st.header(""":blue Singapore sales prediction""")
        st.write(""":red[Problem Statement:]:black There are many factors that affect the selling price of resale apartments
                  in Singapore. Hence, the objective of implementing decision tree regression is to examine the 
                 relationship between the selling price of a resale flat and its various attributes, including its distance to the Central Business District (CBD), 
                 distance to the nearest MRT station, flat size, floor level, and remaining lease duration.""")
        st.write(''':red[Results:] 
                    :black The project will benefit both potential buyers and sellers in the Singapore housing market.
                    Buyers can use the application to estimate resale prices and make informed decisions, while sellers can get an idea of their flat's potential market value. Additionally, the project demonstrates the practical application of machine learning in real estate and web development.
                ''') 

        with tab2:
        # Load CSV files
            town = pd.read_csv(r'D:\Singapore\ResaleFlatPricesBasedonApprovalDate19901999.csv')
            flat_type = pd.read_csv(r'D:\Singapore\ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv')
            street_name = pd.read_csv(r'D:\Singapore\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv')
            storey_range = pd.read_csv(r'D:\Singapore\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv')
            flat_model = pd.read_csv(r'D:\Singapore\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv')

            # Define the possible values for the dropdown menus
            month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            Town = town['town'].unique()
            Flat_type = flat_type['flat_type'].unique()
            Street_name = street_name['street_name'].unique()
            Storey_range = storey_range['storey_range'].unique()
            Flat_model = flat_model['flat_model'].unique()

            # Define the widgets for user input
            with st.form("my_form"):
                col1, col2, col3 = st.columns([5, 2, 5])
                with col1:
                    st.write(' ')
                    month = st.selectbox("month", month, key=1)
                    Town = st.selectbox("Department", Town, key=2)
                    Flat_type = st.selectbox('Flat Type', Flat_type, key=3)
                    Block = st.number_input("Enter block", value=1, step=1)
                    Street_name = st.selectbox("Enter street name", Street_name, key=4)

                with col3:
                    Storey_range = st.selectbox("Storey Range", Storey_range, key=5)
                    Floor_area_sqm = st.number_input("Enter floor area (sqm)", value=50.0, step=0.1)
                    Flat_model = st.selectbox("Flat Model", Flat_model, key=6)
                    Lease_commence_date = st.number_input("Enter Lease commence date", value=1998, step=1)
                    Year = st.number_input("Enter the year", value=1998, step=1)
                    submit_button = st.form_submit_button(label="PREDICT Resale Price")
                    st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                            background-color: #009999;
                            color: white;
                            width: 100%;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                import pickle
                with open(r'D:\Singapore\dt.pkl', 'rb') as file:
                        dt = pickle.load(file)
                with open(r'D:\Singapore\rf.pkl', 'rb') as f:
                        rf = pickle.load(f)
                with open(r'D:\Singapore\lr.pkl', 'rb') as f:
                        lr = pickle.load(f)

                ns = np.array([month, Town, Flat_type, Block, Street_name, Storey_range,
                                Floor_area_sqm, Flat_model, Lease_commence_date, Year])

                # Label Encoding
                le = LabelEncoder()
                ns_encoded = le.fit_transform(ns)

                # Reshape back to the original structure
                ns_encoded = ns_encoded.reshape(1, -1)

                # Make predictions
                dt_prediction = dt.predict(ns_encoded)
                rf_prediction = rf.predict(ns_encoded)
                lr_prediction = lr.predict(ns_encoded)

                # Display predictions
                st.write('## :green[Predicted Resale Price:] ')
                st.write('### :red[Decision Tree Regressor] :', dt_prediction)
                st.write('### :red[Random Forest Regressor] :', rf_prediction)
                st.write('## :green[Predicted Resale Price:] ', lr_prediction)    