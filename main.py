import streamlit as st
import pandas as pd
from os import path
import numpy as np
import pickle #to read pickle file

st.title("Flower species predictor")

petal_length = st.number_input("Please choose a Petal length",placeholder = "Please choose a Petal length between 1.0 and 6.9" ,
                               min_value = 1.0, max_value = 6.9, value = None)
petal_width = st.number_input("Please choose a  Petal width",placeholder = "Please choose a Petal width between 0.1 and 2.5",
                              min_value = 0.1, max_value = 2.5, value = None)
sepal_length = st.number_input("Please choose a Sepal length", placeholder= "Please choose a Sepal length between 4.3 and 7.9",
                               min_value = 4.3, max_value = 7.9, value = None)
sepal_width = st.number_input("Please choose a  Sepal width", placeholder="Please choose a Sepal width between 2.0 and 4.4",
                              min_value = 2.0, max_value = 4.4, value = None)

#search = st.text_input("Search species: ", placeholder="e.g. setosa, virginica, versicolor")

#generate the dataframe for prediction
df_user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns = ["sepal_length","sepal_width","petal_length","petal_width"])
#using the .pkl file creating an ML model named "Iris predictor"
st.write(df_user_input)

model_path = path.join("Model","iris_classifier.pkl")
with open(model_path, 'rb') as file:
    iris_predictor = pickle.load(file)