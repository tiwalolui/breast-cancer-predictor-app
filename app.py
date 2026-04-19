import streamlit as st
import numpy as np
import pickle

with open("cancer_model.pkl","rb") as f:
    model = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

st.title("Breast Cancer Prediction App")
st.write("Enter the measurements below to predict whether a mass is malignant or benign.")

mean_radius = st.number_input("Mean Radius", min_value=6.0,max_value=30.0,value=14.0)
mean_texture = st.number_input("Mean Texture", min_value=9.0,max_value=40.0,value=19.0)
mean_concavity = st.number_input("Mean Concavity", min_value=0.0,max_value=0.5,value=0.1)
worst_radius = st.number_input("Worst Radius", min_value=7.0,max_value=40.0,value=16.0)
worst_concavity = st.number_input("Worst Concavity", min_value=0.0,max_value=1.3,value=0.3)

if st.button("Predict"):
    features = np.zeros(30)
    features[0] = mean_radius
    features[1] = mean_texture
    features[6] = mean_concavity
    features[20] = worst_radius
    features[26] = worst_concavity

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.success("Prediction: Benign (Not Cancerous)")
    else:
        st.error("Prediction: Malignant (Cancerous)")

    


