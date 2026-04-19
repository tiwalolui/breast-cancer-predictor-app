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
    features = np.array([1.41272917e+01, 1.92896485e+01, 9.19690334e+01, 6.54889104e+02,
 9.63602812e-02, 1.04340984e-01, 8.87993158e-02, 4.89191459e-02,
 1.81161863e-01, 6.27976098e-02, 4.05172056e-01, 1.21685343e+00,
 2.86605923e+00, 4.03370791e+01, 7.04097891e-03, 2.54781388e-02,
 3.18937163e-02, 1.17961371e-02, 2.05422988e-02, 3.79490387e-03,
 1.62691898e+01, 2.56772232e+01, 1.07261213e+02, 8.80583128e+02,
 1.32368594e-01, 2.54265044e-01, 2.72188483e-01, 1.14606223e-01,
 2.90075571e-01, 8.39458172e-02])
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

    


