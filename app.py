import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

print(os.getcwd())


# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="Application ML", layout="wide")

# =====================================
# STYLE CSS
# =====================================
st.markdown("""
<style>
.main {
    background-color: #F4F6F9;
}
h1 {
    color: #1E3A8A;
    text-align: center;
}
.stButton>button {
    background-color: #2563EB;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}
.card {
    padding: 20px;
    background-color: white;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Application de Prédiction Machine Learning")

# =====================================
# LOAD MODELS
# =====================================


with open("census.pkl", "rb") as f:
    census_model = pickle.load(f)

with open("auto-mpg.pkl", "rb") as f:
    mpg_model = pickle.load(f)

with open("scaler_census.pkl", "rb") as f:
    census_scaler = pickle.load(f)

with open("scaler_mpg.pkl", "rb") as f:
    mpg_scaler = pickle.load(f)

with open("imputer_mpg.pkl", "rb") as f:
    mpg_imputer = pickle.load(f)

with open("census_columns.pkl", "rb") as f:
    census_columns = pickle.load(f)

# =====================================
# SECTION 1 – CENSUS
# =====================================
st.header("Prédiction du revenu")

age = st.number_input("Age", 0, 100, 30)
education = st.number_input("Education", 1, 20, 10)
hours = st.number_input("Hours per week", 1, 80, 40)
population = st.number_input("Population", 0, 100000, 5000)

if st.button("Prédire revenu"):

    df = pd.DataFrame([[0]*len(census_columns)], columns=census_columns)

    if "Age" in df.columns:
        df["Age"] = age
    if "Education" in df.columns:
        df["Education"] = education
    if "HoursPerWeek" in df.columns:
        df["HoursPerWeek"] = hours
    if "Population" in df.columns:
        df["Population"] = population

    df_scaled = census_scaler.transform(df)

    prediction = census_model.predict(df_scaled)[0]

    if prediction == 1:
        st.success("Revenu ≤ 50K")
    else:
        st.success("Revenu > 50K")


# =====================================
# SECTION 2 – MPG
# =====================================
st.header("Prédiction consommation voiture (MPG)")

cyl = st.number_input("Cylinders", 2, 12, 4)
disp = st.number_input("Displacement", 50, 600, 150)
hp = st.number_input("Horsepower", 40, 250, 90)
weight = st.number_input("Weight", 1500, 6000, 2500)
acc = st.number_input("Acceleration", 5.0, 30.0, 15.0)
year = st.number_input("Model Year", 70, 83, 76)


if st.button("Prédire MPG"):

    df = pd.DataFrame([[cyl, disp, hp, weight, acc, year]])

    df_scaled = mpg_scaler.transform(df)
    df_scaled = mpg_imputer.transform(df_scaled)

    prediction = mpg_model.predict(df_scaled)[0]

    st.success(f"Consommation estimée : {prediction:.2f} MPG")
    