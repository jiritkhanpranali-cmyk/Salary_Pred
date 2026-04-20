# app.py

import streamlit as st
import pandas as pd
import pickle
import io

# -----------------------------
# 1. Load model
# -----------------------------
st.title("💼 Salary Prediction App")

try:
    with open("random_forest_regressor.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# 2. Load dataset (for preview only)
# -----------------------------
df = pd.read_csv(
    "https://raw.githubusercontent.com/jiritkhanpranali-cmyk/Salary_Pred/refs/heads/main/Salary_Data.csv"
)

st.write("## 📊 Dataset Preview")
st.dataframe(df.head())

st.write("## ℹ️ Dataset Info")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

st.write("## ❓ Missing Values")
st.write(df.isnull().sum())

# -----------------------------
# 3. User Input UI
# -----------------------------
st.write("## 🎯 Enter Employee Details")

age = st.slider("Age", 18, 65, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

education_level = st.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

job_title = st.text_input("Job Title", "Software Engineer")

years_of_experience = st.slider("Years of Experience", 0, 40, 5)

# -----------------------------
# 4. Encoding (must match training)
# -----------------------------

gender_encoded = 1 if gender == "Male" else 0

education_mapping = {
    "Bachelor's": 0,
    "Master's": 1,
    "PhD": 2
}
education_encoded = education_mapping[education_level]

# ⚠️ IMPORTANT NOTE:
# This is a placeholder. Must match training LabelEncoder.
job_title_encoded = abs(hash(job_title)) % 200  # safer fallback encoding

# -----------------------------
# 5. Prediction input
# -----------------------------
input_data = pd.DataFrame([[
    age,
    gender_encoded,
    education_encoded,
    job_title_encoded,
    years_of_experience
]], columns=[
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience"
])

# -----------------------------
# 6. Prediction
# -----------------------------
st.write("## 🔮 Prediction")

if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")
