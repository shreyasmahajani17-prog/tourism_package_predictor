import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -----------------------------
# Load Model
# -----------------------------
# Download and load the trained model
model_path = hf_hub_download(repo_id="tourism_package_predictor/tourism-package-prediction", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# -----------------------------
# App UI
# -----------------------------
st.title("🧳 Wellness Tourism Package Purchase Predictor")

st.write("""
This application predicts whether a customer is likely to **purchase a Wellness Tourism Package** based on demographic and interaction attributes.

Please enter the customer details below to generate a prediction.
""")

# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input("Age", min_value=18, max_value=100, value=30)

typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business Owner", "Student", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])

num_visitors = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
property_star = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=1)
passport = st.selectbox("Passport Available", ["No", "Yes"])
own_car = st.selectbox("Owns a Car", ["No", "Yes"])
num_children = st.number_input("Number of Children Visiting (Below 5 Years)", min_value=0, max_value=5, value=0)

designation = st.selectbox("Designation Level", ["Executive", "Senior Manager", "Manager", "Trainee", "Other"])
income = st.number_input("Monthly Income (INR)", min_value=5000, max_value=1000000, value=50000)

pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Premium"])

num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)
pitch_duration = st.number_input("Duration of Pitch (Minutes)", min_value=1, max_value=120, value=15)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_visitors,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": pitch_duration
}])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✔ The customer is **LIKELY to purchase** the package.\n\nConfidence: **{probability:.2f}%**")
    else:
        st.error(f"✖ The customer is **NOT likely to purchase** the package.\n\nConfidence: **{probability:.2f}%**")

