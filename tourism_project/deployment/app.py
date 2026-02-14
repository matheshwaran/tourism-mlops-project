"""
Streamlit Web Application for Tourism Package Purchase Prediction.
Loads the trained model from Hugging Face Model Hub and provides
an interactive UI for predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# ---- Page Configuration ----
st.set_page_config(
    page_title="Tourism Package Predictor", page_icon="✈️", layout="wide"
)


@st.cache_resource
def load_model():
    """Load the trained model from Hugging Face Model Hub."""
    model_repo_id = "Matheshrangasamy/tourism-model"

    model_path = hf_hub_download(
        repo_id=model_repo_id,
        filename="best_model.joblib",
        repo_type="model",
    )

    feature_path = hf_hub_download(
        repo_id=model_repo_id,
        filename="feature_names.joblib",
        repo_type="model",
    )

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_path)
    return model, feature_names


# Load model
model, feature_names = load_model()

# ---- App Title ----
st.title("✈️ Tourism Wellness Package - Purchase Prediction")
st.markdown(
    "Predict whether a customer will purchase the **Wellness Tourism Package**."
)
st.markdown("---")

# ---- Sidebar Inputs ----
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 65, 35)
type_of_contact = st.sidebar.selectbox(
    "Type of Contact", ["Self Enquiry", "Company Invited"]
)
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.sidebar.slider("Duration of Pitch (minutes)", 5, 40, 15)
occupation = st.sidebar.selectbox(
    "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
num_persons_visiting = st.sidebar.slider("Number of Persons Visiting", 1, 5, 3)
num_followups = st.sidebar.slider("Number of Followups", 1, 6, 3)
product_pitched = st.sidebar.selectbox(
    "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)
preferred_property_star = st.sidebar.slider("Preferred Property Star", 3, 5, 3)
marital_status = st.sidebar.selectbox(
    "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
)
num_trips = st.sidebar.slider("Number of Trips (Annual)", 1, 10, 3)
passport = st.sidebar.selectbox(
    "Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.sidebar.selectbox(
    "Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
num_children_visiting = st.sidebar.slider("Number of Children Visiting", 0, 3, 1)
designation = st.sidebar.selectbox(
    "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
monthly_income = st.sidebar.slider("Monthly Income", 10000, 40000, 20000)

# ---- Encode inputs ----
type_of_contact_enc = 0 if type_of_contact == "Self Enquiry" else 1
occupation_map = {
    "Salaried": 0,
    "Small Business": 1,
    "Large Business": 2,
    "Free Lancer": 3,
}
gender_enc = 0 if gender == "Male" else 1
product_map = {"Basic": 0, "Standard": 1, "Deluxe": 2, "Super Deluxe": 3, "King": 4}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Unmarried": 3}
designation_map = {
    "Executive": 0,
    "Manager": 1,
    "Senior Manager": 2,
    "AVP": 3,
    "VP": 4,
}

# ---- Build input dataframe ----
input_data = pd.DataFrame(
    {
        "Age": [age],
        "TypeofContact": [type_of_contact_enc],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "Occupation": [occupation_map[occupation]],
        "Gender": [gender_enc],
        "NumberOfPersonVisiting": [num_persons_visiting],
        "NumberOfFollowups": [num_followups],
        "ProductPitched": [product_map[product_pitched]],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_map[marital_status]],
        "NumberOfTrips": [num_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [num_children_visiting],
        "Designation": [designation_map[designation]],
        "MonthlyIncome": [monthly_income],
    }
)

# Ensure column order matches training
input_data = input_data[feature_names]

# ---- Prediction ----
st.subheader("Input Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Occupation:** {occupation}")
    st.write(f"**Monthly Income:** ₹{monthly_income:,}")
with col2:
    st.write(f"**Contact Type:** {type_of_contact}")
    st.write(f"**City Tier:** {city_tier}")
    st.write(f"**Product Pitched:** {product_pitched}")
    st.write(f"**Designation:** {designation}")
with col3:
    st.write(f"**Marital Status:** {marital_status}")
    st.write(f"**Passport:** {'Yes' if passport else 'No'}")
    st.write(f"**Own Car:** {'Yes' if own_car else 'No'}")
    st.write(f"**Trips/Year:** {num_trips}")

st.markdown("---")

if st.button("🔮 Predict Purchase Likelihood", type="primary"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    col_left, col_right = st.columns(2)

    with col_left:
        if prediction == 1:
            st.success(
                "✅ **Customer is LIKELY to purchase** the Wellness Tourism Package!"
            )
        else:
            st.warning(
                "❌ **Customer is UNLIKELY to purchase** the Wellness Tourism Package."
            )

    with col_right:
        st.metric("Purchase Probability", f"{probability[1]*100:.1f}%")
        st.metric("No-Purchase Probability", f"{probability[0]*100:.1f}%")

st.markdown("---")
st.caption(
    "Built with Streamlit | Model: Gradient Boosting Classifier | Data: Tourism Customer Dataset"
)
