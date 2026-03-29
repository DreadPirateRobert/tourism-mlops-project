import streamlit as st
import pandas as pd
import joblib, os
from huggingface_hub import hf_hub_download

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tourism Package Predictor", page_icon="✈️", layout="wide")

# ─── Load model from HF Hub ───────────────────────────────────────────────────
HF_USERNAME = "PratikSal13"

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="PratikSal13/tourism-best-model",
        filename="best_model.pkl",
        token=os.environ.get("HF_TOKEN"),
    )
    return joblib.load(path)

model = load_model()

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.markdown("Predict whether a customer will purchase the **Wellness Tourism Package**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")
    age                      = st.number_input("Age", 18, 100, 35)
    type_of_contact          = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city_tier                = st.selectbox("City Tier", [1, 2, 3])
    occupation               = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
    gender                   = st.selectbox("Gender", ["Female", "Male"])
    marital_status           = st.selectbox("Marital Status", ["Divorced", "Married", "Single", "Unmarried"])
    designation              = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
    monthly_income           = st.number_input("Monthly Income (₹)", 10000, 100000, 20000, step=500)
    passport                 = st.selectbox("Has Passport", ["No", "Yes"])
    own_car                  = st.selectbox("Owns a Car", ["No", "Yes"])

with col2:
    st.subheader("Interaction & Trip Details")
    duration_of_pitch        = st.number_input("Duration of Pitch (mins)", 1, 60, 15)
    number_of_followups      = st.number_input("Number of Follow-ups", 1, 10, 3)
    product_pitched          = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    preferred_property_star  = st.selectbox("Preferred Property Stars", [3, 4, 5])
    number_of_person_visiting = st.number_input("Persons Visiting", 1, 10, 2)
    number_of_children       = st.number_input("Children Below 5 Visiting", 0, 5, 0)
    number_of_trips          = st.number_input("Annual Trips", 1, 20, 3)

# Encode inputs (must match LabelEncoder order used during training)
type_map        = {"Company Invited": 0, "Self Enquiry": 1}
occupation_map  = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
gender_map      = {"Female": 0, "Male": 1}
product_map     = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
marital_map     = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
designation_map = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}
binary_map      = {"No": 0, "Yes": 1}

input_df = pd.DataFrame([{
    "Age":                       age,
    "TypeofContact":             type_map[type_of_contact],
    "CityTier":                  city_tier,
    "DurationOfPitch":           duration_of_pitch,
    "Occupation":                occupation_map[occupation],
    "Gender":                    gender_map[gender],
    "NumberOfPersonVisiting":    number_of_person_visiting,
    "NumberOfFollowups":         number_of_followups,
    "ProductPitched":            product_map[product_pitched],
    "PreferredPropertyStar":     preferred_property_star,
    "MaritalStatus":             marital_map[marital_status],
    "NumberOfTrips":             number_of_trips,
    "Passport":                  binary_map[passport],
    "PitchSatisfactionScore":    pitch_satisfaction_score,
    "OwnCar":                    binary_map[own_car],
    "NumberOfChildrenVisiting":  number_of_children,
    "Designation":               designation_map[designation],
    "MonthlyIncome":             monthly_income,
}])

st.divider()
if st.button("🔍 Predict Purchase Likelihood", type="primary", use_container_width=True):
    pred  = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.success(f"✅ **LIKELY to purchase** the Wellness Package  ·  Confidence: {proba:.1%}")
    else:
        st.error(f"❌ **UNLIKELY to purchase** the Wellness Package  ·  Confidence: {1-proba:.1%}")

    with st.expander("View input data"):
        st.dataframe(input_df, use_container_width=True)
