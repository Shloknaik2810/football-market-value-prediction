import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Football Player Market Value Predictor",
    page_icon="⚽",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

def load_model():
    preprocessor = joblib.load("football_preprocessor.pkl")

    xgb_model = XGBRegressor()
    xgb_model.load_model("football_xgboost.json")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ])

    return model


model = load_model()

# Get the exact categorical values used during model training
categorical_columns = [
    "sub_position",
    "position",
    "foot",
    "competition_name",
    "competition_country",
    "competition_confederation"
]

onehot_encoder = (
    model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
)

category_map = dict(
    zip(categorical_columns, onehot_encoder.categories_)
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("⚽ Football Player Market Value Predictor")

st.write(
    "Predict a football player's estimated market value using "
    "a machine learning model trained on player characteristics, "
    "recent performance, and competition information."
)

st.divider()

# ---------------------------------------------------------
# PLAYER INFORMATION
# ---------------------------------------------------------

st.subheader("👤 Player Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age at valuation",
        min_value=15,
        max_value=45,
        value=24
    )

with col2:
    height = st.number_input(
        "Height (cm)",
        min_value=150,
        max_value=220,
        value=180
    )

with col3:
   foot = st.selectbox(
    "Preferred foot",
    category_map["foot"].tolist()
)

col1, col2 = st.columns(2)

with col1:
    position = st.selectbox(
    "Position",
    category_map["position"].tolist()
)

with col2:
    sub_position = st.selectbox(
    "Sub-position",
    category_map["sub_position"].tolist()
)

# ---------------------------------------------------------
# PERFORMANCE INFORMATION
# ---------------------------------------------------------

st.subheader("📊 Performance in Previous 365 Days")

col1, col2, col3 = st.columns(3)

with col1:
    appearances = st.number_input(
        "Appearances",
        min_value=1,
        max_value=100,
        value=25
    )

with col2:
    minutes = st.number_input(
        "Minutes played",
        min_value=1,
        max_value=6000,
        value=1800
    )

with col3:
    goals = st.number_input(
        "Goals",
        min_value=0,
        max_value=100,
        value=10
    )

col1, col2, col3 = st.columns(3)

with col1:
    assists = st.number_input(
        "Assists",
        min_value=0,
        max_value=100,
        value=5
    )

with col2:
    yellow_cards = st.number_input(
        "Yellow cards",
        min_value=0,
        max_value=30,
        value=3
    )

with col3:
    red_cards = st.number_input(
        "Red cards",
        min_value=0,
        max_value=10,
        value=0
    )

# ---------------------------------------------------------
# DERIVED PERFORMANCE FEATURES
# ---------------------------------------------------------

if minutes >= 450:
    goals_per_90 = (goals * 90) / minutes
    assists_per_90 = (assists * 90) / minutes
    has_sufficient_minutes = 1
else:
    goals_per_90 = 0
    assists_per_90 = 0
    has_sufficient_minutes = 0

# ---------------------------------------------------------
# COMPETITION INFORMATION
# ---------------------------------------------------------

st.subheader("🏆 Competition Information")

col1, col2, col3 = st.columns(3)

with col1:
    competition = st.selectbox(
        "Competition",
        [
            "premier-league",
            "laliga",
            "serie-a",
            "bundesliga",
            "ligue-1",
            "scottish-premiership",
            "eredivisie",
            "super-league-1",
            "Saudi Pro League",
            "MLS",
            "Unknown"
        ]
    )

with col2:
    competition_country = st.selectbox(
    "Competition country",
    category_map["competition_country"].tolist()
)

with col3:
    competition_confederation = st.selectbox(
    "Confederation",
    category_map["competition_confederation"].tolist()
)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

st.divider()

if st.button("⚽ Predict Market Value", type="primary"):

    # Create dataframe using EXACT training feature names
    input_data = pd.DataFrame({
        "age_at_valuation": [age],
        "sub_position": [sub_position],
        "position": [position],
        "foot": [foot],
        "height_in_cm": [height],
        "appearances_365d": [appearances],
        "minutes_365d": [minutes],
        "goals_365d": [goals],
        "assists_365d": [assists],
        "yellow_cards_365d": [yellow_cards],
        "red_cards_365d": [red_cards],
        "goals_per_90_adjusted": [goals_per_90],
        "assists_per_90_adjusted": [assists_per_90],
        "has_sufficient_minutes": [has_sufficient_minutes],
        "competition_name": [competition],
        "competition_country": [competition_country],
        "competition_confederation": [competition_confederation]
    })

    # Model predicts LOG market value
    predicted_log_value = model.predict(input_data)[0]

    # Convert back to actual EUR
    predicted_value = np.expm1(predicted_log_value)

    # Display result
    st.success("Prediction completed!")

    st.metric(
        label="Estimated Market Value",
        value=f"€{predicted_value:,.0f}"
    )

    # More readable representation
    if predicted_value >= 1_000_000:
        value_millions = predicted_value / 1_000_000

        st.info(
            f"Estimated value: **€{value_millions:.2f} million**"
        )

    elif predicted_value >= 1_000:
        value_thousands = predicted_value / 1_000

        st.info(
            f"Estimated value: **€{value_thousands:.0f} thousand**"
        )

    else:
        st.info(
            f"Estimated value: **€{predicted_value:.0f}**"
        )

    st.caption(
        "This prediction is an ML-based estimate and should not be "
        "interpreted as an actual transfer fee or guaranteed market value."
    )

# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

with st.expander("ℹ️ About the model"):
    st.write(
        "The prediction is generated using an XGBoost regression model "
        "trained on historical Transfermarkt player valuation data."
    )

    st.write(
        "**Final 2026 test performance:**"
    )

    st.write(
        "- R²: 0.6721"
    )

    st.write(
        "- RMSE: 0.9273"
    )

    st.write(
        "- MAE: 0.7283"
    )

    st.write(
        "- Mean Absolute Error in market value: approximately €3.54 million"
    )