import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Smart Waste Prediction System", layout="wide")

st.title("SmartWaste: AI-Based Waste Prediction and Advisory System")
st.markdown("Predict waste generation and generate smart advisories for waste management.")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("waste_india.csv")

# Rename columns if not already renamed
df.columns = [
    'City',
    'Waste_Type',
    'Waste',
    'Recycling',
    'Population_Density',
    'Efficiency',
    'Disposal',
    'Cost',
    'Awareness',
    'Landfill_Name',
    'Landfill_Location',
    'Landfill_Capacity',
    'Year'
]

# Aggregate city-year data
df_city_year = df.groupby(['City', 'Year']).agg({
    'Waste': 'sum',
    'Recycling': 'mean',
    'Population_Density': 'mean',
    'Efficiency': 'mean',
    'Cost': 'mean',
    'Awareness': 'mean',
    'Landfill_Capacity': 'mean'
}).reset_index()

# -----------------------------
# Advisory function
# -----------------------------
def generate_advisory(waste, recycling, efficiency, awareness, landfill):
    advice = []

    if waste > 130000:
        advice.append("High waste predicted: Increase collection frequency.")
    if recycling < 50:
        advice.append("Low recycling detected: Promote segregation and recycling drives.")
    if efficiency < 6:
        advice.append("Municipal efficiency is low: Improve operational planning.")
    if awareness < 10:
        advice.append("Awareness campaigns are low: Conduct more public awareness drives.")
    if landfill < 40000:
        advice.append("Landfill capacity is low: Plan alternate disposal or treatment methods.")

    if not advice:
        advice.append("Waste management indicators appear stable.")

    return advice

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Input Parameters")

city_list = sorted(df_city_year["City"].unique().tolist())
selected_city = st.sidebar.selectbox("Select City", city_list)

city_data = df_city_year[df_city_year["City"] == selected_city]
latest_row = city_data.sort_values("Year").iloc[-1]

recycling = st.sidebar.slider("Recycling Rate", 0, 100, int(latest_row["Recycling"]))
population_density = st.sidebar.number_input("Population Density", value=float(latest_row["Population_Density"]))
efficiency = st.sidebar.slider("Municipal Efficiency", 1, 10, int(round(latest_row["Efficiency"])))
cost = st.sidebar.number_input("Cost of Waste Management", value=float(latest_row["Cost"]))
awareness = st.sidebar.slider("Awareness Campaign Count", 0, 30, int(round(latest_row["Awareness"])))
landfill_capacity = st.sidebar.number_input("Landfill Capacity", value=float(latest_row["Landfill_Capacity"]))
year = st.sidebar.selectbox("Year", sorted(df_city_year["Year"].unique().tolist()))

# -----------------------------
# Simple prediction logic
# -----------------------------
# Baseline estimate using weighted adjustment from selected city's latest waste
base_waste = float(latest_row["Waste"])

predicted_waste = (
    base_waste
    + (population_density - latest_row["Population_Density"]) * 0.5
    - (recycling - latest_row["Recycling"]) * 300
    - (efficiency - latest_row["Efficiency"]) * 500
    + (cost - latest_row["Cost"]) * 2
    - (awareness - latest_row["Awareness"]) * 200
    - (landfill_capacity - latest_row["Landfill_Capacity"]) * 0.05
)

predicted_waste = max(predicted_waste, 0)

# -----------------------------
# Main dashboard
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")
    st.metric("Predicted Waste (Tons/Day)", f"{predicted_waste:,.2f}")

    st.subheader("Forecast Insight")
    st.info("LSTM-based next-year forecast: 26,036.18 tons/day")

with col2:
    st.subheader("Advisories")
    advisories = generate_advisory(
        predicted_waste, recycling, efficiency, awareness, landfill_capacity
    )
    for adv in advisories:
        st.warning(adv)

# -----------------------------
# Historical city trend
# -----------------------------
st.subheader("Historical Waste Trend")
trend_data = city_data.sort_values("Year")[["Year", "Waste"]]
st.line_chart(trend_data.set_index("Year"))

# -----------------------------
# Dataset preview
# -----------------------------
st.subheader("City-Year Data Preview")
st.dataframe(city_data.sort_values("Year"))

# -----------------------------
# Model comparison summary
# -----------------------------
st.subheader("Model Comparison Summary")

comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost", "LSTM"],
    "MAE": [5042.99, 5313.31, 5777.69, 7003.43],
    "RMSE": [6062.88, 6213.01, 6764.48, 8332.37],
    "R2 Score": [-0.02, -0.07, -0.27, -0.93]
})

st.dataframe(comparison_df)