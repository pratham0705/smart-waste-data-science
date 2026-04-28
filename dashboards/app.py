import os
import sys
import pickle
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.risk_engine import calculate_risk, risk_level
from src.collection_optimizer import (
    assign_collection_priority,
    assign_vehicle_requirement,
    assign_action_plan
)

st.set_page_config(
    page_title="SmartWaste AI System",
    page_icon="♻️",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
}
[data-testid="stSidebar"] {
    background-color: #161B22;
}
.card {
    background-color: #1C2128;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #30363D;
    margin-bottom: 15px;
}
.card h4 {
    color: #E6EDF3;
    margin-bottom: 8px;
}
.card p {
    color: #C9D1D9;
    font-size: 17px;
}
h1, h2, h3 {
    color: #E6EDF3;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    white-space: normal !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "models", "city_year_processed.csv"))
hotspot_df = pd.read_csv(os.path.join(BASE_DIR, "models", "hotspot_data.csv"))
feature_df = pd.read_csv(os.path.join(BASE_DIR, "models", "feature_importance.csv"))
landfill_df = pd.read_csv(os.path.join(BASE_DIR, "data", "delhi_landfills.csv"))

with open(os.path.join(BASE_DIR, "models", "waste_prediction_model.pkl"), "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("♻️ SmartWaste Filters")
st.sidebar.markdown("City-level waste prediction, risk analysis and collection planning")

city_list = sorted(df["City"].unique())
selected_city = st.sidebar.selectbox("🏙️ Select City", city_list)

selected_city_data = df[df["City"] == selected_city].sort_values("Year").iloc[-1]

st.sidebar.divider()
st.sidebar.metric("Current Waste", f"{selected_city_data['Waste']:,.0f}")
st.sidebar.metric("Recycling Rate", f"{selected_city_data['Recycling']:.1f}%")
st.sidebar.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------------
# Header
# -----------------------------
st.title("♻️ SmartWaste AI System")
st.markdown("### AI-Based Waste Prediction, Hotspot Detection, Risk Analysis & Smart Collection Planning")
st.success("This system supports live-ready decision-making using historical data, simulation inputs and Delhi landfill monitoring.")
st.info(f"Currently analyzing: **{selected_city}**")

# -----------------------------
# Common Risk Score
# -----------------------------
base_risk_score = calculate_risk(
    selected_city_data["Waste"],
    selected_city_data["Recycling"],
    selected_city_data["Efficiency"],
    selected_city_data["Awareness"],
    selected_city_data["Landfill_Capacity"]
)

base_risk_level = risk_level(base_risk_score)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview",
    "Prediction",
    "Hotspots",
    "Risk Analysis",
    "Recommendations",
    "Collection Plan",
    "Delhi Landfills",
    "Live Data Roadmap",
    "Model Performance"
])

# -----------------------------
# TAB 1: Overview
# -----------------------------
with tab1:
    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cities", df["City"].nunique())
    col2.metric("Total Records", len(df))
    col3.metric("Latest Year", int(df["Year"].max()))
    col4.metric("Average Waste", f"{df['Waste'].mean():,.0f}")

    st.markdown("""
    SmartWaste is an AI-based decision-support system for intelligent waste management. 
    It predicts waste generation, identifies high-risk hotspot regions, calculates waste risk, 
    suggests priority-based collection plans and provides Delhi-specific landfill monitoring.
    """)

    st.subheader("Selected City Historical Data")
    st.dataframe(
        df[df["City"] == selected_city].sort_values("Year"),
        use_container_width=True
    )

# -----------------------------
# TAB 2: Prediction
# -----------------------------
with tab2:
    st.subheader("⚡ Waste Prediction & Live Simulation")

    live_mode = st.toggle("Enable Live Simulation Mode", value=False)

    if live_mode:
        st.info("Adjust parameters to simulate current/live waste management conditions.")

        col1, col2, col3 = st.columns(3)

        with col1:
            recycling = st.slider(
                "Recycling Rate (%)",
                0,
                100,
                int(selected_city_data["Recycling"])
            )
            efficiency = st.slider(
                "Municipal Efficiency",
                1,
                10,
                int(round(selected_city_data["Efficiency"]))
            )

        with col2:
            awareness = st.slider(
                "Awareness Campaigns",
                0,
                30,
                int(round(selected_city_data["Awareness"]))
            )
            population_density = st.number_input(
                "Population Density",
                value=float(selected_city_data["Population_Density"])
            )

        with col3:
            landfill_capacity = st.number_input(
                "Landfill Capacity",
                value=float(selected_city_data["Landfill_Capacity"])
            )
            cost = st.number_input(
                "Cost of Waste Management",
                value=float(selected_city_data["Cost"])
            )

    else:
        recycling = selected_city_data["Recycling"]
        efficiency = selected_city_data["Efficiency"]
        awareness = selected_city_data["Awareness"]
        population_density = selected_city_data["Population_Density"]
        landfill_capacity = selected_city_data["Landfill_Capacity"]
        cost = selected_city_data["Cost"]

    input_data = pd.DataFrame([{
        "City": selected_city,
        "Recycling": recycling,
        "Population_Density": population_density,
        "Efficiency": efficiency,
        "Cost": cost,
        "Awareness": awareness,
        "Landfill_Capacity": landfill_capacity,
        "Year": selected_city_data["Year"] + 1
    }])

    prediction = model.predict(input_data)[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Waste Next Year", f"{prediction:,.2f}")
    col2.metric("Current Waste", f"{selected_city_data['Waste']:,.2f}")
    col3.metric("Prediction Year", int(selected_city_data["Year"] + 1))

    st.subheader("Historical Waste Trend")
    trend_data = df[df["City"] == selected_city].sort_values("Year")
    st.line_chart(trend_data.set_index("Year")["Waste"])

# -----------------------------
# TAB 3: Hotspots
# -----------------------------
with tab3:
    st.subheader("🔥 Top 10 Hotspot Cities")

    top_hotspots = hotspot_df[
        ["City", "Waste", "Recycling", "Efficiency", "Landfill_Capacity", "Hotspot_Level", "Hotspot_Score"]
    ].head(10)

    st.dataframe(top_hotspots, use_container_width=True)

    st.subheader("Hotspot Score Visualization")
    fig = px.bar(
        top_hotspots,
        x="Hotspot_Score",
        y="City",
        color="Hotspot_Level",
        orientation="h",
        title="Top 10 Waste Hotspot Cities",
        text="Hotspot_Score"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hotspot Level Distribution")
    hotspot_count = hotspot_df["Hotspot_Level"].value_counts().reset_index()
    hotspot_count.columns = ["Hotspot Level", "Count"]

    fig2 = px.bar(
        hotspot_count,
        x="Hotspot Level",
        y="Count",
        color="Hotspot Level",
        title="Distribution of Hotspot Levels"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TAB 4: Risk Analysis
# -----------------------------
with tab4:
    st.subheader("⚠️ Waste Risk Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", base_risk_score)
    col2.metric("Risk Level", base_risk_level)
    col3.metric("Landfill Capacity", f"{selected_city_data['Landfill_Capacity']:,.0f}")

    st.subheader("Risk Factor Details")

    risk_factors = pd.DataFrame({
        "Factor": [
            "Waste Generated",
            "Recycling Rate",
            "Municipal Efficiency",
            "Awareness Campaigns",
            "Landfill Capacity"
        ],
        "Value": [
            selected_city_data["Waste"],
            selected_city_data["Recycling"],
            selected_city_data["Efficiency"],
            selected_city_data["Awareness"],
            selected_city_data["Landfill_Capacity"]
        ]
    })

    st.dataframe(risk_factors, use_container_width=True)

    fig = px.bar(
        risk_factors,
        x="Factor",
        y="Value",
        title="Risk Factor Analysis",
        text="Value"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 5: Recommendations
# -----------------------------
with tab5:
    st.subheader("✅ Smart Recommendations")

    st.metric("Risk Level for Selected City", base_risk_level)

    if base_risk_score >= 75:
        st.error("Critical Risk: Immediate waste management action is required.")
        st.write("- Increase collection frequency immediately.")
        st.write("- Deploy additional collection vehicles.")
        st.write("- Improve landfill capacity planning.")
        st.write("- Start urgent public awareness campaigns.")
        st.write("- Prioritize this city in collection planning.")

    elif base_risk_score >= 50:
        st.warning("High Risk: Waste management improvement is required.")
        st.write("- Improve recycling and source segregation.")
        st.write("- Increase municipal operational efficiency.")
        st.write("- Monitor landfill capacity regularly.")
        st.write("- Run local awareness campaigns.")
        st.write("- Schedule more frequent collection.")

    elif base_risk_score >= 25:
        st.info("Medium Risk: Preventive action is suggested.")
        st.write("- Maintain regular collection schedule.")
        st.write("- Improve public participation.")
        st.write("- Monitor future waste growth.")
        st.write("- Strengthen recycling awareness.")

    else:
        st.success("Low Risk: Waste management indicators are stable.")
        st.write("- Continue existing waste management practices.")
        st.write("- Maintain recycling performance.")
        st.write("- Keep monitoring yearly trends.")

    st.subheader("Delhi-Focused Preventive Measures")

    st.markdown("""
    **Short-Term Measures**
    - Increase collection frequency in high-risk zones.
    - Improve waste segregation at source.
    - Deploy additional vehicles for hotspot regions.

    **Medium-Term Measures**
    - Improve recycling infrastructure.
    - Conduct awareness campaigns in dense localities.
    - Monitor landfill pressure regularly.

    **Long-Term Measures**
    - Promote waste-to-energy and composting solutions.
    - Use IoT-based smart bins for live monitoring.
    - Reduce dependency on landfill disposal.
    """)

# -----------------------------
# TAB 6: Collection Plan
# -----------------------------
with tab6:
    st.subheader("🚛 Smart Collection Plan")

    city_hotspot = hotspot_df[hotspot_df["City"] == selected_city]

    if not city_hotspot.empty:
        hotspot_level = city_hotspot.iloc[0]["Hotspot_Level"]
    else:
        hotspot_level = "Low Hotspot"

    priority = assign_collection_priority(base_risk_score, hotspot_level)
    vehicle = assign_vehicle_requirement(selected_city_data["Waste"])
    action_plan = assign_action_plan(priority)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
        <h4>Collection Priority</h4>
        <p>{priority}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
        <h4>Vehicle Requirement</h4>
        <p>{vehicle}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
        <h4>Hotspot Level</h4>
        <p>{hotspot_level}</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Recommended Action Plan")
    st.success(action_plan)

    collection_df = pd.DataFrame({
        "Parameter": [
            "Selected City",
            "Risk Score",
            "Risk Level",
            "Hotspot Level",
            "Collection Priority",
            "Vehicle Requirement",
            "Action Plan"
        ],
        "Value": [
            selected_city,
            base_risk_score,
            base_risk_level,
            hotspot_level,
            priority,
            vehicle,
            action_plan
        ]
    })

    st.dataframe(collection_df, use_container_width=True)

# -----------------------------
# TAB 7: Delhi Landfills
# -----------------------------
with tab7:
    st.subheader("🗺️ Delhi Landfill Monitoring Map")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Major Landfill Sites", 3)
    col2.metric("MSW Facility", "Narela-Bawana")
    col3.metric("Highest Legacy Waste", "Ghazipur")
    col4.metric("Focus Area", "Delhi NCR")

    fig = px.scatter_mapbox(
        landfill_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Landfill",
        hover_data={
            "Zone": True,
            "Commissioned_Year": True,
            "Estimated_Legacy_Waste_Lakh_MT": True,
            "Risk_Level": True,
            "Status": True,
            "Latitude": False,
            "Longitude": False
        },
        size="Estimated_Legacy_Waste_Lakh_MT",
        color="Risk_Level",
        zoom=9,
        height=550,
        title="Delhi Landfill Risk and Legacy Waste Map"
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Delhi Landfill Details")
    st.dataframe(landfill_df, use_container_width=True)

    st.info("""
    This module connects the AI-based waste prediction system with real-world landfill risk zones in Delhi.
    It makes the project more practical, location-specific and focused on landfill pressure reduction.
    """)

# -----------------------------
# TAB 8: Live Data Roadmap
# -----------------------------
with tab8:
    st.subheader("🔴 Live Data Integration Roadmap")

    st.success("Current system is designed as a live-ready decision-support framework.")

    st.markdown("""
    ### Current Implementation
    - Historical waste data is used for model training.
    - Dynamic user inputs simulate current/live conditions.
    - Delhi landfill map provides location-based monitoring.
    - Dashboard updates results based on selected city and parameters.

    ### Future Real-Time Integration
    - IoT smart bins can send live fill-level data.
    - Municipal APIs can provide daily collection records.
    - GPS data from collection vehicles can support route optimization.
    - Weather and event data can improve short-term waste forecasting.
    - Satellite or drone monitoring can support landfill tracking.

    ### Why This Matters
    The system is not limited to old data. It is designed so real-time sources can be connected later 
    without changing the core AI pipeline.
    """)

# -----------------------------
# TAB 9: Model Performance
# -----------------------------
with tab9:
    st.subheader("📊 Model Performance & Explainable AI")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Metrics")

        performance_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R2 Score"],
            "Value": [5240.52, 6198.89, -0.0666]
        })

        st.dataframe(performance_df, use_container_width=True)

        st.warning("""
        The dataset size is limited, so prediction accuracy is moderate.
        The system strength lies in its complete decision-support pipeline:
        prediction + hotspot detection + risk analysis + collection planning.
        """)

    with col2:
        st.subheader("Top Influencing Factors")

        top_features = feature_df.head(10).copy()
        top_features["Feature"] = top_features["Feature"].str.replace("num__", "", regex=False)
        top_features["Feature"] = top_features["Feature"].str.replace("city__City_", "", regex=False)

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Key Insights
    - Municipal Efficiency, Recycling Rate, Cost, Population Density and Awareness are major factors.
    - The project does not only predict waste but also explains the major causes behind waste variation.
    - This improves transparency and makes the system useful for decision-making.
    """)