import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "waste_india.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df = df.rename(columns={
    "City/District": "City",
    "Waste Type": "Waste_Type",
    "Waste Generated (Tons/Day)": "Waste",
    "Recycling Rate (%)": "Recycling",
    "Population Density (People/km²)": "Population_Density",
    "Municipal Efficiency Score (1-10)": "Efficiency",
    "Disposal Method": "Disposal",
    "Cost of Waste Management (₹/Ton)": "Cost",
    "Awareness Campaigns Count": "Awareness",
    "Landfill Name": "Landfill_Name",
    "Landfill Location (Lat, Long)": "Landfill_Location",
    "Landfill Capacity (Tons)": "Landfill_Capacity",
    "Year": "Year"
})

df_city = df.groupby(["City", "Year"]).agg({
    "Waste": "sum",
    "Recycling": "mean",
    "Population_Density": "mean",
    "Efficiency": "mean",
    "Cost": "mean",
    "Awareness": "mean",
    "Landfill_Capacity": "mean"
}).reset_index()

features = [
    "City",
    "Recycling",
    "Population_Density",
    "Efficiency",
    "Cost",
    "Awareness",
    "Landfill_Capacity",
    "Year"
]

target = "Waste"

X = df_city[features]
y = df_city[target]

numeric_features = [
    "Recycling",
    "Population_Density",
    "Efficiency",
    "Cost",
    "Awareness",
    "Landfill_Capacity",
    "Year"
]

categorical_features = ["City"]

preprocessor = ColumnTransformer(
    transformers=[
        ("city", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Improved Model Training Completed")
print("---------------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

with open(os.path.join(MODEL_DIR, "waste_prediction_model.pkl"), "wb") as f:
    pickle.dump(pipeline, f)

with open(os.path.join(MODEL_DIR, "model_features.pkl"), "wb") as f:
    pickle.dump(features, f)

df_city.to_csv(os.path.join(MODEL_DIR, "city_year_processed.csv"), index=False)

print("\nSaved files:")
print("models/waste_prediction_model.pkl")
print("models/model_features.pkl")
print("models/city_year_processed.csv")