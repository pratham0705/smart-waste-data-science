import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "models", "city_year_processed.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "models", "hotspot_data.csv")


def assign_hotspot_level(cluster_rank):
    if cluster_rank == 3:
        return "Critical Hotspot"
    elif cluster_rank == 2:
        return "High Hotspot"
    elif cluster_rank == 1:
        return "Medium Hotspot"
    else:
        return "Low Hotspot"


df = pd.read_csv(PROCESSED_DATA_PATH)

latest_year = df["Year"].max()
latest_df = df[df["Year"] == latest_year].copy()

features = [
    "Waste",
    "Recycling",
    "Population_Density",
    "Efficiency",
    "Awareness",
    "Landfill_Capacity"
]

X = latest_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
latest_df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_summary = latest_df.groupby("Cluster")["Waste"].mean().sort_values()
cluster_rank_map = {
    cluster: rank for rank, cluster in enumerate(cluster_summary.index)
}

latest_df["Cluster_Rank"] = latest_df["Cluster"].map(cluster_rank_map)
latest_df["Hotspot_Level"] = latest_df["Cluster_Rank"].apply(assign_hotspot_level)

latest_df["Raw_Hotspot_Score"] = (
    latest_df["Waste"] * 0.40
    + latest_df["Population_Density"] * 0.20
    - latest_df["Recycling"] * 200
    - latest_df["Efficiency"] * 500
    - latest_df["Awareness"] * 200
    - latest_df["Landfill_Capacity"] * 0.05
)

min_score = latest_df["Raw_Hotspot_Score"].min()
max_score = latest_df["Raw_Hotspot_Score"].max()

latest_df["Hotspot_Score"] = (
    (latest_df["Raw_Hotspot_Score"] - min_score)
    / (max_score - min_score)
    * 100
).round(2)

latest_df = latest_df.sort_values("Hotspot_Score", ascending=False)

latest_df.to_csv(OUTPUT_PATH, index=False)

print("Hotspot Detection Completed")
print("---------------------------")
print(f"Latest Year Used: {latest_year}")
print(f"Saved File: models/hotspot_data.csv")

print("\nTop 10 Hotspot Cities:")
print(
    latest_df[
        ["City", "Waste", "Recycling", "Efficiency", "Landfill_Capacity", "Hotspot_Level", "Hotspot_Score"]
    ].head(10)
)