import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_prediction_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "models", "feature_importance.csv")

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values("Importance", ascending=False)

importance_df.to_csv(OUTPUT_PATH, index=False)

print("Feature Importance Generated")
print("----------------------------")
print("Saved File: models/feature_importance.csv")
print("\nTop 15 Important Features:")
print(importance_df.head(15))