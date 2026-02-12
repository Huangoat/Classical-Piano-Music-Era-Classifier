import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = (PROJECT_ROOT / "data" / "processed" / "features_unnormalized.csv").as_posix()
OUTPUT_DIR = (PROJECT_ROOT / "data" / "processed" / "features.csv").as_posix()

SCALER_DIR = (PROJECT_ROOT / "data" / "scaler" / "scaler.pkl").as_posix()
df = pd.read_csv(SOURCE_DIR)

features_to_scale = df.drop(columns=["label", "filename"])
labels = df["label"]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_to_scale)

df_normalized = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
df_normalized["label"] = labels.values

df_normalized.to_csv(OUTPUT_DIR, index=False)
joblib.dump(scaler, SCALER_DIR)