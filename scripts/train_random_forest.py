"""Train UrbanPulse Random Forest from a saved feature table and labels.

The interactive app is the preferred path. This script is for reproducible
batch training once a labelled dataset has been collected.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from urbanpulse.ml import load_labels, prepare_labels, train_random_forest, save_model

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True, help="Feature table CSV/Parquet containing cell_id/year/row/col")
ap.add_argument("--labels", required=True, help="Label CSV/GeoJSON")
ap.add_argument("--out", default="models/urbanpulse_rf.joblib")
args = ap.parse_args()

features_path = Path(args.features)
features = pd.read_parquet(features_path) if features_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(features_path)
labels_raw = load_labels(args.labels)
labels = prepare_labels(labels_raw, features)
model, cols, report, cm, importance, metadata = train_random_forest(features, labels)
save_model(args.out, model, cols, metadata)
print(f"Saved: {args.out}")
print(f"Spatial validation: {metadata['cv']}")
print(f"Macro-F1: {metadata['macro_f1']:.3f}")
print(f"Weighted-F1: {metadata['weighted_f1']:.3f}")
print("Top features:")
print(importance.head(15).to_string())
