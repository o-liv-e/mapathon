"""Reproducible UrbanPulse RF/XGBoost training from promoted weak labels.

This is a weak-label proof of concept, not an independently verified accuracy benchmark.
"""
from pathlib import Path
import sys
import geopandas as gpd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urbanpulse.ml import load_labels, prepare_labels, train_models, save_bundle

features_path=ROOT/'data'/'training'/'promoted_weak_feature_cells_2025.geojson'
labels_path=ROOT/'data'/'training'/'promoted_weak_labels_2025.csv'
out=ROOT/'models'/'urbanpulse_production.joblib'

gdf=gpd.read_file(features_path)
labels=load_labels(str(labels_path))
gdf_labeled, y=prepare_labels(gdf, labels)
models, cols, evaluations, metadata, table=train_models(gdf_labeled,y)
save_bundle(str(out),models,cols,metadata)
print('selected:',metadata['selected_model'])
for name,e in evaluations.items():
    print(name,'spatial_macro_f1=',round(e['cv_report']['macro avg']['f1-score'],4))
