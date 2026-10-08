# UrbanPulse ML completion plan

## Current state

UrbanPulse already has a transparent IMI morphology baseline. The supervised increment is now implemented in `urbanpulse/ml.py` and exposed in the Streamlit app.

## End-to-end supervised workflow

```text
Real Sentinel-2 + OSM
        ↓
100 m feature table
        ↓
QGIS reference labeling
        ↓
formal/informal/etc. ground truth
        ↓
spatial-block Random Forest
        ↓
class + calibrated-style model score
        ↓
TreeSHAP explanation
```

The model's `confidence` is the maximum Random Forest class probability. It should be described as a model score/probability output, and calibration should be checked separately before calling it a calibrated probability.

## What the app now does

- Downloads a 100 m labeling template for the current real AOI.
- Accepts labelled GeoJSON/CSV.
- Trains a balanced Random Forest with 500 trees.
- Uses spatial-block cross-validation to reduce spatial leakage.
- Reports macro-F1 and weighted-F1.
- Shows global Random Forest feature importance.
- Saves `models/urbanpulse_rf.joblib` and metadata JSON.
- Can apply the trained RF to the current map.
- Shows TreeSHAP evidence for an inspected cell when SHAP is available.

## Important scientific rule

Do not use `predicted_class`, `informal_probability`, or `confidence` from the heuristic baseline as training labels. The target must come from independent reference labeling.

## Minimum practical training set

For an initial Chennai prototype, collect at least several hundred labelled cells with all important classes represented, and deliberately include hard negatives. More labels are better; the final dataset should be spatially diverse and independently validated.

## Final evaluation

Keep one geographically separated test region untouched until the end. Report per-class precision/recall/F1, macro-F1, confusion matrix and the spatial split definition. Do not report synthetic-demo accuracy as real Chennai accuracy.
