# UrbanPulse supervised ML deployment

The production bundle is `models/urbanpulse_production.joblib`.

It contains:
- Random Forest and XGBoost models
- the selected model (chosen by spatially grouped macro-F1)
- the label encoder
- the 38 numeric morphology/spectral feature names
- training metadata

The model is trained from 371 promoted weak/reference labels for the 2025 AOI:
- 67 informal
- 81 open_vegetated
- 223 sparse_low_development

**Important:** these are weak/reference labels, so the spatial-CV scores are a proof-of-concept and must not be presented as independently verified ground-truth accuracy.

## Reproduce training

```bash
python scripts/train_production_ml.py
```

## Scale to a new AOI

The real pipeline only needs to produce the same feature columns. After Sentinel-2/OSM feature extraction, `display_results()` automatically loads the production bundle and applies it to every 100 m cell. No retraining is required for each AOI.

## Explain a cell

TreeSHAP is computed on demand for the selected cell. This avoids precomputing explanations for an entire city and keeps inference scalable.
