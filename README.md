# UrbanPulse v3

Explainable urban morphology intelligence for mapping informal-neighbourhood morphology.

## What is new in v3

- Upload an arbitrary **GeoJSON AOI** in the Streamlit app.
- Enforces the hackathon minimum of **10 km²**.
- Builds a **100 m analysis grid** and **1 km processing tiles**.
- Uses **Sentinel-2 L2A Surface Reflectance Harmonized** imagery.
- Downloads imagery tile-by-tile instead of loading a city-sized raster.
- Optional OpenStreetMap buildings and roads provide structural morphology.
- Adds a **built-up gate** so sparse/open areas cannot automatically become informal morphology.
- Adds `sparse_low_development` as an explicit output class.
- Real-data mode reports morphology/IMI outputs while clearly separating them from socioeconomic claims.
- Bundled synthetic demo remains available for presentations.

## Real deployment

The Streamlit app needs Earth Engine credentials. Put these in Streamlit Cloud **Settings → Secrets**:

```toml
GEE_PROJECT_ID = "your-project-id"
GEE_SERVICE_ACCOUNT = "service-account@your-project.iam.gserviceaccount.com"
GEE_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
```

Never commit the private key. Local development can use the same secrets file or normal Earth Engine authentication.

## Supervised ML step

The real-data mode starts with a transparent morphology baseline. The supervised ML module is now implemented in `urbanpulse/ml.py` and can be trained from independent reference labels:

1. download the 100 m labeling template from the Streamlit ML panel;
2. label Chennai cells in QGIS using independent high-resolution/reference evidence;
3. upload the labelled GeoJSON/CSV;
4. train the Random Forest with spatial-block cross-validation;
5. inspect macro-F1, per-class metrics and feature importance;
6. apply the trained model to the current AOI;
7. inspect TreeSHAP explanations for individual cells.

The target classes are defined in `urbanpulse/config.py`: informal, planned_residential, high_rise, commercial, industrial, open_vegetated, water and sparse_low_development.

**Important:** the heuristic IMI output is not used as ground truth. Labels must come from independent reference interpretation. Synthetic demo accuracy must not be presented as Chennai accuracy.

### v3.1 Sentinel-2 download fix

The real-data pipeline now explicitly bounds every Earth Engine tile download
with the tile geometry and a 10 m scale. This fixes Earth Engine's
`Image is unbounded` thumbnail/download error and keeps the tile-based
processing architecture intact.
