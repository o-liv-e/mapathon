# UrbanPulse v2

**Explainable urban morphology intelligence for informal-neighbourhood mapping.**

UrbanPulse keeps satellite imagery at the centre of the solution while adding building morphology, road structure, spatial context, uncertainty, explainability and temporal change detection.

## What is new in v2

- Sentinel-2-first feature pipeline
- Building and road morphology features
- Local neighbourhood context
- Spatial-block cross-validation
- Per-cell confidence and informal probability
- **Informal Morphology Index (IMI)**, a transparent 0-1 morphology score
- Per-cell explanation text for dashboard inspection
- Temporal morphology table
- **Emerging informal-morphology zones** based on observed change
- GeoJSON layers for informal, emerging and uncertain areas
- Streamlit demo dashboard
- Docker deployment

## Run the synthetic demo

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m urbanpulse.run demo
streamlit run app.py
```

Synthetic results are for pipeline testing only. They are **not Chennai findings**.

## Run with real data

See `GUIDE.md` for Earth Engine, building footprints, roads and labeling.

```powershell
python -m urbanpulse.run real --buildings data/buildings/buildings.parquet --roads data/roads/roads.gpkg --rasters data/satellite --labels data/samples/labels.geojson
```

## Outputs

- `results/cells.geojson` — cell-level class, confidence, IMI and emerging score
- `results/informal_morphology.geojson` — informal morphology cells
- `results/emerging_zones.geojson` — cells with strong positive morphology change
- `results/uncertain_zones.geojson` — low-confidence cells
- `results/clusters_YYYY.geojson` — current informal clusters
- `results/clusters_by_year.geojson` — annual clusters
- `results/temporal_cells.csv` — multi-year feature history
- `results/stats.json` — metrics, validation and methodology

## Interpretation

The model identifies **observable informal-type urban morphology**. It does not infer residents' income, identity or socioeconomic status from satellite imagery.
