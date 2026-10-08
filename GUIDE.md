
## v2 differentiation features

The real pipeline now computes:
- Informal Morphology Index (IMI), 0-1, from observable built-form/spectral/context features.
- Per-cell informal probability and model confidence.
- Spatial-block cross-validation (reported as the main ML metric).
- Multi-year IMI / NDBI / building-density changes.
- Emerging informal-morphology zones. This is change detection, not a socioeconomic forecast.
- `informal_morphology.geojson`, `emerging_zones.geojson`, `uncertain_zones.geojson`, and `temporal_cells.csv`.
- `app.py` provides a Streamlit demo dashboard.

For a competition submission, present the satellite imagery as the primary evidence and buildings/roads as structural context.
