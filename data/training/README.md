# UrbanPulse supervised training dataset

The Random Forest must be trained from **reference labels**, not from the heuristic IMI output.

## Unit

One row/feature record = one **100 m × 100 m UrbanPulse cell** for one year.

## Recommended labels

Use these classes from `urbanpulse/config.py`:

- `informal`
- `planned_residential`
- `high_rise`
- `commercial`
- `industrial`
- `open_vegetated`
- `water`
- `sparse_low_development`

For the hackathon, prioritize a balanced set of Chennai cells and include hard negatives such as dense planned residential areas, high-rise developments and industrial areas.

## Ground truth workflow

1. Run a real UrbanPulse analysis for the target Chennai AOI/year.
2. Download the **100 m labeling template** from the Streamlit ML panel.
3. Open it in QGIS.
4. Inspect high-resolution reference imagery and relevant local reference information.
5. Fill the `label` attribute for cells you can confidently label. Leave uncertain cells blank.
6. Save as GeoJSON.
7. Upload it back into UrbanPulse and train the Random Forest.

The template already contains `cell_id`, `row`, `col`, `year`, `geometry` and an empty `label` field.

## Quality rules

- Do not label a cell from the UrbanPulse IMI itself; that would leak the heuristic into the target.
- Keep a representative number of examples for every class you intend to predict.
- Include difficult look-alikes: formal dense housing, high-rise blocks, industrial areas and old low-rise neighbourhoods.
- Prefer labels supported by high-resolution imagery or authoritative local reference data.
- Keep an independent spatial test area that is not used for training.

## Validation

UrbanPulse uses spatial-block cross-validation. The reported macro-F1 is a spatial generalization estimate, not a random-pixel accuracy.
