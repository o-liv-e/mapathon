# UrbanPulse — UI patch

## Add the confusion matrix + limitations

1. Copy `urbanpulse/ui_additions.py` into your existing repository's:
   `urbanpulse/` folder.

2. In `app.py`, add this import near the other imports:

```python
from urbanpulse.ui_additions import render_validation_and_limitations
```

3. Add this call after the production model / validation metrics section and
   before the cell-level XAI section:

```python
render_validation_and_limitations()
```

4. Commit and push to GitHub. Streamlit Community Cloud will rebuild the app.

## Current XGBoost spatial-validation matrix

Rows = actual/reference label
Columns = predicted label

| Actual \ Predicted | informal | open_vegetated | sparse_low_development |
|---|---:|---:|---:|
| informal | 53 | 1 | 13 |
| open_vegetated | 0 | 80 | 1 |
| sparse_low_development | 6 | 0 | 217 |

These values correspond to the current production training run.

## Important scientific wording

Do not describe these results as independently verified ground-truth accuracy.
The training data are promoted weak/reference labels.

Recommended wording:
> XGBoost achieved a spatial cross-validation Macro-F1 of 0.9313 on our
> promoted weak/reference-labelled dataset.

