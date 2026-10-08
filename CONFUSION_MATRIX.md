# XGBoost spatial-validation confusion matrix

| Actual \ Predicted | informal | open_vegetated | sparse_low_development |
|---|---:|---:|---:|
| **informal** | **53** | 1 | 13 |
| **open_vegetated** | 0 | **80** | 1 |
| **sparse_low_development** | 6 | 0 | **217** |

This is the current out-of-fold spatial-validation confusion matrix for the
production XGBoost run.

The evaluation uses promoted weak/reference labels, not independently verified
ground truth.
