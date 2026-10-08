"""Explicit spatial block CV helpers."""
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report
from .config import CLASSES


def spatial_groups(df, block_size_cells=4):
    return (df["row"] // block_size_cells) * 100000 + (df["col"] // block_size_cells)


def evaluate_spatial_cv(model, X, y, groups):
    n_groups = int(np.unique(groups).size)
    if n_groups < 2:
        model.fit(X, y)
        pred = model.predict(X)
    else:
        cv = GroupKFold(n_splits=min(5, n_groups))
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=1)
    present = sorted(np.unique(y))
    return classification_report(
        y, pred, labels=present,
        target_names=[CLASSES[i] for i in present],
        output_dict=True, zero_division=0
    )
