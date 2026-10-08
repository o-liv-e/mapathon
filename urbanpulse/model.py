"""MODULES 6-7 - stratified sampling, spatial-block validation, Random Forest, temporal smoothing."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict
from .config import CLASSES, CLASS_IDX

META = {"cell_id", "year", "row", "col", "label"}


def feature_columns(df):
    return [c for c in df.columns if c not in META]


def stratified_sample(truth, n_per_class, seed=0):
    """truth: cell_id, year, label -> at most n_per_class rows per class."""
    parts = [g.sample(min(len(g), n_per_class), random_state=seed) for _, g in truth.groupby("label")]
    return pd.concat(parts, ignore_index=True)


def train_model(feats, labels, seed=0, block=4):
    d = feats.merge(labels, on=["cell_id", "year"])
    cols = feature_columns(d)
    y = d.label.map(CLASS_IDX)
    groups = (d.row // block) * 1000 + d.col // block          # spatial blocks -> no leakage
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                class_weight="balanced_subsample", random_state=seed)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    pred = cross_val_predict(rf, d[cols], y, cv=cv, groups=groups)
    present = sorted(y.unique())
    report = classification_report(y, pred, labels=present, target_names=[CLASSES[i] for i in present],
                                   output_dict=True, zero_division=0)
    rf.fit(d[cols], y)
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    return rf, cols, report, imp


def predict_matrix(model, cols, feats, years):
    f = feats.sort_values(["year", "cell_id"])
    proba = model.predict_proba(f[cols])
    cls = model.classes_[proba.argmax(1)]
    n = f.cell_id.nunique()
    M = cls.reshape(len(years), n).T.astype(int)
    C = proba.max(1).reshape(len(years), n).T
    return M, C


def smooth_temporal(M):
    """Remove one-year flicker: A-B-A -> A-A-A."""
    M = M.copy()
    prev, nxt = M[:, :-2].copy(), M[:, 2:].copy()
    flick = (prev == nxt) & (M[:, 1:-1] != prev)
    mid = M[:, 1:-1]
    mid[flick] = prev[flick]
    return M
