"""Temporal feature utilities for multi-year morphology monitoring."""
import pandas as pd


def wide_year_features(feats, columns, years):
    rows = []
    for cell_id, g in feats.groupby("cell_id"):
        row = {"cell_id": cell_id}
        for _, r in g.iterrows():
            y = int(r.year)
            for c in columns:
                if c in r:
                    row[f"{c}_{y}"] = r[c]
        rows.append(row)
    return pd.DataFrame(rows)
