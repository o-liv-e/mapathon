"""Automatic/weak labelling helpers for the UrbanPulse training bootstrap.

These labels are deliberately marked as weak/reference-derived. They are useful to
bootstrap a dataset and reduce manual QGIS work, but they are not a substitute for
an independently verified test set.
"""
from __future__ import annotations

import io
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd

SLUM_LAYER_URL = (
    "https://services2.arcgis.com/I9cUOJUZvdGAJncI/arcgis/rest/services/"
    "Chennai_Slum_Boundaries/FeatureServer/51/query"
)


def fetch_chennai_slums() -> gpd.GeoDataFrame:
    """Fetch the published Chennai Slum Boundaries layer as GeoJSON."""
    params = (
        "where=1%3D1&outFields=OBJECTID%2CName%2CFolderPath&returnGeometry=true"
        "&outSR=4326&f=geojson"
    )
    req = Request(SLUM_LAYER_URL + "?" + params, headers={"User-Agent": "UrbanPulse/1.0"})
    with urlopen(req, timeout=60) as response:
        data = response.read()
    slums = gpd.read_file(io.BytesIO(data))
    if slums.empty:
        raise RuntimeError("The Chennai slum-boundary service returned no polygons.")
    if slums.crs is None:
        slums = slums.set_crs(4326)
    return slums


def _slum_overlap(cells: gpd.GeoDataFrame, slums: gpd.GeoDataFrame) -> pd.Series:
    """Return the maximum slum overlap fraction for each cell."""
    slums = slums.to_crs(cells.crs)
    # Unioning avoids double-counting overlapping source polygons.
    union = slums.geometry.union_all()
    inter = cells.geometry.intersection(union)
    return inter.area / cells.geometry.area


def generate_weak_labels(gdf: gpd.GeoDataFrame, slums: gpd.GeoDataFrame | None = None):
    """Add high-confidence automatic labels plus review suggestions.

    Only the most defensible labels are promoted automatically. Built-up classes
    that cannot be reliably inferred from the current feature set remain
    ``needs_review`` rather than being silently fabricated.
    """
    out = gdf.copy()
    if slums is None:
        slums = fetch_chennai_slums()
    out["slum_overlap_fraction"] = _slum_overlap(out, slums)

    out["auto_label"] = ""
    # This is the field you can review/edit in QGIS. It contains only promoted
    # high-confidence labels; suggested labels remain separate.
    out["label"] = ""
    out["label_status"] = "needs_review"
    out["label_reason"] = "No high-confidence automatic rule."
    out["label_source"] = ""

    water = (
        (out["water_fraction"] >= 0.60)
        & (out["building_density"] < 0.05)
        & (out["vegetation_fraction"] < 0.45)
    )
    vegetation = (
        (out["vegetation_fraction"] >= 0.65)
        & (out["water_fraction"] < 0.20)
        & (out["building_density"] < 0.05)
    )
    sparse = (
        (~out["built_gate"].astype(bool))
        & (out["water_fraction"] < 0.35)
        & (out["vegetation_fraction"] < 0.65)
        & (out["impervious_fraction"] < 0.35)
    )
    informal = out["slum_overlap_fraction"] >= 0.35

    out.loc[water, ["auto_label", "label", "label_status", "label_reason", "label_source"]] = [
        "water", "water", "auto_high_confidence", "Strong Sentinel-2 water signal", "Sentinel-2 spectral rule"
    ]
    out.loc[vegetation, ["auto_label", "label", "label_status", "label_reason", "label_source"]] = [
        "open_vegetated", "open_vegetated", "auto_high_confidence", "Strong vegetation signal with low building density", "Sentinel-2 spectral rule"
    ]
    out.loc[sparse, ["auto_label", "label", "label_status", "label_reason", "label_source"]] = [
        "sparse_low_development", "sparse_low_development", "auto_high_confidence", "Low built-form signal", "Sentinel-2 + OSM morphology rule"
    ]
    out.loc[informal, ["auto_label", "label", "label_status", "label_reason", "label_source"]] = [
        "informal", "informal", "reference_high_confidence", "Cell overlaps a published Chennai slum boundary by >=35%", "Chennai Slum Boundaries (ArcGIS)"
    ]

    # Suggestions are intentionally separate from the training label.
    unresolved = out["auto_label"].eq("")
    suggested = np.full(len(out), "", dtype=object)
    suggested[informal.to_numpy()] = "informal"
    suggested[water.to_numpy()] = "water"
    suggested[vegetation.to_numpy()] = "open_vegetated"
    suggested[sparse.to_numpy()] = "sparse_low_development"

    # Weak built-form suggestions; these must be reviewed before becoming labels.
    built = out["built_gate"].astype(bool).to_numpy()
    highrise = built & (out["building_density"].to_numpy() >= 0.30) & (out["mean_building_area"].to_numpy() >= 350)
    suggested[highrise & unresolved.to_numpy()] = "high_rise"
    remaining = (suggested == "") & built
    # A regular/compact footprint pattern is only a weak suggestion for planned residential.
    planned = remaining & (out["building_density"].to_numpy() >= 0.05) & (out["footprint_cv"].to_numpy() < 1.0)
    suggested[planned] = "planned_residential"
    out["suggested_label"] = suggested

    # Slum cells below the promotion threshold are useful review candidates.
    review_informal = (out["slum_overlap_fraction"] >= 0.10) & (out["slum_overlap_fraction"] < 0.35) & unresolved
    out.loc[review_informal, "suggested_label"] = "informal"
    out.loc[review_informal, "label_reason"] = "Partial overlap with published Chennai slum boundary; review before training."
    out.loc[review_informal, "label_source"] = "Chennai Slum Boundaries (ArcGIS)"

    return out


def training_labels_from_auto(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return only promoted automatic labels in the schema expected by ML."""
    cols = ["cell_uid", "cell_id", "year", "auto_label"]
    d = gdf[[c for c in cols if c in gdf.columns]].copy()
    d = d[d["auto_label"].astype(str).str.len() > 0].copy()
    return d.rename(columns={"auto_label": "label"})
