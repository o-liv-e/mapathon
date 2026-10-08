"""Real Sentinel-2 inference for arbitrary AOIs >= 10 km².

This module deliberately keeps processing tile-based: Sentinel-2 data are fetched one 1 km
processing tile at a time and aggregated to the UrbanPulse 100 m analysis grid.
"""
from **future** import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import rasterio
from rasterio.io import MemoryFile
import ee

from .aoi import prepare_aoi, make_aoi_grid, make_tiles
from .features import compute_features
from .morphology import compute_imi, add_spatial_context, add_cell_explanations

BANDS_IN = ["B2", "B3", "B4", "B8", "B11", "B12"]
BANDS_OUT = ["blue", "green", "red", "nir", "swir1", "swir2"]

def initialize_ee(project_id=None, service_account=None, private_key=None):
"""Initialize Earth Engine from Streamlit secrets or environment variables."""
project_id = project_id or os.getenv("GEE_PROJECT_ID")
service_account = service_account or os.getenv("GEE_SERVICE_ACCOUNT")
private_key = private_key or os.getenv("GEE_PRIVATE_KEY")

```
if service_account and private_key:
    creds = ee.ServiceAccountCredentials(
        service_account,
        key_data=private_key.replace("\\n", "\n"),
    )
    ee.Initialize(creds, project=project_id)

elif project_id:
    ee.Initialize(project=project_id)

else:
    raise RuntimeError(
        "Earth Engine credentials are not configured. "
        "Add GEE_PROJECT_ID, GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY "
        "in Streamlit Secrets."
    )
```

def _mask_s2(img):
"""Mask clouds, cloud shadows and other invalid Sentinel-2 pixels."""
scl = img.select("SCL")

```
bad = (
    scl.eq(0)   # No data
    .Or(scl.eq(1))   # Saturated/defective
    .Or(scl.eq(3))   # Cloud shadow
    .Or(scl.eq(8))   # Cloud medium probability
    .Or(scl.eq(9))   # Cloud high probability
    .Or(scl.eq(10))  # Thin cirrus
    .Or(scl.eq(11))  # Snow/ice
)

return img.updateMask(bad.Not())
```

def sentinel_composite(region, year):
"""Create a Sentinel-2 SR Harmonized median composite.

```
The image intentionally retains the original Sentinel-2 band names
(B2, B3, B4, B8, B11, B12). They are renamed only after the raster
has been downloaded/processed, because download_tile() requests the
original Earth Engine band names.
"""

start = f"{year}-01-01"
end = f"{year + 1}-01-01"

# First attempt: reasonably strict cloud filtering.
col = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate(start, end)
    .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 70))
    .map(_mask_s2)
    .select(BANDS_IN)
)

count = col.size().getInfo()

# If the cloud-filtered collection is empty, retry without the
# scene-level cloud percentage filter.
if count == 0:
    print(
        f"No Sentinel-2 scenes passed the cloud filter for {year}; "
        "retrying without the scene-level cloud filter."
    )

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .map(_mask_s2)
        .select(BANDS_IN)
    )

    count = col.size().getInfo()

if count == 0:
    raise RuntimeError(
        f"No usable Sentinel-2 SR Harmonized scenes were found "
        f"for {year} in the selected AOI."
    )

print(f"Sentinel-2 scenes found: {count}")

# IMPORTANT:
# Do NOT rename the bands here. download_tile() expects B2/B3/B4/B8/B11/B12.
image = col.median().select(BANDS_IN)

# Sentinel-2 SR Harmonized reflectance is scaled by 10000.
return image.divide(10000)
```

def download_tile(image, geometry, crs):
"""Download a bounded Sentinel-2 tile.

```
Earth Engine requires an explicit region for getDownloadURL().
The image is clipped to the requested tile and downloaded at 10 m.
"""

# Convert GeoPandas/Shapely geometry to WGS84 GeoJSON coordinates.
if hasattr(geometry, "to_crs"):
    geom_wgs84 = geometry.to_crs("EPSG:4326")
    region = geom_wgs84.__geo_interface__
else:
    region = geometry.__geo_interface__

ee_region = ee.Geometry(region)

# Explicitly select the bands required by the feature engine.
# This also gives a useful failure point if the image is malformed.
bounded = (
    image
    .clip(ee_region)
    .select(BANDS_IN)
)

# Verify that the image actually contains the expected bands before
# requesting the download URL.
band_names = bounded.bandNames().getInfo()

missing = [band for band in BANDS_IN if band not in band_names]

if missing:
    raise RuntimeError(
        "Sentinel-2 image is missing expected bands: "
        f"{missing}. Available bands: {band_names}"
    )

url = bounded.getDownloadURL({
    "name": "urbanpulse_s2",
    "bands": BANDS_IN,
    "region": region,
    "scale": 10,
    "crs": "EPSG:4326",
    "filePerBand": False,
    "format": "GEO_TIFF",
})

response = requests.get(url, timeout=180)
response.raise_for_status()

return response.content
```

def fetch_osm(aoi_ll, crs, include_buildings=True, include_roads=True):
"""Optional OSM structural context. Returns building/road GeoDataFrames."""
import osmnx as ox

```
bld = roads = None

if include_buildings:
    b = ox.features_from_polygon(
        aoi_ll,
        tags={"building": True},
    )
    b = b[
        b.geom_type.isin(["Polygon", "MultiPolygon"])
    ][["geometry"]]
    bld = b.to_crs(crs).reset_index(drop=True)

if include_roads:
    g = ox.graph_from_polygon(
        aoi_ll,
        network_type="all",
    )
    e = ox.graph_to_gdfs(
        g,
        nodes=False,
    ).to_crs(crs)

    e["highway"] = e["highway"].map(
        lambda h: h[0] if isinstance(h, list) else h
    )

    roads = e[
        ["highway", "geometry"]
    ].reset_index(drop=True)

return bld, roads
```

def spectral_feature_frame(cells, tile_bytes, tile_geom):
"""Aggregate a downloaded 10 m Sentinel tile through the existing feature engine."""
with MemoryFile(tile_bytes) as mem:
with mem.open() as src:
# compute_features expects a path; write a temporary file so
# rasterio's window API remains identical.
with tempfile.NamedTemporaryFile(
suffix=".tif",
delete=False,
) as f:
f.write(tile_bytes)
path = f.name

```
try:
    return compute_features(
        cells,
        None,
        None,
        path,
    )
finally:
    Path(path).unlink(missing_ok=True)
```

def run_real_inference(
aoi_ll,
year=2025,
include_osm=True,
progress=None,
):
"""Run a real Sentinel-2 + optional OSM analysis for one year."""

```
aoi_p, area = prepare_aoi(aoi_ll)

cells = make_aoi_grid(
    aoi_p,
    100.0,
)

tiles = make_tiles(
    aoi_p,
    1000.0,
)

initialize_ee()

aoi_ee = ee.Geometry(
    aoi_p
    .to_crs(4326)
    .geometry
    .iloc[0]
    .__geo_interface__
)

image = sentinel_composite(
    aoi_ee,
    year,
)

bld = roads = None

if include_osm:
    bld, roads = fetch_osm(
        aoi_p
        .to_crs(4326)
        .geometry
        .iloc[0],
        aoi_p.crs,
    )

parts = []

total = len(tiles)

for i, row in enumerate(
    tiles.itertuples(index=False),
    1,
):
    tc = cells[
        cells.geometry.intersects(row.geometry)
    ].copy()

    if tc.empty:
        continue

    content = download_tile(
        image,
        row.geometry,
        aoi_p.crs,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".tif",
        delete=False,
    ) as f:
        f.write(content)
        path = f.name

    try:
        p = compute_features(
            tc,
            bld,
            roads,
            path,
            tile_cells=10,
        )
        parts.append(p)

    finally:
        Path(path).unlink(
            missing_ok=True
        )

    if progress:
        progress(i / total)

if not parts:
    raise RuntimeError(
        "No Sentinel-2 processing tiles produced any features. "
        "Check the AOI, year, and Sentinel-2 data availability."
    )

feats = (
    pd.concat(
        parts,
        ignore_index=True,
    )
    .drop_duplicates("cell_id")
)

feats = add_spatial_context(
    feats,
    radius_cells=1,
)

feats = compute_imi(
    feats,
)

# Critical false-positive guard: informal morphology only makes sense
# where there is enough built form.
feats["built_gate"] = (
    (feats["building_count"] >= 8)
    & (feats["building_density"] >= 0.15)
    & (feats["impervious_fraction"] >= 0.25)
    & (
        feats.get(
            "local_building_density",
            feats["building_density"],
        )
        >= 0.10
    )
)

feats["imi"] = feats["imi"].where(
    feats["built_gate"],
    0.0,
)

feats["predicted_class"] = np.select(
    [
        ~feats["built_gate"],
        feats["water_fraction"] > 0.35,
        feats["vegetation_fraction"] > 0.55,
        feats["imi"] >= 0.62,
        feats["imi"] >= 0.45,
    ],
    [
        "sparse_low_development",
        "water",
        "open_vegetated",
        "informal",
        "built_mixed",
    ],
    default="built_planned_like",
)

feats["informal_probability"] = np.where(
    feats["built_gate"],
    feats["imi"],
    0.0,
)

feats["confidence"] = np.clip(
    np.where(
        feats["built_gate"],
        0.55
        \+ 0.45
        * np.abs(feats["imi"] - 0.5)
        * 2,
        0.85,
    ),
    0,
    1,
)

feats = add_cell_explanations(
    feats,
)

out = cells.merge(
    feats,
    on="cell_id",
    how="left",
)

out = gpd.GeoDataFrame(
    out,
    geometry="geometry",
    crs=aoi_p.crs,
)

return out, area, len(tiles)
```
