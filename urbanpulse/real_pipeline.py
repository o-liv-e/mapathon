"""Real Sentinel-2 inference for arbitrary AOIs >= 10 km².

This module deliberately keeps processing tile-based: Sentinel-2 data are fetched one 1 km
processing tile at a time and aggregated to the UrbanPulse 100 m analysis grid.
"""
from __future__ import annotations
import io, json, os, tempfile
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
    if service_account and private_key:
        creds = ee.ServiceAccountCredentials(service_account, key_data=private_key.replace("\\n", "\n"))
        ee.Initialize(creds, project=project_id)
    elif project_id:
        ee.Initialize(project=project_id)
    else:
        raise RuntimeError("Earth Engine credentials are not configured. Add GEE_PROJECT_ID, GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY in Streamlit Secrets.")


def _mask_s2(img):
    scl = img.select("SCL")
    bad = scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return img.updateMask(bad.Not())


def sentinel_composite(region, year):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region)
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
           .map(_mask_s2))
    if col.size().getInfo() == 0:
        raise RuntimeError(f"No usable Sentinel-2 scenes found for {year} in the selected AOI.")
    # Keep the native Sentinel-2 band names. download_tile() requests these
    # bands directly, and compute_features() consumes the downloaded bands
    # by position. Renaming them here would make the later B2/B3/... request
    # fail with: Image has no band named "B2".
    image = col.select(BANDS_IN).median().divide(10000)

    available = image.bandNames().getInfo()
    missing = [band for band in BANDS_IN if band not in available]
    if missing:
        raise RuntimeError(
            f"Sentinel-2 composite is missing bands: {missing}. "
            f"Available bands: {available}"
        )

    print(f"Sentinel-2 composite bands: {available}")
    return image


def download_tile(image, geometry, crs):
    """Download a bounded Sentinel-2 tile at 10 m resolution."""
    if hasattr(geometry, "to_crs"):
        geometry = geometry.to_crs("EPSG:4326")

    region = geometry.__geo_interface__
    ee_region = ee.Geometry(region)

    # Select the native Sentinel-2 bands on the image itself. Do not pass a
    # second ``bands`` selection to getDownloadURL(); the previous version
    # could cause Earth Engine to resolve B2 against an image whose bands had
    # already been renamed.
    bounded = image.clip(ee_region).select(BANDS_IN)

    available = bounded.bandNames().getInfo()
    missing = [band for band in BANDS_IN if band not in available]
    if missing:
        raise RuntimeError(
            f"Cannot download Sentinel-2 tile. Missing bands: {missing}. "
            f"Available bands: {available}"
        )

    print(f"Download image bands: {available}")

    url = bounded.getDownloadURL({
        "name": "urbanpulse_s2",
        "region": region,
        "scale": 10,
        "crs": "EPSG:4326",
        "filePerBand": False,
        "format": "GEO_TIFF",
    })

    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return response.content

def fetch_osm(aoi_ll, crs, include_buildings=True, include_roads=True):
    """Optional OSM structural context. Returns building/road GeoDataFrames."""
    import osmnx as ox
    bld = roads = None
    if include_buildings:
        b = ox.features_from_polygon(aoi_ll, tags={"building": True})
        b = b[b.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]]
        bld = b.to_crs(crs).reset_index(drop=True)
    if include_roads:
        g = ox.graph_from_polygon(aoi_ll, network_type="all")
        e = ox.graph_to_gdfs(g, nodes=False).to_crs(crs)
        e["highway"] = e["highway"].map(lambda h: h[0] if isinstance(h, list) else h)
        roads = e[["highway", "geometry"]].reset_index(drop=True)
    return bld, roads


def spectral_feature_frame(cells, tile_bytes, tile_geom):
    """Aggregate a downloaded 10 m Sentinel tile through the existing feature engine."""
    with MemoryFile(tile_bytes) as mem:
        with mem.open() as src:
            # compute_features expects a path; write a temporary file so rasterio's window API remains identical.
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
                f.write(tile_bytes)
                path = f.name
    try:
        return compute_features(cells, None, None, path)
    finally:
        Path(path).unlink(missing_ok=True)


def run_real_inference(aoi_ll, year=2025, include_osm=True, progress=None):
    """Run a real Sentinel-2 + optional OSM analysis for one year."""
    aoi_p, area = prepare_aoi(aoi_ll)
    cells = make_aoi_grid(aoi_p, 100.0)
    tiles = make_tiles(aoi_p, 1000.0)
    initialize_ee()
    aoi_ee = ee.Geometry(aoi_p.to_crs(4326).geometry.iloc[0].__geo_interface__)
    image = sentinel_composite(aoi_ee, year)

    bld = roads = None
    if include_osm:
        bld, roads = fetch_osm(aoi_p.to_crs(4326).geometry.iloc[0], aoi_p.crs)

    parts = []
    total = len(tiles)
    for i, row in enumerate(tiles.itertuples(index=False), 1):
        tc = cells[cells.geometry.intersects(row.geometry)].copy()
        if tc.empty:
            continue
        content = download_tile(image, row.geometry, aoi_p.crs)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            f.write(content); path = f.name
        try:
            p = compute_features(tc, bld, roads, path, tile_cells=10)
            parts.append(p)
        finally:
            Path(path).unlink(missing_ok=True)
        if progress:
            progress(i / total)

    feats = pd.concat(parts, ignore_index=True).drop_duplicates("cell_id")
    feats = add_spatial_context(feats, radius_cells=1)
    feats = compute_imi(feats)

    # Critical false-positive guard: informal morphology only makes sense where there is enough built form.
    feats["built_gate"] = (
        (feats["building_count"] >= 8) &
        (feats["building_density"] >= 0.15) &
        (feats["impervious_fraction"] >= 0.25) &
        (feats.get("local_building_density", feats["building_density"]) >= 0.10)
    )
    feats["imi"] = feats["imi"].where(feats["built_gate"], 0.0)
    feats["predicted_class"] = np.select(
        [~feats["built_gate"], feats["water_fraction"] > 0.35, feats["vegetation_fraction"] > 0.55,
         feats["imi"] >= 0.62, feats["imi"] >= 0.45],
        ["sparse_low_development", "water", "open_vegetated", "informal", "built_mixed"],
        default="built_planned_like",
    )
    feats["informal_probability"] = np.where(feats["built_gate"], feats["imi"], 0.0)
    feats["confidence"] = np.clip(np.where(feats["built_gate"], 0.55 + 0.45 * np.abs(feats["imi"] - 0.5) * 2, 0.85), 0, 1)
    feats = add_cell_explanations(feats)
    out = cells.merge(feats, on="cell_id", how="left")
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=aoi_p.crs)
    return out, area, len(tiles)
