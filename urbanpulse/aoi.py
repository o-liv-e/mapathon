"""AOI validation and scalable 100 m analysis grid."""
import json
from io import BytesIO
import geopandas as gpd
import numpy as np
from shapely.geometry import box

MIN_AOI_KM2 = 10.0


def read_aoi(uploaded_file):
    """Read a GeoJSON/Shapefile-like upload. Streamlit uploader returns bytes."""
    name = getattr(uploaded_file, "name", "aoi.geojson").lower()
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    if name.endswith(".geojson") or name.endswith(".json"):
        obj = json.loads(raw.decode("utf-8"))
        gdf = gpd.GeoDataFrame.from_features(obj["features"], crs=obj.get("crs", "EPSG:4326"))
    else:
        # GeoPandas cannot reliably read a zipped shapefile from raw bytes without a filesystem.
        raise ValueError("Please upload GeoJSON for the web app. Shapefile/GeoPackage support is available in the CLI.")
    if gdf.empty or gdf.geometry.is_empty.all():
        raise ValueError("AOI contains no valid geometry.")
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    geom = gdf.geometry.union_all()
    return gpd.GeoDataFrame({"geometry": [geom]}, crs=4326)


def prepare_aoi(gdf, min_km2=MIN_AOI_KM2):
    """Return a projected single-polygon AOI and its area."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    metric_crs = gdf.estimate_utm_crs()
    p = gdf.to_crs(metric_crs)
    geom = p.geometry.union_all()
    area = geom.area / 1e6
    if area < min_km2:
        raise ValueError(f"AOI is {area:.2f} km². The minimum study area is {min_km2:.0f} km².")
    return gpd.GeoDataFrame({"geometry": [geom]}, crs=metric_crs), area


def make_aoi_grid(aoi_projected, cell_size=100.0):
    geom = aoi_projected.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    x0 = np.floor(minx / cell_size) * cell_size
    y0 = np.floor(miny / cell_size) * cell_size
    x1 = np.ceil(maxx / cell_size) * cell_size
    y1 = np.ceil(maxy / cell_size) * cell_size
    xs = np.arange(x0, x1, cell_size)
    ys = np.arange(y0, y1, cell_size)
    rows = []
    geoms = []
    for r, y in enumerate(ys[::-1]):
        for c, x in enumerate(xs):
            g = box(x, y, x + cell_size, y + cell_size)
            if g.intersects(geom):
                rows.append((len(rows), r, c))
                geoms.append(g.intersection(geom))
    cells = gpd.GeoDataFrame(rows, columns=["cell_id", "row", "col"], geometry=geoms, crs=aoi_projected.crs)
    cells["cell_area_m2"] = cells.geometry.area
    cells = cells[cells.cell_area_m2 >= cell_size * cell_size * 0.25].reset_index(drop=True)
    cells["cell_id"] = np.arange(len(cells))
    return cells


def make_tiles(aoi_projected, tile_size=1000.0):
    geom = aoi_projected.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    x0 = np.floor(minx / tile_size) * tile_size
    y0 = np.floor(miny / tile_size) * tile_size
    tiles = []
    i = 0
    x = x0
    while x < maxx:
        y = np.floor(miny / tile_size) * tile_size
        while y < maxy:
            t = box(x, y, x + tile_size, y + tile_size)
            if t.intersects(geom):
                tiles.append((f"t{i:04d}", t.intersection(geom)))
                i += 1
            y += tile_size
        x += tile_size
    return gpd.GeoDataFrame({"tile_id": [x[0] for x in tiles], "geometry": [x[1] for x in tiles]}, crs=aoi_projected.crs)
