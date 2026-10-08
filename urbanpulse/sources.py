"""STEPS 3-5 - building footprints, roads, label conversion. Run on YOUR machine (needs internet).
    python -m urbanpulse.sources buildings-osm     # quick option (OpenStreetMap)
    python -m urbanpulse.sources buildings-ms      # Microsoft Global ML Building Footprints
    python -m urbanpulse.sources roads
"""
import sys
import geopandas as gpd
from .config import Config
from .grid import make_grid


def _aoi_ll(cfg):
    cells, _ = make_grid(cfg)
    return cells.to_crs(4326).union_all(), cells


def buildings_osm(cfg, out="data/buildings/buildings.parquet"):
    import osmnx as ox
    aoi, cells = _aoi_ll(cfg)
    g = ox.features_from_polygon(aoi, tags={"building": True})
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]].to_crs(cfg.crs).reset_index(drop=True)
    g.to_parquet(out); print(len(g), "buildings ->", out)


def buildings_ms(cfg, out="data/buildings/buildings.parquet"):
    """Microsoft footprints are split into quadkey tiles listed in dataset-links.csv."""
    import io, gzip, json, mercantile, requests, pandas as pd
    from shapely.geometry import shape
    aoi, cells = _aoi_ll(cfg)
    links = pd.read_csv("https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv")
    links = links[links.Location == "India"]
    minx, miny, maxx, maxy = aoi.bounds
    qks = {mercantile.quadkey(t) for t in mercantile.tiles(minx, miny, maxx, maxy, 9)}
    sel = links[links.QuadKey.astype(str).apply(lambda q: any(q.startswith(k) or k.startswith(q) for k in qks))]
    feats = []
    for url in sel.Url:
        txt = gzip.decompress(requests.get(url, timeout=300).content).decode()
        feats += [shape(json.loads(l)["geometry"]) for l in txt.splitlines() if l.strip()]
    g = gpd.GeoDataFrame(geometry=feats, crs=4326)
    g = g[g.intersects(aoi)].to_crs(cfg.crs).reset_index(drop=True)
    g.to_parquet(out); print(len(g), "buildings ->", out)


def roads(cfg, out="data/roads/roads.gpkg"):
    import osmnx as ox
    aoi, _ = _aoi_ll(cfg)
    e = ox.graph_to_gdfs(ox.graph_from_polygon(aoi, network_type="all"), nodes=False).to_crs(cfg.crs)
    e["highway"] = e["highway"].map(lambda h: h[0] if isinstance(h, list) else h)
    e[["highway", "geometry"]].reset_index(drop=True).to_file(out, driver="GPKG"); print(len(e), "road segments ->", out)


def labels_from_polygons(path, cells):
    """labels.geojson polygons with columns: class (see config.CLASSES), year -> (cell_id, year, label)."""
    polys = gpd.read_file(path).to_crs(cells.crs)
    cen = cells[["cell_id"]].assign(geometry=cells.centroid)
    j = gpd.sjoin(gpd.GeoDataFrame(cen, crs=cells.crs), polys[["class", "year", "geometry"]], predicate="within")
    return j[["cell_id", "year", "class"]].rename(columns={"class": "label"}).reset_index(drop=True)


if __name__ == "__main__":
    {"buildings-osm": buildings_osm, "buildings-ms": buildings_ms, "roads": roads}[sys.argv[1]](Config())
