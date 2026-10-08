"""MODULES 3-5 - per-cell features: spectral + building morphology + road morphology.
Everything is computed tile-by-tile so the same code scales to a whole city."""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree
from shapely.geometry import box

MAJOR = {"motorway", "trunk", "primary", "secondary"}


def _building_feats(tc, bld, buf=60.0):
    cols = ["building_count", "building_area", "building_density", "mean_building_area",
            "footprint_cv", "compactness", "rectangularity", "small_bldg_frac", "mean_nn_dist"]
    out = pd.DataFrame(0.0, index=tc.cell_id.values, columns=cols)
    out["mean_nn_dist"] = 100.0
    if bld is None or len(bld) == 0:
        return out
    minx, miny, maxx, maxy = tc.total_bounds
    sub = bld.iloc[bld.sindex.query(box(minx - buf, miny - buf, maxx + buf, maxy + buf))]
    if len(sub) == 0:
        return out
    a = sub.geometry.area.values
    p = sub.geometry.length.values
    rect = a / sub.geometry.minimum_rotated_rectangle().area.values
    cen = sub.geometry.centroid
    xy = np.c_[cen.x.values, cen.y.values]
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.full(len(xy), 100.0)
    d = pd.DataFrame({"area": a, "compact": 4 * np.pi * a / p ** 2, "rect": rect,
                      "nn": nn, "small": (a < 50).astype(float)})
    pts = gpd.GeoDataFrame(d, geometry=cen.values, crs=bld.crs)
    j = gpd.sjoin(pts, tc[["cell_id", "geometry"]], predicate="within")
    g = j.groupby("cell_id")
    r = pd.DataFrame({
        "building_count": g.size(), "building_area": g.area.sum(),
        "mean_building_area": g.area.mean(),
        "footprint_cv": (g.area.std() / g.area.mean()).fillna(0),
        "compactness": g.compact.mean(), "rectangularity": g.rect.mean(),
        "small_bldg_frac": g.small.mean(), "mean_nn_dist": g.nn.mean()})
    r["building_density"] = r.building_area / tc.geometry.area.iloc[0]
    out.loc[r.index, r.columns] = r
    return out


def _road_feats(tc, roads, major_union):
    out = pd.DataFrame({"road_density": 0.0, "road_segments": 0.0}, index=tc.cell_id.values)
    if roads is not None and len(roads):
        sub = roads.iloc[roads.sindex.query(box(*tc.total_bounds))]
        if len(sub):
            inter = gpd.overlay(sub[["geometry"]], tc[["cell_id", "geometry"]],
                                how="intersection", keep_geom_type=True)
            g = inter.groupby("cell_id")
            ha = tc.geometry.area.iloc[0] / 1e4
            out.loc[g.size().index, "road_segments"] = g.size()
            out.loc[g.size().index, "road_density"] = g.apply(lambda x: x.geometry.length.sum()) / ha  # m/ha
    out["dist_major_road"] = (tc.geometry.centroid.distance(major_union).values
                              if major_union is not None else 1000.0)
    return out


def _spectral_feats(tc, src):
    minx, miny, maxx, maxy = tc.total_bounds
    win = from_bounds(minx, miny, maxx, maxy, src.transform).round_offsets().round_lengths()
    arr = src.read(window=win, boundless=True, fill_value=np.nan).astype("float32")
    tr = src.window_transform(win)
    ids = rasterize(((g, int(i) + 1) for g, i in zip(tc.geometry, tc.cell_id)),
                    out_shape=arr.shape[1:], transform=tr, fill=0, dtype="int32").ravel()
    b, g_, r, nir, sw = arr.reshape(arr.shape[0], -1)[:5]
    e = 1e-6
    ndvi = (nir - r) / (nir + r + e)
    ndbi = (sw - nir) / (sw + nir + e)
    ndwi = (g_ - nir) / (g_ + nir + e)
    layers = {"blue": b, "green": g_, "red": r, "nir": nir, "swir1": sw,
              "ndvi": ndvi, "ndbi": ndbi, "ndwi": ndwi,
              "vegetation_fraction": (ndvi > 0.4).astype(float),
              "impervious_fraction": ((ndbi > 0) & (ndvi < 0.2)).astype(float),
              "water_fraction": (ndwi > 0.3).astype(float)}
    valid = (ids > 0) & np.isfinite(b)
    idx = ids[valid]
    m = int(tc.cell_id.max()) + 2
    cnt = np.maximum(np.bincount(idx, minlength=m), 1)
    rows = tc.cell_id.values + 1

    def zmean(x):
        return np.bincount(idx, weights=x[valid].astype(float), minlength=m) / cnt

    out = {k: zmean(v)[rows] for k, v in layers.items()}
    for k in ("ndvi", "ndbi", "nir"):                                   # texture proxy
        v = layers[k]
        var = zmean(v ** 2) - zmean(v) ** 2
        out[k + "_std"] = np.sqrt(np.clip(var, 0, None))[rows]
    return pd.DataFrame(out, index=tc.cell_id.values)


def compute_features(cells, bld, roads, raster_path, tile_cells=16):
    major = None
    if roads is not None and len(roads) and "highway" in roads:
        mj = roads[roads.highway.isin(MAJOR)]
        major = mj.union_all() if len(mj) else None
    cells = cells.assign(tile=(cells.row // tile_cells) * 1000 + cells.col // tile_cells)
    parts = []
    with rasterio.open(raster_path) as src:
        for _, tc in cells.groupby("tile"):
            p = pd.concat([_building_feats(tc, bld), _road_feats(tc, roads, major),
                           _spectral_feats(tc, src)], axis=1)
            parts.append(p)
    f = pd.concat(parts).sort_index()
    f["open_space_fraction"] = (1 - f.building_density).clip(0, 1)
    f.index.name = "cell_id"
    return f.reset_index()
