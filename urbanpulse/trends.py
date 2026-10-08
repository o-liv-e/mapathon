"""MODULES 8-10 - clusters, area statistics, annual trends, transition matrix."""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
from .config import CLASSES, CLASS_IDX

INF = CLASS_IDX["informal"]


def informal_clusters(mask_grid, min_cells):
    lab, n = ndimage.label(mask_grid, structure=np.ones((3, 3)))      # 8-connectivity
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    for i in range(1, n + 1):
        if sizes[i] < min_cells:
            lab[lab == i] = 0
    lab, n = ndimage.label(lab > 0, structure=np.ones((3, 3)))
    return lab, n


def area_by_year(M, years, cs):
    rows = {y: np.bincount(M[:, j], minlength=len(CLASSES)) * cs * cs / 1e6 for j, y in enumerate(years)}
    return pd.DataFrame(rows, index=CLASSES).T.rename_axis("year")


def change_table(area, years):
    y0, y1 = years[0], years[-1]
    t = pd.DataFrame({"area_first_km2": area.loc[y0], "area_last_km2": area.loc[y1]})
    t["abs_change_km2"] = t.area_last_km2 - t.area_first_km2
    t["pct_change"] = np.where(t.area_first_km2 > 0, t.abs_change_km2 / t.area_first_km2 * 100, np.nan)
    t["annual_growth_pct"] = np.where(t.area_first_km2 > 0,
                                      ((t.area_last_km2 / t.area_first_km2) ** (1 / (y1 - y0)) - 1) * 100, np.nan)
    return t


def transition_matrix(M, years, cs):
    a = pd.Series(pd.Categorical.from_codes(M[:, 0], CLASSES), name=str(years[0]))
    b = pd.Series(pd.Categorical.from_codes(M[:, -1], CLASSES), name=str(years[-1]))
    return (pd.crosstab(a, b) * cs * cs / 1e6).reindex(index=CLASSES, columns=CLASSES, fill_value=0)


def trajectory(M):
    first, last = M[:, 0] == INF, M[:, -1] == INF
    return np.select([first & last, ~first & last, first & ~last],
                     ["stable_informal", "new_informal", "lost_informal"], default="other")


def cluster_polygons(cells, lab_flat, extra=None):
    g = cells.assign(cluster_id=lab_flat)
    g = g[g.cluster_id > 0].dissolve(by="cluster_id", aggfunc="size").reset_index()
    g = g.rename(columns={g.columns[-1]: "n_cells"}) if "n_cells" not in g else g
    return g[["cluster_id", "n_cells", "geometry"]]


def cluster_table(cells, M, C, feats_last, years, n, cfg):
    cs, cell_km2 = cfg.cell_size, cfg.cell_size ** 2 / 1e6
    lab, k = informal_clusters((M[:, -1] == INF).reshape(n, n), cfg.min_cluster_cells)
    lab = lab.ravel()
    bcount = feats_last.set_index("cell_id").building_count
    rows = []
    for cid in range(1, k + 1):
        idx = np.where(lab == cid)[0]
        series = (M[idx] == INF).sum(0) * cell_km2
        first = np.nonzero(series)[0]
        rows.append(dict(cluster_id=cid, n_cells=len(idx), area_km2=len(idx) * cell_km2,
                         buildings=int(bcount.iloc[idx].sum()), mean_confidence=float(C[idx, -1].mean()),
                         first_detected=int(years[first[0]]) if len(first) else None,
                         area_change_km2=float(series[-1] - series[0])))
    tab = pd.DataFrame(rows)
    poly = cells.assign(cluster_id=lab)
    poly = poly[poly.cluster_id > 0].dissolve(by="cluster_id").reset_index()[["cluster_id", "geometry"]]
    return tab, poly.merge(tab, on="cluster_id") if len(tab) else poly, lab


def clusters_by_year(cells, M, years, n, cfg):
    out = []
    for j, y in enumerate(years):
        lab, k = informal_clusters((M[:, j] == INF).reshape(n, n), cfg.min_cluster_cells)
        g = cells.assign(cluster_id=lab.ravel())
        g = g[g.cluster_id > 0].dissolve(by="cluster_id").reset_index()[["cluster_id", "geometry"]]
        g["year"] = y
        g["area_km2"] = g.geometry.area / 1e6
        out.append(g)
    return pd.concat(out, ignore_index=True)


def expansion_zone(cells, traj, n, cfg):
    lab, k = ndimage.label((traj == "new_informal").reshape(n, n), structure=np.ones((3, 3)))
    if k == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    cid = int(sizes.argmax()) + 1
    geom = cells[lab.ravel() == cid].union_all()
    c = gpd.GeoSeries([geom.centroid], crs=cells.crs).to_crs(4326).iloc[0]
    return dict(area_km2=float(sizes.max() * cfg.cell_size ** 2 / 1e6), centroid_lon=c.x, centroid_lat=c.y)
