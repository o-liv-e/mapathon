"""Synthetic Chennai-like city so the whole pipeline can be tested without any downloads.
NOT real data - accuracy numbers on it are optimistic. Informal areas expand, some get redeveloped."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import Polygon, LineString
from .config import CLASSES, CLASS_IDX

INF, PLAN, HIGH, COM, IND, OPEN, WATER = range(7)
# (count range, footprint area range m2, irregular?)
P = {INF: ((60, 110), (15, 45), True), PLAN: ((8, 14), (90, 180), False),
     HIGH: ((2, 4), (500, 1400), False), COM: ((3, 6), (300, 1000), False),
     IND: ((1, 2), (2500, 5500), False), OPEN: ((0, 2), (10, 40), False), WATER: ((0, 0), (1, 1), False)}
SPEC = np.array([[.10, .12, .14, .22, .28], [.08, .10, .11, .28, .26], [.12, .14, .16, .20, .30],
                 [.13, .15, .17, .21, .31], [.14, .16, .18, .22, .33], [.04, .07, .05, .40, .18],
                 [.05, .06, .04, .02, .01]])


def _buildings(k, x, y, cs, rng):
    (c0, c1), (a0, a1), irr = P[k]
    out = []
    for _ in range(rng.integers(c0, c1 + 1)):
        a = rng.uniform(a0, a1)
        asp = rng.uniform(.5, 1.8) if irr else rng.uniform(1, 1.4)
        w = np.sqrt(a * asp); h = a / w
        th = rng.uniform(0, np.pi) if irr else 0.0
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
        out.append(Polygon(corners @ R.T + [x + rng.uniform(4, cs - 4), y + rng.uniform(4, cs - 4)]))
    return out


def _roads(k, x, y, cs, rng):
    L = []
    if k == INF:
        for _ in range(rng.integers(5, 9)):
            p = np.array([x + rng.uniform(0, cs), y + rng.uniform(0, cs)])
            t, ln = rng.uniform(0, 2 * np.pi), rng.uniform(25, 50)
            L.append(LineString([p, p + ln * np.array([np.cos(t), np.sin(t)])]))
    elif k == PLAN:
        L += [LineString([(x, y + cs / 2), (x + cs, y + cs / 2)]), LineString([(x + cs / 2, y), (x + cs / 2, y + cs)])]
    elif k in (HIGH, COM, IND):
        L.append(LineString([(x, y + cs / 2), (x + cs, y + cs / 2)]))
    return L


def make_synthetic_city(cells, meta, cfg, out_dir):
    rng = np.random.default_rng(cfg.seed)
    n, cs, x0, ytop = meta["n"], meta["cell"], meta["x0"], meta["ytop"]
    years = list(cfg.years)
    # --- land-use map at start year: Voronoi blobs + a river
    ns = 26
    sr, sc = rng.uniform(0, n, ns), rng.uniform(0, n, ns)
    sk = rng.choice(6, ns, p=[.28, .20, .08, .10, .08, .26])
    rr, cc = np.mgrid[0:n, 0:n]
    base = sk[((rr[..., None] - sr) ** 2 + (cc[..., None] - sc) ** 2).argmin(-1)]
    base[np.abs(cc - (0.65 * n + 3 * np.sin(rr / 5))) < 1.2] = WATER
    maps = [base]
    for _ in years[1:]:
        prev = maps[-1]; m = prev.copy(); r = rng.random((n, n))
        nb = lambda k: ndimage.convolve((prev == k).astype(float), np.ones((3, 3)), mode="constant") - (prev == k)
        m[(nb(INF) > 0) & (prev == OPEN) & (r < .18)] = INF                      # informal growth
        red = (prev == INF) & (nb(COM) + nb(HIGH) > 0) & (r < .05)               # redevelopment
        m[red] = rng.choice([COM, HIGH], size=red.sum())
        m[(prev == PLAN) & (nb(COM) > 0) & (r < .03)] = COM
        maps.append(m)
    flat = [m.ravel() for m in maps]
    truth = pd.DataFrame([(cid, y, CLASSES[flat[j][cid]]) for j, y in enumerate(years) for cid in range(n * n)],
                         columns=["cell_id", "year", "label"])
    # --- buildings (cached until a cell changes class; 10% of cells look like another class)
    cache, bld = {}, {}
    for j, y in enumerate(years):
        geoms = []
        for cid in range(n * n):
            k = flat[j][cid]
            if cid not in cache or cache[cid][0] != k:
                gk = k if rng.random() > .10 else rng.integers(0, 6)
                cache[cid] = (k, _buildings(gk, x0 + (cid % n) * cs, ytop - (cid // n + 1) * cs, cs, rng))
            geoms += cache[cid][1]
        bld[y] = gpd.GeoDataFrame(geometry=geoms, crs=cfg.crs)
    # --- roads (static) + two major roads
    lines = [l for cid in range(n * n) for l in _roads(flat[0][cid], x0 + (cid % n) * cs, ytop - (cid // n + 1) * cs, cs, rng)]
    major = [LineString([(x0, ytop - .4 * n * cs - 17), (x0 + n * cs, ytop - .4 * n * cs - 17)]),
             LineString([(x0 + .3 * n * cs + 13, ytop), (x0 + .3 * n * cs + 13, ytop - n * cs)])]
    roads = gpd.GeoDataFrame({"highway": ["residential"] * len(lines) + ["primary"] * 2},
                             geometry=lines + major, crs=cfg.crs)
    # --- Sentinel-like rasters, 10 m pixels, 5 bands
    out_dir.mkdir(parents=True, exist_ok=True)
    rasters, H = {}, n * 10
    for j, y in enumerate(years):
        px = np.kron(maps[j], np.ones((10, 10), int))
        off = np.kron(rng.normal(0, .01, (n, n)), np.ones((10, 10)))
        arr = np.clip(SPEC[px] + off[..., None] + rng.normal(0, .025, (H, H, 5)), .001, 1)
        p = out_dir / f"s2_{y}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=H, width=H, count=5, dtype="float32",
                           crs=cfg.crs, transform=from_origin(x0, ytop, 10, 10)) as dst:
            dst.write(np.moveaxis(arr, -1, 0).astype("float32"))
        rasters[y] = str(p)
    return SimpleNamespace(truth=truth, buildings=bld, roads=roads, rasters=rasters)
