"""MODULE 1 - AOI + 100 m grid. Cell ids run row-major from the NW corner."""
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box


def make_grid(cfg):
    cx, cy = Transformer.from_crs("EPSG:4326", cfg.crs, always_xy=True).transform(*cfg.centre_lonlat)
    cs = cfg.cell_size
    n = int(round((cfg.area_km2 * 1e6) ** 0.5 / cs))          # cells per side
    side = n * cs
    x0, ytop = cx - side / 2, cy + side / 2
    rows, cols = np.divmod(np.arange(n * n), n)
    geoms = [box(x0 + c * cs, ytop - (r + 1) * cs, x0 + (c + 1) * cs, ytop - r * cs)
             for r, c in zip(rows, cols)]
    cells = gpd.GeoDataFrame({"cell_id": np.arange(n * n), "row": rows, "col": cols},
                             geometry=geoms, crs=cfg.crs)
    meta = dict(n=n, x0=x0, ytop=ytop, cell=cs, side=side)
    return cells, meta
