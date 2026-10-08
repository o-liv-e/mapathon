from dataclasses import dataclass

CLASSES = ["informal", "planned_residential", "high_rise", "commercial",
           "industrial", "open_vegetated", "water"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
BANDS = ["blue", "green", "red", "nir", "swir1"]   # band order of every raster


@dataclass
class Config:
    centre_lonlat: tuple = (80.2707, 13.0827)   # EDIT: centre of your AOI (Chennai default)
    area_km2: float = 10.0
    crs: str = "EPSG:32644"                     # UTM 44N, correct for Chennai
    cell_size: float = 100.0                    # metres
    years: tuple = tuple(range(2019, 2027))
    tile_cells: int = 16                        # tile edge in cells (scaling unit)
    min_cluster_cells: int = 3
    out_dir: str = "results"
    seed: int = 42
