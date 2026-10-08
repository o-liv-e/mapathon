"""STEP 2 (run on YOUR machine, needs a free Earth Engine account) - annual Sentinel-2 composites.

    pip install earthengine-api
    earthengine authenticate
    python -m urbanpulse.gee_export --project YOUR_GCP_PROJECT

Exports s2_<year>.tif (5 bands: blue, green, red, nir, swir1; reflectance 0-1) to your Google Drive
folder 'urbanpulse', pixel-aligned to the 100 m analysis grid. Download them into data/satellite/.
"""
import argparse
import ee
from pyproj import Transformer
from .config import Config
from .grid import make_grid

BANDS_IN = ["B2", "B3", "B4", "B8", "B11"]
BANDS_OUT = ["blue", "green", "red", "nir", "swir1"]


def mask_s2(img):
    scl = img.select("SCL")
    bad = scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return img.updateMask(bad.Not())


def composite(aoi, year):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70)).map(mask_s2))
    img = col.select(BANDS_IN, BANDS_OUT).median().divide(10000)      # harmonized SR: scale 1e-4
    return img, col.size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    a = ap.parse_args()
    ee.Initialize(project=a.project)
    cfg = Config()
    _, meta = make_grid(cfg)
    x0, ytop, side = meta["x0"], meta["ytop"], meta["side"]
    to_ll = Transformer.from_crs(cfg.crs, "EPSG:4326", always_xy=True)
    ring = [to_ll.transform(x, y) for x, y in
            [(x0, ytop), (x0 + side, ytop), (x0 + side, ytop - side), (x0, ytop - side), (x0, ytop)]]
    aoi = ee.Geometry.Polygon([ring])
    for y in cfg.years:
        img, n = composite(aoi, y)
        print(y, "scenes used:", n.getInfo())            # sanity check: should be dozens
        ee.batch.Export.image.toDrive(
            image=img.toFloat(), description=f"s2_{y}", folder="urbanpulse", fileNamePrefix=f"s2_{y}",
            region=aoi, crs=cfg.crs, crsTransform=[10, 0, x0, 0, -10, ytop], maxPixels=1e9,
            fileFormat="GeoTIFF").start()
    print("Tasks started - watch https://code.earthengine.google.com/tasks then download from Drive.")


if __name__ == "__main__":
    main()
