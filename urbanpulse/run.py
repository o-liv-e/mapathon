"""UrbanPulse orchestrator.

Demo:
    python -m urbanpulse.run demo

Real:
    python -m urbanpulse.run real --buildings data/buildings/buildings.parquet \
      --roads data/roads/roads.gpkg --rasters data/satellite \
      --labels data/samples/labels.geojson
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from .config import Config, CLASSES, CLASS_IDX
from .grid import make_grid
from .features import compute_features
from .model import stratified_sample, predict_matrix, smooth_temporal
from .spatial_cv import spatial_groups, evaluate_spatial_cv
from .morphology import compute_imi, add_spatial_context, add_cell_explanations, detect_emerging_zones
from .temporal import wide_year_features
from . import trends as T


def train_model(feats, labels, seed=0):
    d = feats.merge(labels, on=["cell_id", "year"])
    meta = {"cell_id", "year", "row", "col", "label", "explanation"}
    cols = [c for c in d.columns if c not in meta and pd.api.types.is_numeric_dtype(d[c])]
    y = d.label.map(CLASS_IDX)
    groups = spatial_groups(d, block_size_cells=4)
    rf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, n_jobs=-1,
        class_weight="balanced_subsample", random_state=seed
    )
    report = evaluate_spatial_cv(rf, d[cols], y, groups)
    rf.fit(d[cols], y)
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    return rf, cols, report, imp


def add_probabilities(model, cols, feats):
    out = feats.copy()
    proba = model.predict_proba(out[cols])
    classes = model.classes_.astype(int)
    informal_idx = np.where(classes == CLASS_IDX["informal"])[0]
    out["confidence"] = proba.max(axis=1)
    out["informal_probability"] = proba[:, informal_idx[0]] if len(informal_idx) else 0.0
    out["predicted_class_idx"] = classes[proba.argmax(axis=1)]
    return out


def run_pipeline(cfg, cells, meta, bld_by_year, roads, rasters, labels, truth=None):
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    years, n, cs = list(cfg.years), meta["n"], cfg.cell_size
    t0 = time.time()

    # 1. Satellite + buildings + roads at cell level.
    feats = []
    for y in years:
        f = compute_features(cells, bld_by_year[y], roads, rasters[y], cfg.tile_cells)
        f = f.merge(cells[["cell_id", "row", "col"]], on="cell_id")
        f = add_spatial_context(f)
        f = compute_imi(f)
        feats.append(f.assign(year=y))
        print(f"  features {y} done ({time.time() - t0:.0f}s)")
    feats = pd.concat(feats, ignore_index=True)

    # 2. Spatially independent model validation.
    model, cols, report, imp = train_model(feats, labels, cfg.seed)
    print(f"Spatial-block CV macro-F1: {report['macro avg']['f1-score']:.3f}")

    # 3. Predict every year and retain probability + confidence.
    pred_feats = add_probabilities(model, cols, feats)
    raw, C = predict_matrix(model, cols, pred_feats, years)
    M = smooth_temporal(raw)

    # Map smoothed classes back to per-row predictions.
    pred_feats = pred_feats.sort_values(["year", "cell_id"]).reset_index(drop=True)
    pred_feats["class_idx_raw"] = pred_feats["predicted_class_idx"].astype(int)
    pred_feats["class_idx"] = M.T.reshape(-1)
    pred_feats["predicted_class"] = pred_feats["class_idx"].map(lambda k: CLASSES[int(k)])
    pred_feats = add_cell_explanations(pred_feats)

    # 4. Temporal summary and emerging morphology.
    wide = wide_year_features(pred_feats, ["imi", "building_density", "ndbi", "informal_probability"], years)
    wide = detect_emerging_zones(wide, years)

    # 5. Standard spatial outputs.
    area = T.area_by_year(M, years, cs)
    change = T.change_table(area, years)
    trans = T.transition_matrix(M, years, cs)
    traj = T.trajectory(M)
    last = pred_feats[pred_feats.year == years[-1]].sort_values("cell_id")
    ctab, cpoly, lab = T.cluster_table(cells, M, C, last, years, n, cfg)
    cby = T.clusters_by_year(cells, M, years, n, cfg)
    exp = T.expansion_zone(cells, traj, n, cfg)

    # 6. Export cell-level intelligence.
    res = cells.copy()
    for j, y in enumerate(years):
        res[f"class_{y}"] = [CLASSES[k] for k in M[:, j]]
        res[f"conf_{y}"] = C[:, j].round(3)
        py = pred_feats[pred_feats.year == y].sort_values("cell_id")
        res[f"imi_{y}"] = py.imi.round(3).values
        res[f"informal_prob_{y}"] = py.informal_probability.round(3).values
    res["trajectory"] = traj
    res["cluster_last"] = lab
    res = res.merge(wide[["cell_id", "emerging_score", "emerging_zone"]], on="cell_id", how="left")
    for c in ["building_count", "building_density", "road_density", "ndvi", "ndbi", "imi", "informal_probability", "confidence", "explanation"]:
        if c in last:
            res[c] = last[c].values
    res.to_crs(4326).to_file(out / "cells.geojson", driver="GeoJSON")

    cpoly.to_crs(4326).to_file(out / f"clusters_{years[-1]}.geojson", driver="GeoJSON")
    cby.to_crs(4326).to_file(out / "clusters_by_year.geojson", driver="GeoJSON")
    wide.to_csv(out / "temporal_cells.csv", index=False)
    res[res.emerging_zone == 1].to_crs(4326).to_file(out / "emerging_zones.geojson", driver="GeoJSON")
    res[res["class_%s" % years[-1]] == "informal"].to_crs(4326).to_file(out / "informal_morphology.geojson", driver="GeoJSON")
    res[res["conf_%s" % years[-1]] < 0.75].to_crs(4326).to_file(out / "uncertain_zones.geojson", driver="GeoJSON")

    area.to_csv(out / "area_by_year.csv")
    change.to_csv(out / "change.csv")
    trans.to_csv(out / "transitions.csv")

    total = n * n * cs * cs / 1e6
    inf_last = area.loc[years[-1], "informal"]
    head = dict(
        study_area_km2=total,
        informal_area_km2=float(inf_last),
        informal_pct=float(inf_last / total * 100),
        informal_clusters=int(len(ctab)),
        total_buildings=int(len(bld_by_year[years[-1]])),
        buildings_per_km2=float(len(bld_by_year[years[-1]]) / total),
        informal_annual_growth_pct=float(change.loc["informal", "annual_growth_pct"]),
        changed_class_area_km2=float((M[:, 0] != M[:, -1]).sum() * cs * cs / 1e6),
        largest_expansion_zone=exp,
        mean_confidence=float(C[:, -1].mean()),
        high_confidence_share=float((C[:, -1] > .75).mean()),
        mean_imi=float(last.imi.mean()),
        emerging_cells=int(wide.emerging_zone.sum()),
    )
    stats = dict(
        headline=head,
        area_by_year=area.round(4).to_dict("index"),
        change=change.round(4).to_dict("index"),
        cv_report=report,
        top_features=imp.head(15).round(4).to_dict(),
        clusters=ctab.round(4).to_dict("records") if len(ctab) else [],
        methodology={
            "satellite": "Sentinel-2 SR Harmonized annual composites",
            "spatial_unit": f"{cfg.cell_size:.0f}m grid cells",
            "classifier": "Random Forest",
            "validation": "spatial block cross-validation",
            "primary_output": "informal-neighbourhood morphology",
            "imi": "transparent 0-1 Informal Morphology Index",
            "emerging_zone": "temporal morphology change score; not a socioeconomic forecast",
        },
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    if truth is not None:
        tm = truth.pivot(index="cell_id", columns="year", values="label").map(CLASS_IDX.get).values
        print(f"Full-map macro-F1 vs synthetic truth: raw {f1_score(tm.ravel(), raw.ravel(), average='macro'):.3f} | smoothed {f1_score(tm.ravel(), M.ravel(), average='macro'):.3f}")
    print("\nHEADLINE\n", json.dumps(head, indent=2, default=float), f"\nOutputs in {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demo", "real"])
    ap.add_argument("--buildings"); ap.add_argument("--roads"); ap.add_argument("--rasters"); ap.add_argument("--labels"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = Config(out_dir=a.out)
    cells, meta = make_grid(cfg)
    if a.mode == "demo":
        from .synthetic import make_synthetic_city
        city = make_synthetic_city(cells, meta, cfg, Path(cfg.out_dir) / "synthetic")
        labels = stratified_sample(city.truth, n_per_class=120, seed=cfg.seed)
        run_pipeline(cfg, cells, meta, city.buildings, city.roads, city.rasters, labels, city.truth)
    else:
        from .sources import labels_from_polygons
        b = gpd.read_parquet(a.buildings) if a.buildings.endswith("parquet") else gpd.read_file(a.buildings)
        b = b.to_crs(cfg.crs)
        roads = gpd.read_file(a.roads).to_crs(cfg.crs)
        rasters = {y: str(Path(a.rasters) / f"s2_{y}.tif") for y in cfg.years}
        run_pipeline(cfg, cells, meta, {y: b for y in cfg.years}, roads, rasters, labels_from_polygons(a.labels, cells))

if __name__ == "__main__":
    main()
