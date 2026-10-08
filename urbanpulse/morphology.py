"""Urban morphology intelligence: IMI, per-cell explanations and emerging zones."""
import numpy as np
import pandas as pd


def _minmax(s):
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def add_normalized_morphology_features(df):
    out = df.copy()
    # Building heterogeneity is a useful observable morphology signal.
    out["building_irregularity"] = out.get("footprint_cv", 0.0)
    out["road_irregularity"] = 1.0 / (1.0 + out.get("road_segments", 0.0))
    # High road density alone is not a reliable informal signal, so combine it with
    # distance from major roads and neighbourhood context later.
    for src, dst in [
        ("building_density", "building_density_norm"),
        ("building_irregularity", "building_irregularity_norm"),
        ("impervious_fraction", "impervious_fraction_norm"),
        ("ndbi", "ndbi_norm"),
    ]:
        out[dst] = _minmax(out[src].fillna(0))
    out["road_irregularity_norm"] = _minmax(out["road_irregularity"].fillna(0))
    return out


def add_spatial_context(df, radius_cells=1):
    """Add grid-neighbour statistics without assuming row order."""
    out = df.copy()
    if not {"row", "col"}.issubset(out.columns):
        return out
    idx = out.set_index(["row", "col"])
    numeric = [c for c in ["building_density", "ndbi", "ndvi", "impervious_fraction"] if c in out]
    for c in numeric:
        vals = []
        for _, r in out.iterrows():
            key = (int(r.row), int(r.col))
            neighbours = []
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    if dr == 0 and dc == 0:
                        continue
                    k = (key[0] + dr, key[1] + dc)
                    if k in idx.index:
                        v = idx.loc[k, c]
                        if np.isscalar(v):
                            neighbours.append(v)
            vals.append(float(np.mean(neighbours)) if neighbours else float(r[c]))
        out[f"local_{c}"] = vals
    return out


def compute_imi(df):
    """Continuous 0-1 informal morphology score using observable features."""
    out = add_normalized_morphology_features(df)
    # The weights are deliberately transparent and can later be calibrated on labels.
    components = {
        "building_density_norm": 0.22,
        "building_irregularity_norm": 0.18,
        "road_irregularity_norm": 0.12,
        "impervious_fraction_norm": 0.13,
        "ndbi_norm": 0.12,
        "local_building_density": 0.08,
        "local_ndbi": 0.08,
        "local_impervious_fraction": 0.07,
    }
    # Local variables are normalized before use.
    for c in ["local_building_density", "local_ndbi", "local_impervious_fraction"]:
        if c in out:
            out[c + "_norm"] = _minmax(out[c].fillna(0))
    terms = []
    for c, w in components.items():
        cc = c if c in out.columns else c + "_norm"
        if cc in out.columns:
            terms.append(w * out[cc].fillna(0))
    out["imi"] = np.clip(sum(terms), 0, 1)
    return out


def add_cell_explanations(df, informal_probability_col="informal_probability"):
    out = df.copy()
    cols = [
        "building_density_norm", "building_irregularity_norm",
        "road_irregularity_norm", "impervious_fraction_norm", "ndbi_norm",
    ]
    labels = {
        "building_density_norm": "building density",
        "building_irregularity_norm": "building irregularity",
        "road_irregularity_norm": "road structure",
        "impervious_fraction_norm": "impervious surface",
        "ndbi_norm": "built-up intensity",
    }
    available = [c for c in cols if c in out]
    for i, row in out.iterrows():
        vals = sorted(((labels[c], float(row[c])) for c in available), key=lambda x: abs(x[1]), reverse=True)
        out.at[i, "explanation"] = "; ".join(f"{k}: {v:.2f}" for k, v in vals[:5])
    return out


def detect_emerging_zones(df, years):
    """Detect cells whose informal morphology is increasing rapidly.

    This is a change-detection layer, not a claim about future socioeconomic status.
    """
    out = df.copy()
    # Expected columns: imi_<year>, building_density_<year>, ndbi_<year>.
    y0, y1 = years[0], years[-1]
    for base in ["imi", "building_density", "ndbi"]:
        a, b = f"{base}_{y0}", f"{base}_{y1}"
        if a in out and b in out:
            out[f"{base}_change"] = out[b] - out[a]
        else:
            out[f"{base}_change"] = 0.0
    def norm(s):
        return _minmax(s.fillna(0))
    out["emerging_score"] = (
        0.50 * norm(out["imi_change"]) +
        0.30 * norm(out["building_density_change"]) +
        0.20 * norm(out["ndbi_change"])
    )
    out["emerging_zone"] = (out["emerging_score"] >= 0.65).astype(int)
    return out
