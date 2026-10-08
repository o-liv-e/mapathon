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
    # Observable built-form signals. These are morphology indicators, not
    # socioeconomic or legal-status labels.
    out["building_irregularity"] = out.get("footprint_cv", 0.0).fillna(0)
    out["small_building_fraction"] = out.get("small_bldg_frac", 0.0).fillna(0)
    # Closely spaced small footprints are a useful density/texture signal.
    nn = out.get("mean_nn_dist", pd.Series(100.0, index=out.index)).fillna(100.0)
    out["building_spacing_score"] = 1.0 - _minmax(nn)
    # Lower compactness is treated as more irregular footprint geometry.
    compact = out.get("compactness", pd.Series(0.5, index=out.index)).fillna(0.5)
    out["footprint_irregularity"] = 1.0 - compact.clip(0, 1)

    for src, dst in [
        ("building_density", "building_density_norm"),
        ("building_irregularity", "building_irregularity_norm"),
        ("small_building_fraction", "small_building_fraction_norm"),
        ("building_spacing_score", "building_spacing_norm"),
        ("footprint_irregularity", "footprint_irregularity_norm"),
        ("impervious_fraction", "impervious_fraction_norm"),
        ("ndbi", "ndbi_norm"),
    ]:
        out[dst] = _minmax(out[src].fillna(0))
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
    """Continuous 0-1 informal-morphology score using observable features.

    This is intentionally a transparent screening score. It is NOT a validated
    probability of informal housing and must be calibrated against labelled local
    examples before production use.
    """
    out = add_normalized_morphology_features(df)
    components = {
        "building_density_norm": 0.18,
        "small_building_fraction_norm": 0.16,
        "building_irregularity_norm": 0.14,
        "building_spacing_norm": 0.10,
        "footprint_irregularity_norm": 0.10,
        "impervious_fraction_norm": 0.12,
        "ndbi_norm": 0.08,
        "local_building_density_norm": 0.07,
        "local_impervious_fraction_norm": 0.05,
    }
    for c in ["local_building_density", "local_impervious_fraction"]:
        if c in out:
            out[c + "_norm"] = _minmax(out[c].fillna(0))
    terms = [w * out[c].fillna(0) for c, w in components.items() if c in out.columns]
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
