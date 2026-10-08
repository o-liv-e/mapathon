"""UrbanPulse - review-ready real-data dashboard.
Run: streamlit run app.py
"""
import json
from pathlib import Path
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UrbanPulse", page_icon="🛰️", layout="wide")
ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

st.title("🛰️ UrbanPulse")
st.caption("Explainable Urban Morphology Intelligence — real Sentinel-2 + geospatial context")


def _style(feature):
    p = feature.get("properties", {})
    cls = str(p.get("predicted_class", ""))
    imi = float(p.get("imi", 0) or 0)
    colors = {
        "informal_morphology_candidate": "#e63946",
        "informal": "#e63946",
        "built_mixed": "#f4a261",
        "built_planned_like": "#2a9d8f",
        "open_vegetated": "#6a994e",
        "water": "#457b9d",
        "sparse_low_development": "#adb5bd",
    }
    return {
        "color": colors.get(cls, "#6c757d"),
        "weight": 1,
        "fillColor": colors.get(cls, "#6c757d"),
        "fillOpacity": 0.18 + 0.52 * min(max(imi, 0), 1),
    }


def show_map(gdf, layer_name="UrbanPulse"):
    import folium
    from streamlit_folium import st_folium

    if gdf is None or gdf.empty:
        st.info("No features to display.")
        return

    gdf = gdf.to_crs(4326)
    c = gdf.geometry.centroid
    center = [float(c.y.mean()), float(c.x.mean())]
    m = folium.Map(location=center, zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Street map", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False, control=True,
    ).add_to(m)

    fields = [
        c for c in [
            "predicted_class", "imi", "informal_probability", "confidence",
            "building_density", "building_irregularity", "road_irregularity",
            "impervious_fraction", "ndvi", "ndbi", "built_gate", "explanation",
        ] if c in gdf.columns
    ]
    aliases = [x.replace("_", " ").title() for x in fields]
    folium.GeoJson(
        gdf.to_json(),
        name=layer_name,
        style_function=_style,
        highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.65},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True),
    ).add_to(m)

    legend = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index:9999;
         background:white; padding:10px 12px; border:1px solid #aaa;
         border-radius:6px; font-size:13px; color:#222">
      <b>UrbanPulse morphology</b><br>
      <span style='color:#e63946'>■</span> Informal morphology candidate<br>
      <span style='color:#f4a261'>■</span> Built mixed<br>
      <span style='color:#2a9d8f'>■</span> Planned-like built<br>
      <span style='color:#6a994e'>■</span> Vegetated<br>
      <span style='color:#457b9d'>■</span> Water<br>
      <span style='color:#adb5bd'>■</span> Sparse
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)
    # Fit precisely to the uploaded AOI/result extent.
    minx, miny, maxx, maxy = gdf.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(10, 10))
    st_folium(m, width=None, height=680, returned_objects=[])


def fingerprint(row):
    st.subheader("Morphological fingerprint")
    names = [
        ("Building density", "building_density"),
        ("Building irregularity", "building_irregularity"),
        ("Road irregularity", "road_irregularity"),
        ("Impervious fraction", "impervious_fraction"),
        ("NDBI", "ndbi"),
        ("Vegetation", "vegetation_fraction"),
    ]
    vals = []
    for label, col in names:
        if col in row.index:
            v = float(row[col]) if pd.notna(row[col]) else 0.0
            vals.append((label, max(0.0, min(1.0, v))))
    for label, v in vals:
        st.progress(v, text=f"{label}: {v:.2f}")
    a, b = st.columns(2)
    a.metric("IMI", f"{float(row.get('imi', 0)):.2f}")
    b.metric("Confidence", f"{float(row.get('confidence', 0))*100:.0f}%")
    st.caption(str(row.get("explanation", "No explanation available.")))



def apply_rf_bundle(gdf, bundle):
    from urbanpulse.ml import predict
    return predict(bundle["model"], bundle["features"], gdf)

def ml_training_panel(gdf, year):
    import streamlit as st
    import pandas as pd
    from urbanpulse.ml import load_labels, prepare_labels, train_models, save_bundle, numeric_feature_columns, predict_grid, explain_cell_prediction

    st.subheader("🤖 Phase A/B — Training dataset + supervised ML")

    uploaded_file = st.file_uploader("Upload labelled cells/polygons", type=["csv", "geojson", "gpkg", "json"])
    
    if uploaded_file is not None:
        labels_df = load_labels(uploaded_file)
        gdf_labeled, labels = prepare_labels(gdf, labels_df)
    else:
        # Fallback to existing labels in gdf if present
        gdf_labeled, labels = gdf, None

    if st.button("🧠 Train Random Forest + XGBoost"):
        with st.spinner("Training models with spatial validation..."):
            try:
                models, cols, evaluations, metadata, training_table = train_models(gdf_labeled, labels)
                
                # Safely extract metadata with defaults
                selected = metadata.get("selected_model", "xgboost")
                n_rows = metadata.get("n_rows", len(gdf_labeled))
                
                st.success(f"Trained {len(models)} models on {n_rows} labelled cells. Selected: {selected}.")
                st.info(f"Selected model: {selected}")

                # Build summary performance table defensively
                rows = []
                if isinstance(evaluations, dict):
                    for name, e in evaluations.items():
                        if isinstance(e, dict):
                            cv_rep = e.get("cv_report") or e.get("report") or {}
                            holdout_data = e.get("holdout") or {}
                            ho_rep = holdout_data.get("report") if isinstance(holdout_data, dict) else {}

                            macro_f1 = cv_rep.get("macro avg", {}).get("f1-score") if isinstance(cv_rep, dict) else None
                            weighted_f1 = cv_rep.get("weighted avg", {}).get("f1-score") if isinstance(cv_rep, dict) else None
                            ho_macro_f1 = ho_rep.get("macro avg", {}).get("f1-score") if isinstance(ho_rep, dict) else None

                            rows.append({
                                "Model": name,
                                "Spatial CV Macro-F1": round(macro_f1, 4) if macro_f1 is not None else None,
                                "Spatial CV Weighted-F1": round(weighted_f1, 4) if weighted_f1 is not None else None,
                                "Spatial Holdout Macro-F1": round(ho_macro_f1, 4) if ho_macro_f1 is not None else None
                            })

                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # Render label breakdown safely if available
                label_counts = metadata.get("label_counts")
                if label_counts:
                    st.write("**Label counts:**", label_counts)

                # --- Inference & SHAP XAI Engine ---
                selected_model_obj = models[selected]
                gdf_predicted = predict_grid(gdf_labeled, selected_model_obj, cols, metadata['label_encoder'])

                st.subheader("🗺️ Inference & Informal Morphology Index (IMI)")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Informal Morphology Index", f"{gdf_predicted['imi'].mean():.2f}")
                with col2:
                    high_imi_count = (gdf_predicted['imi'] > 0.5).sum()
                    st.metric("High-IMI Priority Cells (>0.5)", int(high_imi_count))

                # Interactive Local Feature Explanation (SHAP)
                st.subheader("💡 Explainable AI (XAI) Cell Inspector")
                cell_ids = gdf_predicted['cell_id'].tolist() if 'cell_id' in gdf_predicted.columns else gdf_predicted.index.tolist()
                selected_cell = st.selectbox("Select Cell ID to inspect feature contributions:", cell_ids)

                if selected_cell:
                    cell_data = gdf_predicted[gdf_predicted['cell_id'] == selected_cell] if 'cell_id' in gdf_predicted.columns else gdf_predicted.loc[[selected_cell]]
                    explanation_df = explain_cell_prediction(selected_model_obj, cell_data, cols)
                    
                    st.write(f"**Predicted Class:** `{cell_data['predicted_class'].values[0]}` | **IMI Score:** `{cell_data['imi'].values[0]:.3f}`")
                    st.bar_chart(explanation_df.set_index('feature')['shap_value'])

                # GeoJSON Export
                st.subheader("📥 Export Results")
                geojson_data = gdf_predicted.to_json()
                st.download_button(
                    label="Download Predictions as GeoJSON",
                    data=geojson_data,
                    file_name=f"urbanpulse_predictions_{year}.geojson",
                    mime="application/geo+json"
                )

            except Exception as e:
                st.error(f"ML training failed: {e}")
                
def display_results(gdf, area_km2, n_tiles, year):
    if st.session_state.get("use_rf") and st.session_state.get("rf_bundle") is not None:
        gdf = apply_rf_bundle(gdf, st.session_state["rf_bundle"])
    informal = float(gdf.loc[gdf.predicted_class.isin(["informal", "informal_morphology_candidate"]), "cell_area_m2"].sum() / 1e6)
    built = float(gdf.loc[gdf.built_gate, "cell_area_m2"].sum() / 1e6)
    high_imi = int((gdf["imi"] >= 0.45).sum())
    classes = gdf["predicted_class"].value_counts()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AOI", f"{area_km2:.2f} km²")
    c2.metric("Built-up", f"{built:.2f} km²")
    c3.metric("Informal morphology", f"{informal:.2f} km²")
    c4.metric("High-IMI cells", high_imi)
    c5.metric("Processing tiles", n_tiles)
    st.caption(f"Real Sentinel-2 SR Harmonized morphology baseline for {year}. Red cells are morphology candidates, not verified housing-status labels.")

    show_map(gdf, f"UrbanPulse {year}")

    st.subheader("Classification summary")
    summary = gdf.groupby("predicted_class")["cell_area_m2"].sum().div(1e6).sort_values(ascending=False)
    st.bar_chart(summary.rename("area_km2"))

    st.subheader("Morphological fingerprint")
    if "imi" in gdf.columns:
        choices = gdf.sort_values("imi", ascending=False).head(100)
        labels = [f"{r.cell_id} — {r.predicted_class} — IMI {r.imi:.2f}" for _, r in choices.iterrows()]
        selected = st.selectbox("Inspect a high-information cell", range(len(labels)), format_func=lambda i: labels[i])
        selected_row = choices.iloc[selected]
        fingerprint(selected_row)
        if st.session_state.get("use_rf") and st.session_state.get("rf_bundle") is not None:
            rf_fingerprint(selected_row, st.session_state["rf_bundle"])

    st.subheader("Most informative cells")
    cols = [c for c in [
        "cell_id", "predicted_class", "imi", "confidence", "building_count",
        "building_density", "impervious_fraction", "ndvi", "ndbi", "explanation"
    ] if c in gdf.columns]
    st.dataframe(gdf[cols].sort_values("imi", ascending=False).head(30), use_container_width=True)
    st.download_button("Download real analysis GeoJSON", gdf.to_json(), file_name=f"urbanpulse_{year}.geojson", mime="application/geo+json")


with st.sidebar:
    st.header("Analysis")
    mode = st.radio("Mode", ["Real AOI inference", "Demo / presentation"], index=0)
    st.divider()
    st.markdown("**Review-ready status**")
    st.success("Real Sentinel-2 pipeline connected")
    st.success("100 m morphology grid")
    st.success("Explainable IMI + confidence")
    st.success("OSM structural context")
    st.info("RF + XGBoost + spatial validation + SHAP module ready")

if mode == "Real AOI inference":
    st.subheader("Run UrbanPulse on a real AOI")
    st.info("Upload a GeoJSON polygon of at least 10 km². The application downloads Sentinel-2 surface reflectance tile-by-tile and aggregates it to 100 m cells.")
    upload = st.file_uploader("AOI GeoJSON", type=["geojson", "json"])
    year = st.slider("Analysis year", min_value=2019, max_value=2026, value=2025)
    use_osm = st.checkbox("Use OpenStreetMap buildings + roads", value=True)
    run = st.button("🚀 Run real analysis", type="primary", disabled=upload is None)

    with st.expander("Earth Engine setup"):
        st.write("Required Streamlit secrets: GEE_PROJECT_ID, GEE_SERVICE_ACCOUNT, GEE_PRIVATE_KEY.")
        st.caption("The service account must have Service Usage Consumer and the Earth Engine API must be enabled for the same Google Cloud project.")

    if run and upload is not None:
        try:
            from urbanpulse.aoi import read_aoi
            from urbanpulse.real_pipeline import run_real_inference
            aoi = read_aoi(upload)
            status = st.status("Running UrbanPulse…", expanded=True)
            bar = st.progress(0.0)
            with status:
                st.write("Validating AOI…")
                st.write("Initializing Earth Engine…")
                def progress(x):
                    bar.progress(min(1.0, max(0.0, x)))
                gdf, area, tiles = run_real_inference(aoi, year=year, include_osm=use_osm, progress=progress)
                st.write(f"Processed {tiles} × 1 km tiles.")
            status.update(label="Analysis complete", state="complete")
            st.session_state["real_results"] = {"gdf": gdf, "area": area, "tiles": tiles, "year": year}
        except Exception as e:
            st.error(str(e))
            st.exception(e)

    result = st.session_state.get("real_results")
    if result is not None:
        display_results(result["gdf"], result["area"], result["tiles"], result["year"])
        with st.expander("🤖 Train the supervised Random Forest", expanded=False):
            ml_training_panel(result["gdf"], result["year"])

else:
    st.subheader("🎤 Hackathon review / presentation mode")
    st.warning("This mode uses the bundled demo results for the temporal/cluster story. Do not present demo numbers as real Sentinel-2 findings.")
    stats_path = RESULTS / "stats.json"
    if not stats_path.exists():
        st.error("Bundled demo results are missing.")
        st.stop()
    stats = json.loads(stats_path.read_text())
    h = stats["headline"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Study area", f"{h['study_area_km2']:.2f} km²")
    c2.metric("Informal morphology", f"{h['informal_area_km2']:.2f} km²")
    c3.metric("Clusters", h["informal_clusters"])
    c4.metric("Confidence", f"{h['high_confidence_share']*100:.1f}%")
    c5.metric("Emerging cells", h.get("emerging_cells", 0))
    files = {
        "Informal morphology": RESULTS / "informal_morphology.geojson",
        "Emerging zones": RESULTS / "emerging_zones.geojson",
        "Uncertain zones": RESULTS / "uncertain_zones.geojson",
        "All cells": RESULTS / "cells.geojson",
        "Clusters": RESULTS / f"clusters_{max(stats['area_by_year'].keys())}.geojson",
    }
    import geopandas as gpd
    layer = st.selectbox("Presentation map layer", list(files.keys()))
    if files[layer].exists():
        show_map(gpd.read_file(files[layer]), layer)
    st.subheader("Temporal change")
    area = pd.DataFrame(stats["area_by_year"]).T
    if "informal" in area:
        st.line_chart(area[["informal"]].rename(columns={"informal": "Informal morphology (km²)"}))
    st.info("Pitch: WHERE is the morphology? WHY was it classified? HOW SURE are we? WHAT is changing over time?")

with st.expander("Methodology and interpretation"):
    st.markdown("""
    **UrbanPulse does not identify people, income or legal status.** It maps observable urban morphology.

    **Real-data pipeline:** AOI validation → 1 km processing tiles → Sentinel-2 L2A surface reflectance → spectral indices → optional OSM building/road morphology → 100 m spatial context → built-up gate → morphology classification → IMI and confidence.

    **Current status:** the real-data mode is a transparent morphology inference baseline. Thresholds are starting values and should be calibrated with labelled Chennai cells before claiming production accuracy.

    **Review scope:** real Sentinel-2 inference is operational; temporal/emerging-zone outputs in presentation mode use bundled demo results and are clearly labelled as such.
    """)
