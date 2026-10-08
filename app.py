"""UrbanPulse v3 — demo dashboard + real AOI inference.
Run: streamlit run app.py
"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UrbanPulse", page_icon="🛰️", layout="wide")
ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

st.title("🛰️ UrbanPulse")
st.caption("Explainable urban morphology intelligence for informal-neighbourhood mapping")


def show_map(gdf, layer_name="UrbanPulse"):
    import folium
    import streamlit.components.v1 as components
    gdf = gdf.to_crs(4326)
    if gdf.empty:
        st.info("No features to display.")
        return
    c = gdf.geometry.centroid
    center = [c.y.mean(), c.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street map", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer("OpenTopoMap", name="Terrain", control=True).add_to(m)
    fields = [c for c in ["predicted_class", "imi", "informal_probability", "confidence", "built_gate", "explanation"] if c in gdf.columns]
    folium.GeoJson(gdf.to_json(), name=layer_name,
                   tooltip=folium.GeoJsonTooltip(fields=fields, aliases=[x.replace("_", " ").title() for x in fields])).add_to(m)
    folium.LayerControl().add_to(m)
    # Render as a self-contained HTML iframe. This avoids the Streamlit-Folium
    # widget rerun/blank-map issue and keeps the map visible after analysis.
    components.html(m.get_root().render(), height=700, scrolling=True)


def display_results(gdf, area_km2, n_tiles, title="Real analysis"):
    informal = float(gdf.loc[gdf.predicted_class == "informal", "cell_area_m2"].sum() / 1e6)
    built = float(gdf.loc[gdf.built_gate, "cell_area_m2"].sum() / 1e6)
    emerging = int((gdf.imi >= 0.62).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AOI", f"{area_km2:.2f} km²")
    c2.metric("Informal morphology", f"{informal:.2f} km²")
    c3.metric("Built-up area", f"{built:.2f} km²")
    c4.metric("Processing tiles", n_tiles)
    st.caption("These are observable morphology outputs for the selected AOI/year — not socioeconomic estimates.")
    show_map(gdf, title)
    st.subheader("Morphology summary")
    summary = gdf.groupby("predicted_class", dropna=False)["cell_area_m2"].sum().div(1e6).sort_values(ascending=False)
    st.bar_chart(summary.rename("area_km2"))
    st.subheader("Most informative cells")
    cols = [c for c in ["cell_id", "predicted_class", "imi", "confidence", "building_count", "building_density", "impervious_fraction", "ndvi", "ndbi", "explanation"] if c in gdf.columns]
    st.dataframe(gdf[cols].sort_values("imi", ascending=False).head(30), use_container_width=True)
    st.download_button("Download GeoJSON", gdf.to_json(), file_name="urbanpulse_real.geojson", mime="application/geo+json")


with st.sidebar:
    st.header("Analysis")
    mode = st.radio("Mode", ["Real AOI inference", "Demo results"], index=0)

if mode == "Real AOI inference":
    st.subheader("Run UrbanPulse on a real AOI")
    st.info("Upload a GeoJSON polygon of at least 10 km². The app downloads Sentinel-2 surface reflectance tile-by-tile and aggregates it to 100 m cells.")
    upload = st.file_uploader("AOI GeoJSON", type=["geojson", "json"])
    year = st.slider("Analysis year", min_value=2019, max_value=2026, value=2025)
    use_osm = st.checkbox("Use OpenStreetMap buildings + roads", value=True, help="Adds structural morphology features. This can increase runtime for large AOIs.")
    run = st.button("🚀 Run real analysis", type="primary", disabled=upload is None)

    with st.expander("Earth Engine setup"):
        st.markdown("""
        The deployed app expects these Streamlit secrets: `GEE_PROJECT_ID`, `GEE_SERVICE_ACCOUNT`, and `GEE_PRIVATE_KEY`.
        Create them in the app's Streamlit Cloud **Settings → Secrets**; do not commit the private key to GitHub.
        """)
        st.code('GEE_PROJECT_ID = "your-google-cloud-project"\nGEE_SERVICE_ACCOUNT = "service-account@project.iam.gserviceaccount.com"\nGEE_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"', language="toml")

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

                gdf, area, tiles = run_real_inference(
                    aoi, year=year, include_osm=use_osm, progress=progress
                )
                st.write(f"Processed {tiles} × 1 km tiles.")

            status.update(label="Analysis complete", state="complete")

            # Persist the completed analysis across Streamlit reruns.
            # This keeps the map visible when st_folium or another widget
            # causes the script to rerun.
            st.session_state["real_results"] = {
                "gdf": gdf,
                "area": area,
                "tiles": tiles,
                "year": year,
            }
        except Exception as e:
            st.error(str(e))
            st.exception(e)

    # Render the latest real-analysis result on every rerun so the map does
    # not disappear immediately after the initial analysis completes.
    result = st.session_state.get("real_results")
    if result is not None:
        display_results(
            result["gdf"],
            result["area"],
            result["tiles"],
            f"UrbanPulse {result['year']}",
        )
else:
    stats_path = RESULTS / "stats.json"
    if not stats_path.exists():
        st.warning("No demo results found. Run `python -m urbanpulse.run demo` first.")
        st.stop()
    stats = json.loads(stats_path.read_text())
    h = stats["headline"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Study area", f"{h['study_area_km2']:.2f} km²")
    c2.metric("Informal morphology", f"{h['informal_area_km2']:.2f} km²")
    c3.metric("Clusters", h["informal_clusters"])
    c4.metric("Confidence", f"{h['high_confidence_share']*100:.1f}%")
    c5.metric("Emerging cells", h.get("emerging_cells", 0))
    st.warning("Demo mode uses bundled synthetic results. Do not present these numbers as real Chennai findings.")
    files = {
        "Informal morphology": RESULTS / "informal_morphology.geojson",
        "Emerging zones": RESULTS / "emerging_zones.geojson",
        "Uncertain zones": RESULTS / "uncertain_zones.geojson",
        "All cells": RESULTS / "cells.geojson",
        "Clusters": RESULTS / f"clusters_{max(stats['area_by_year'].keys())}.geojson",
    }
    import geopandas as gpd
    layer = st.selectbox("Map layer", list(files.keys()))
    if files[layer].exists():
        show_map(gpd.read_file(files[layer]), layer)
    area = pd.DataFrame(stats["area_by_year"]).T
    if "informal" in area:
        st.subheader("Temporal change")
        st.line_chart(area[["informal"]].rename(columns={"informal": "Informal morphology (km²)"}))

with st.expander("Methodology and interpretation"):
    st.markdown("""
    **UrbanPulse does not identify people, income or legal status.** It maps observable urban morphology.

    **Real-data pipeline:** AOI validation → 1 km processing tiles → Sentinel-2 L2A surface reflectance → spectral indices → optional OSM building/road morphology → 100 m spatial context → built-up gate → morphology classification → IMI and confidence.

    **Sparse-area safeguard:** cells with insufficient building density, building count, impervious surface and neighbourhood built context are assigned `sparse_low_development`; their IMI is forced to zero. Thresholds are starting values and should be calibrated with real Chennai labels before claiming production accuracy.

    **Important ML note:** the current real-data mode is a transparent morphology inference baseline. For the final hackathon model, collect labelled Chennai cells and train the Random Forest with spatial-block validation; do not treat synthetic training accuracy as real-world accuracy.
    """)
