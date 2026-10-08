"""UrbanPulse hackathon dashboard.

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UrbanPulse",
    page_icon="🛰️",
    layout="wide",
)

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


# ============================================================
# HEADER
# ============================================================

st.title("🛰️ UrbanPulse")

st.caption(
    "Explainable urban morphology intelligence for "
    "informal-neighbourhood mapping"
)


# ============================================================
# LOAD RESULTS
# ============================================================

stats_path = RESULTS / "stats.json"

if not stats_path.exists():
    st.warning(
        "No results found. Run "
        "`python -m urbanpulse.run demo` first."
    )
    st.stop()

try:
    stats = json.loads(
        stats_path.read_text(encoding="utf-8")
    )
except Exception as e:
    st.error(f"Could not read stats.json: {e}")
    st.stop()


headline = stats.get("headline", {})


# ============================================================
# HEADLINE METRICS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Study area",
    f"{headline.get('study_area_km2', 0):.2f} km²",
)

c2.metric(
    "Informal morphology",
    f"{headline.get('informal_area_km2', 0):.2f} km²",
)

c3.metric(
    "Clusters",
    headline.get("informal_clusters", 0),
)

c4.metric(
    "Confidence",
    f"{headline.get('high_confidence_share', 0) * 100:.1f}%",
)

c5.metric(
    "Emerging cells",
    headline.get("emerging_cells", 0),
)


# ============================================================
# WHAT MAKES URBANPULSE DIFFERENT?
# ============================================================

st.subheader("What makes UrbanPulse different?")

st.markdown(
    """
- **Satellite-first:** Sentinel-2 provides the primary observable evidence.
- **Morphology-aware:** buildings, roads and neighbourhood context are fused with spectral features.
- **Explainable:** every cell has a confidence score and an Informal Morphology Index (IMI).
- **Temporal:** annual morphology change reveals stable, changing and emerging zones.
- **Spatially validated:** model evaluation uses geographic blocks rather than random cell splits.
"""
)


# ============================================================
# MAP LAYER SELECTION
# ============================================================

st.subheader("UrbanPulse map")

area_by_year = stats.get("area_by_year", {})

if area_by_year:
    try:
        latest_year = max(
            area_by_year.keys(),
            key=lambda x: int(x)
        )
    except Exception:
        latest_year = "2026"
else:
    latest_year = "2026"


files = {
    "Informal morphology":
        RESULTS / "informal_morphology.geojson",

    "Emerging zones":
        RESULTS / "emerging_zones.geojson",

    "Uncertain zones":
        RESULTS / "uncertain_zones.geojson",

    "All cells":
        RESULTS / "cells.geojson",

    "Clusters":
        RESULTS / f"clusters_{latest_year}.geojson",
}


layer = st.selectbox(
    "UrbanPulse analysis layer",
    list(files.keys()),
)


# ============================================================
# IMPORT MAP LIBRARIES
# ============================================================

try:
    import folium
    import geopandas as gpd
    from streamlit_folium import st_folium

except ImportError:
    st.error(
        "Map dependencies are missing. Run:\n\n"
        "`python -m pip install -r requirements.txt`"
    )
    st.stop()


# ============================================================
# LOAD SELECTED GEOJSON
# ============================================================

path = files[layer]

if not path.exists():

    st.info(
        f"No data available for **{layer}**."
    )

else:

    try:

        gdf = gpd.read_file(path)

        if gdf.empty:

            st.info(
                f"The **{layer}** layer contains no features."
            )

        else:

            # ----------------------------------------------------
            # Coordinate system
            # ----------------------------------------------------

            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")

            gdf = gdf.to_crs("EPSG:4326")

            # ----------------------------------------------------
            # Calculate map center
            # ----------------------------------------------------

            min_lon, min_lat, max_lon, max_lat = (
                gdf.total_bounds
            )

            center_lat = (
                min_lat + max_lat
            ) / 2

            center_lon = (
                min_lon + max_lon
            ) / 2

            # ----------------------------------------------------
            # CREATE MAP
            # ----------------------------------------------------

            m = folium.Map(
                location=[
                    center_lat,
                    center_lon
                ],
                zoom_start=13,
                control_scale=True,
                tiles=None,
            )

            # ====================================================
            # BASEMAP 1 — OPENSTREETMAP
            # ====================================================

            folium.TileLayer(
                tiles="OpenStreetMap",
                name="🗺️ Street Map",
                overlay=False,
                control=True,
            ).add_to(m)

            # ====================================================
            # BASEMAP 2 — SATELLITE
            # ====================================================

            folium.TileLayer(
                tiles=(
                    "https://server.arcgisonline.com/"
                    "ArcGIS/rest/services/World_Imagery/"
                    "MapServer/tile/{z}/{y}/{x}"
                ),
                attr=(
                    "Esri, Maxar, Earthstar Geographics, "
                    "and the GIS User Community"
                ),
                name="🛰️ Satellite",
                overlay=False,
                control=True,
            ).add_to(m)

            # ====================================================
            # BASEMAP 3 — TERRAIN
            # ====================================================

            folium.TileLayer(
                tiles=(
                    "https://{s}.tile.opentopomap.org/"
                    "{z}/{x}/{y}.png"
                ),
                attr="OpenTopoMap",
                name="⛰️ Terrain",
                overlay=False,
                control=True,
            ).add_to(m)

            # ====================================================
            # TOOLTIP FIELDS
            # ====================================================

            preferred_fields = [
                "imi",
                "confidence",
                "emerging_score",
                "predicted_class",
                "trajectory",
                "area_km2",
                "buildings",
                "mean_confidence",
                "first_detected",
                "area_change_km2",
            ]

            tooltip_fields = [
                field
                for field in preferred_fields
                if field in gdf.columns
            ]

            if tooltip_fields:

                tooltip = folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=[
                        field.replace(
                            "_", " "
                        ).title()
                        for field in tooltip_fields
                    ],
                    localize=True,
                    sticky=False,
                    labels=True,
                    style="""
                        background-color: white;
                        color: black;
                        font-family: Arial;
                        font-size: 12px;
                        padding: 8px;
                    """,
                )

            else:
                tooltip = None

            # ====================================================
            # URBANPULSE GEOJSON
            # ====================================================

            geojson_args = {
                "data": gdf.to_json(),
                "name": layer,
                "show": True,
            }

            if tooltip is not None:
                geojson_args["tooltip"] = tooltip

            folium.GeoJson(
                **geojson_args
            ).add_to(m)

            # ====================================================
            # LAYER CONTROL
            # ====================================================

            folium.LayerControl(
                position="topright",
                collapsed=False,
            ).add_to(m)

            # ====================================================
            # DISPLAY MAP
            # ====================================================

            st_folium(
                m,
                width=None,
                height=650,
                returned_objects=[],
            )

    except Exception as e:

        st.error(
            f"Could not display the map: {e}"
        )


# ============================================================
# TEMPORAL CHANGE
# ============================================================

st.subheader("Temporal change")

if area_by_year:

    area = pd.DataFrame(
        area_by_year
    ).T

    try:
        area.index = area.index.astype(int)
        area = area.sort_index()
    except Exception:
        pass

    if "informal" in area.columns:

        informal_series = area[
            ["informal"]
        ].rename(
            columns={
                "informal":
                "Informal morphology (km²)"
            }
        )

        st.line_chart(
            informal_series
        )

    else:

        st.info(
            "Informal morphology time-series data "
            "is not available."
        )

else:

    st.info(
        "No temporal data is available."
    )


# ============================================================
# MODEL EVIDENCE
# ============================================================

st.subheader("Model evidence")

top_features = stats.get(
    "top_features",
    {}
)

if top_features:

    features = (
        pd.Series(
            top_features,
            name="importance",
        )
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        features
    )

else:

    st.info(
        "Feature importance information "
        "is not available."
    )


# ============================================================
# CHANGE STATISTICS
# ============================================================

if "change" in stats:

    with st.expander(
        "Change statistics"
    ):

        change_df = pd.DataFrame(
            stats["change"]
        ).T

        st.dataframe(
            change_df,
            use_container_width=True,
        )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "Methodology"
):

    methodology = stats.get(
        "methodology",
        {},
    )

    if methodology:

        st.json(
            methodology
        )

    else:

        st.write(
            "UrbanPulse combines Sentinel-2 satellite "
            "features with building, road and spatial "
            "morphology features. A machine-learning model "
            "predicts observable urban morphology classes "
            "using spatially separated validation."
        )


# ============================================================
# INTERPRETATION
# ============================================================

with st.expander(
    "Important interpretation note"
):

    st.write(
        "UrbanPulse maps observable informal-type urban "
        "morphology. It does not infer residents' income, "
        "identity, ethnicity, or socioeconomic status from "
        "imagery."
    )


# ============================================================
# DEMO NOTE
# ============================================================

with st.expander(
    "Demo data note"
):

    st.write(
        "If this dashboard is running in demo mode, the "
        "displayed results are generated from synthetic "
        "data for testing the complete UrbanPulse pipeline. "
        "They should not be presented as measured findings "
        "about Chennai."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "UrbanPulse — Explainable, scalable urban morphology intelligence"
)