"""
UrbanPulse Streamlit UI additions.

Add to app.py:
    from urbanpulse.ui_additions import render_validation_and_limitations

Then call:
    render_validation_and_limitations()

Place the call after the production model / metrics section and before the
cell-level XAI section if possible.
"""

import streamlit as st
import pandas as pd


# Current XGBoost spatial-validation confusion matrix from the production
# training run: rows = actual/reference class, columns = predicted class.
XGB_CONFUSION = pd.DataFrame(
    [
        [53, 1, 13],
        [0, 80, 1],
        [6, 0, 217],
    ],
    index=["informal", "open_vegetated", "sparse_low_development"],
    columns=["informal", "open_vegetated", "sparse_low_development"],
)


def render_validation_and_limitations():
    st.markdown("---")
    st.subheader("🧩 Spatial-validation confusion matrix")
    st.caption(
        "XGBoost out-of-fold predictions from spatially grouped cross-validation. "
        "Rows are reference/weak labels; columns are model predictions."
    )

    st.dataframe(XGB_CONFUSION, use_container_width=True)

    normalized = XGB_CONFUSION.div(XGB_CONFUSION.sum(axis=1), axis=0) * 100
    normalized = normalized.round(1)

    st.markdown("**Row-normalized confusion matrix (%)**")
    st.dataframe(normalized, use_container_width=True)

    st.info(
        "Interpretation: most errors occur between morphology classes that can be "
        "physically similar. These are validation results against promoted "
        "weak/reference labels, not independently verified ground truth."
    )

    st.markdown("---")
    st.subheader("⚠️ Model limitations")

    limitations = [
        (
            "1. Weak/reference labels",
            "The current evaluation uses promoted weak/reference labels. "
            "The reported Macro-F1 and confusion matrix therefore measure "
            "agreement with those labels, not independently verified ground truth."
        ),
        (
            "2. Morphology, not socioeconomic status",
            "UrbanPulse identifies physical characteristics associated with "
            "informal urban morphology. It does not infer income, tenure, "
            "legality, household conditions, or resident characteristics."
        ),
        (
            "3. Sentinel-2 spatial resolution",
            "The imagery is aggregated to a 100 m analysis grid, so very small "
            "structures and fine-grained neighbourhood differences may be averaged."
        ),
        (
            "4. OpenStreetMap completeness",
            "Building and road features depend partly on OSM coverage and completeness. "
            "Data quality can vary spatially."
        ),
        (
            "5. Probability is not calibrated certainty",
            "The displayed confidence is based on the model's maximum class probability. "
            "It should not be interpreted as a guaranteed probability of correctness."
        ),
        (
            "6. Chennai proof-of-concept",
            "The current production model is validated as a Chennai proof-of-concept. "
            "Applying it to another city would require additional validation or calibration."
        ),
        (
            "7. Temporal/demo outputs",
            "Any bundled temporal or emerging-zone story should be treated as demonstration "
            "content unless it has been generated from fresh imagery for the selected AOI."
        ),
    ]

    for title, body in limitations:
        with st.expander(title):
            st.write(body)
