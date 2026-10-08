# UrbanPulse model limitations

1. **Weak/reference labels:** The current evaluation uses promoted weak/reference
   labels rather than an independently verified ground-truth benchmark.

2. **Morphology, not socioeconomic status:** The model identifies physical
   characteristics associated with informal urban morphology. It does not infer
   income, tenure, legality, household conditions, or resident characteristics.

3. **Sentinel-2 resolution:** The 100 m analysis grid can average out very small
   structures and fine-grained neighbourhood differences.

4. **OSM completeness:** Building and road features depend partly on OpenStreetMap
   completeness, which varies spatially.

5. **Probability is not calibrated certainty:** Model class probability should not
   be interpreted as guaranteed probability of correctness.

6. **Chennai proof-of-concept:** Transfer to another city requires additional
   validation or calibration.

7. **Temporal/demo content:** Bundled temporal/emerging-zone outputs should not be
   presented as fresh Sentinel-2 findings unless regenerated from current imagery.
