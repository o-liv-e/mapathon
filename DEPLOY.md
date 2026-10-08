# Deploy UrbanPulse v3

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, deploy `app.py`.
3. In the app settings, add Earth Engine secrets:

```toml
GEE_PROJECT_ID = "your-project-id"
GEE_SERVICE_ACCOUNT = "service-account@your-project.iam.gserviceaccount.com"
GEE_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
```

4. Ensure the Google Cloud project/service account has Earth Engine access and the Earth Engine API is enabled.
5. Open the app, choose **Real AOI inference**, upload a GeoJSON of at least 10 km², choose a year, and run.

The app processes Sentinel-2 imagery in 1 km tiles and aggregates to 100 m cells. For large city-scale runs, move the worker layer out of Streamlit into a queued backend; keep Streamlit as the interactive front end.
