# UrbanPulse v3

Explainable urban morphology intelligence for mapping informal-neighbourhood morphology.

## What is new in v3

- Upload an arbitrary **GeoJSON AOI** in the Streamlit app.
- Enforces the hackathon minimum of **10 km²**.
- Builds a **100 m analysis grid** and **1 km processing tiles**.
- Uses **Sentinel-2 L2A Surface Reflectance Harmonized** imagery.
- Downloads imagery tile-by-tile instead of loading a city-sized raster.
- Optional OpenStreetMap buildings and roads provide structural morphology.
- Adds a **built-up gate** so sparse/open areas cannot automatically become informal morphology.
- Adds `sparse_low_development` as an explicit output class.
- Real-data mode reports morphology/IMI outputs while clearly separating them from socioeconomic claims.
- Bundled synthetic demo remains available for presentations.

## Real deployment

The Streamlit app needs Earth Engine credentials. Put these in Streamlit Cloud **Settings → Secrets**:

```toml
GEE_PROJECT_ID = "your-project-id"
GEE_SERVICE_ACCOUNT = "service-account@your-project.iam.gserviceaccount.com"
GEE_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
```

Never commit the private key. Local development can use the same secrets file or normal Earth Engine authentication.

## Final ML step

The real-data mode currently uses a transparent morphology baseline so the app can process an AOI immediately. For the final scientific model:

1. collect labelled Chennai cells for informal / planned / high-rise / commercial / industrial / open / water / sparse;
2. compute the same features;
3. train the Random Forest in `urbanpulse.model`;
4. use spatial-block cross-validation;
5. save the trained model and run inference on arbitrary AOIs.

Synthetic demo accuracy must not be presented as Chennai accuracy.

### v3.1 Sentinel-2 download fix

The real-data pipeline now explicitly bounds every Earth Engine tile download
with the tile geometry and a 10 m scale. This fixes Earth Engine's
`Image is unbounded` thumbnail/download error and keeps the tile-based
processing architecture intact.
