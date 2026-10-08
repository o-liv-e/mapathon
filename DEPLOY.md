# UrbanPulse deployment

## Fastest local demo

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m urbanpulse.run demo
streamlit run app.py
```

Open the URL printed by Streamlit (normally http://localhost:8501).

## Docker

```bash
docker compose up --build
```

Then open http://localhost:8501.

## Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app.
3. Select the repository and `app.py` as the entry point.
4. Deploy.
5. Commit your generated `results/` outputs if the hosted demo is meant to be a static showcase.

For real satellite processing, do **not** put Earth Engine credentials in the repository. Run the Earth Engine export locally/through your secured environment, then publish only the derived outputs needed by the dashboard.

## Render / Railway / generic Docker host

Use the included `Dockerfile`. The application listens on port 8501.

## Production architecture

For the full hackathon system, keep Streamlit as the demo UI and run analysis jobs separately:

`POST /analysis -> job queue -> Python worker -> results/ or object storage -> dashboard`

Do not run Earth Engine exports or model training inside a web request.
