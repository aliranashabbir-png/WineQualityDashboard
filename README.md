# VineSense AI: Wine Quality Dashboard

A Streamlit dashboard for exploring the Wine Quality dataset and estimating a wine's quality from its measured chemical properties.

## Features

- Wine-quality distribution and dataset summaries
- Random Forest quality prediction from 11 chemistry inputs
- Feature importance and model evaluation views
- Interactive Plotly charts

## Run locally

From the project root, install the dependencies and start the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run Steamlit/app.py
```

The app reads `WineQT.csv` from the project root, so keep that file alongside this README when deploying.

## Publish a shareable link

`http://localhost:8501` is only reachable from your own computer. To share the dashboard online:

1. Push this project to a GitHub repository. Include `Steamlit/app.py`, `WineQT.csv`, `requirements.txt`, and this README. Do not upload `.venv` or private data.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Create an app and select your repository and branch.
4. Set the app file path to `Steamlit/app.py` and deploy.
5. Share the generated `*.streamlit.app` URL.

For a branded address, connect a custom domain through a hosting provider that supports proxying the Streamlit app. The generated Streamlit URL is the simplest public link to start with.

## Model note

The displayed score is evaluated on a held-out split of this dataset. It is a demonstration of classification on these data, not a guarantee of how a wine will taste or a substitute for expert evaluation.