# src/api/main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.bnb_analytics.charts import chart_avg_price_by_neighborhood, chart_feature_importance
import pandas as pd

from src.bnb_analytics.inside_airbnb import fetch_and_load_listings
from src.bnb_analytics.data_prep import clean_listings
from src.bnb_analytics.model import train_baseline_model
from src.bnb_analytics.charts import chart_avg_price_by_neighborhood, chart_feature_importance

app = FastAPI(title="bnb-scout-analytics API")

# Allow your Angular dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache (fine for dev)
_df_cache: pd.DataFrame | None = None
_fi_cache: pd.DataFrame | None = None
_metrics_cache: dict | None = None
_city = None
_snapshot = None

class TrainRequest(BaseModel):
    city: str
    date: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/train")
def train(req: TrainRequest):
    city = req.city
    date = req.date
    """
    Fetch -> Clean -> Train. Caches df, feature importances, metrics for chart endpoints.
    """
    global _df_cache, _fi_cache, _metrics_cache, _city, _snapshot

    # 1) Fetch (uses your simplified inside_airbnb function)
    res = fetch_and_load_listings(city, date)
    raw_df = res["df"]

    # 2) Clean
    df = clean_listings(raw_df, min_property_type_count=100)

    # 3) Train
    model, metrics, fi = train_baseline_model(df)

    # 4) Cache for chart endpoints
    _df_cache = df
    _fi_cache = fi
    _metrics_cache = metrics
    _city = res["city"]
    _snapshot = res["snapshot_date"]

    return JSONResponse({
        "status": "ok",
        "city": _city,
        "snapshot": _snapshot,
        "metrics": metrics,
        "top_features": fi.head(12).to_dict(orient="records"),
    })

@app.get("/metrics")
def metrics():
    if _metrics_cache is None:
        raise HTTPException(status_code=400, detail="Train first by calling POST /train")
    return JSONResponse({
        "city": _city,
        "snapshot": _snapshot,
        "metrics": _metrics_cache
    })

@app.get("/charts/avg-price-neighborhood.png")
def avg_price_chart():
    if _df_cache is None:
        raise HTTPException(status_code=400, detail="Train first by calling POST /train")
    buf = chart_avg_price_by_neighborhood(_df_cache)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/charts/feature-importance.png")
def feature_importance_chart(k: int = 12):
    if _fi_cache is None:
        raise HTTPException(status_code=400, detail="Train first by calling POST /train")
    buf = chart_feature_importance(_fi_cache, k=k)
    return StreamingResponse(buf, media_type="image/png")
