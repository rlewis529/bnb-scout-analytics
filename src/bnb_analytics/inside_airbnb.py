# src/bnb_analytics/inside_airbnb.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import pandas as pd
import requests

# --- tiny registry you can extend later ---
# keys are lowercase for simple matching
CITY_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    # city_name_lower: (country, region/state code, city slug used by Inside Airbnb)
    "asheville": ("united-states", "nc", "asheville"),
    # add more later:
    # "boone": ("united-states", "nc", "boone"),
    # "wilmington": ("united-states", "nc", "wilmington"),
    # "outer-banks": ("united-states", "nc", "outer-banks"),
}

@dataclass(frozen=True)
class CityRef:
    country: str      # e.g. "united-states"
    region: str       # e.g. "nc"
    city: str         # e.g. "asheville"

def resolve_city(city_query: str) -> CityRef:
    key = city_query.strip().lower()
    if key not in CITY_REGISTRY:
        raise ValueError(f"City '{city_query}' not in registry. Available: {list(CITY_REGISTRY.keys())}")
    country, region, city = CITY_REGISTRY[key]
    return CityRef(country, region, city)

def build_listings_url(ref: CityRef, snapshot_date: str) -> str:
    """
    Inside Airbnb URL pattern:
    http://data.insideairbnb.com/{country}/{region}/{city}/{YYYY-MM-DD}/data/listings.csv.gz
    """
    return (
        f"http://data.insideairbnb.com/"
        f"{ref.country}/{ref.region}/{ref.city}/{snapshot_date}/data/listings.csv.gz"
    )

def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest

def load_csv(path_or_url: str | Path) -> pd.DataFrame:
    return pd.read_csv(path_or_url, compression="infer")

def fetch_and_load_listings(city: str, snapshot_date: str, out_root: Path | str = "data/raw"):
    """
    Minimal, explicit version:
      - resolve city using our registry
      - build the known URL using a provided snapshot date
      - download and load
    """

    ref = resolve_city(city)
    url = build_listings_url(ref, snapshot_date)
    local_path = Path(out_root) / f"{ref.city}-{ref.region}-{ref.country}" / snapshot_date / "listings.csv.gz"
    download(url, local_path)
    df = load_csv(local_path)
    return {
        "city": ref.city,
        "region": ref.region,
        "country": ref.country,
        "snapshot_date": snapshot_date,
        "url": url,
        "path": str(local_path),
        "df": df,
    }
