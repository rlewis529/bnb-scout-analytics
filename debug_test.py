# debug_test.py
from src.bnb_analytics.inside_airbnb import fetch_and_load_listings

res = fetch_and_load_listings("asheville", "2025-06-17")
print(res["df"].head())
