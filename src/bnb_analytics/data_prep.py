import pandas as pd
import numpy as np

def clean_listings(df: pd.DataFrame, min_property_type_count: int = 100) -> pd.DataFrame:
    """Clean and preprocess the Airbnb listings DataFrame."""

    df = df.copy()

    # Convert price to numeric
    df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)

    # Extract numeric bathrooms
    df['bathrooms'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)

    # Count amenities
    df['amenities_count'] = df['amenities'].apply(lambda x: len(x.strip('{}').split(',')) if pd.notnull(x) else 0)

    # Group low-frequency property types
    property_counts = df['property_type'].value_counts()
    low_freq_types = property_counts[property_counts < min_property_type_count].index
    df['property_type_grouped'] = df['property_type'].apply(
        lambda x: "Other" if x in low_freq_types else x
    )

    # Drop obvious outliers
    df = df[(df['bedrooms'] <= 10) & (df['price'] <= 3000)]

    # Keep only useful columns
    keep_cols = [
        'bedrooms', 'bathrooms', 'accommodates',
        'room_type', 'property_type_grouped',
        'amenities_count', 'review_scores_rating',
        'neighbourhood_cleansed', 'price'
    ]
    df = df[keep_cols].dropna()

    return df
