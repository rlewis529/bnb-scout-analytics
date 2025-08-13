import pandas as pd
from src.bnb_analytics.data_prep import clean_listings

def test_clean_listings_basic():
    # Create a minimal mock dataset
    data = {
        'price': ['$100.00', '$250.50', '$3,500.00'],
        'bathrooms_text': ['1 bath', '2.5 baths', None],
        'bedrooms': [1, 2, 15],  # 15 should be removed by outlier filter
        'accommodates': [2, 4, 6],
        'room_type': ['Entire home/apt', 'Private room', 'Shared room'],
        'property_type': ['House', 'Treehouse', 'Cabin'],
        'amenities': ['{Wifi, Kitchen}', '{Wifi}', None],
        'review_scores_rating': [95.0, 80.0, 90.0],
        'neighbourhood_cleansed': ['28801', '28801', '28805']
    }
    df = pd.DataFrame(data)

    # Run cleaning function
    cleaned = clean_listings(df, min_property_type_count=1)

    # --- Tests ---
    # 1. Outlier removal: should drop the row with bedrooms > 10
    assert cleaned['bedrooms'].max() <= 10

    # 2. Price conversion: should now be floats without symbols
    assert pd.api.types.is_float_dtype(cleaned['price'])
    assert cleaned['price'].iloc[0] == 100.00

    # 3. Bathrooms extraction: should be float and match expected
    assert cleaned['bathrooms'].iloc[0] == 1.0
    assert cleaned['bathrooms'].iloc[1] == 2.5

    # 4. Amenities count: should match number of items
    assert cleaned['amenities_count'].iloc[0] == 2
    assert cleaned['amenities_count'].iloc[1] == 1

    # 5. Property type grouping should exist
    assert 'property_type_grouped' in cleaned.columns

    # 6. No NaNs in final DataFrame
    assert cleaned.isna().sum().sum() == 0
