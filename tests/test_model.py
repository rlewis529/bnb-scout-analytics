import pandas as pd
import pytest

from src.bnb_analytics.model import train_baseline_model


@pytest.fixture
def sample_clean_df():
    """Creates a minimal cleaned dataset for testing."""
    data = {
        "bedrooms": [1, 2, 3, 2, 1],
        "bathrooms": [1, 1.5, 2, 1, 1],
        "accommodates": [2, 4, 6, 3, 2],
        "room_type": ["Entire home/apt", "Private room", "Hotel room", "Private room", "Entire home/apt"],
        "property_type_grouped": ["Entire condo", "Private room in home", "Hotel", "Entire home", "Entire condo"],
        "amenities_count": [10, 5, 15, 7, 12],
        "review_scores_rating": [90, 80, 95, 85, 88],
        "neighbourhood_cleansed": ["Downtown", "West Side", "Downtown", "East Side", "West Side"],
        "price": [100, 60, 200, 80, 120]
    }
    return pd.DataFrame(data)


def test_train_baseline_model_outputs(sample_clean_df):
    model, metrics, feat_importances = train_baseline_model(sample_clean_df)

    # Check model object
    assert model is not None

    # Check metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert isinstance(metrics["RMSE"], float)
    assert isinstance(metrics["R2"], float)

    # Check feature importances DataFrame
    assert isinstance(feat_importances, pd.DataFrame)
    assert set(["feature", "coefficient", "abs_coeff"]).issubset(feat_importances.columns)
    assert len(feat_importances) > 0
