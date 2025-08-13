import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def train_baseline_model(df: pd.DataFrame):
    """
    Train a baseline linear regression model on cleaned Airbnb data.
    Returns: model, metrics, feature_importances_df
    """
    # ----- Define target and features -----
    target = "price"
    features = [
        "bedrooms", "bathrooms", "accommodates", "room_type",
        "property_type_grouped", "amenities_count",
        "review_scores_rating", "neighbourhood_cleansed"
    ]

    X = df[features]
    y = df[target]

    # ----- Identify column types -----
    numeric_features = ["bedrooms", "bathrooms", "accommodates", "amenities_count", "review_scores_rating"]
    categorical_features = ["room_type", "property_type_grouped", "neighbourhood_cleansed"]

    # ----- Preprocessor -----
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # ----- Pipeline -----
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # ----- Train/test split -----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----- Fit -----
    model.fit(X_train, y_train)

    # ----- Predictions -----
    y_pred = model.predict(X_test)

    # ----- Metrics -----
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"RMSE": rmse, "R2": r2}

    # ----- Feature Importance -----
    cat_feature_names = model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    coefficients = model.named_steps["regressor"].coef_

    feature_importances_df = pd.DataFrame({
        "feature": all_feature_names,
        "coefficient": coefficients,
        "abs_coeff": np.abs(coefficients)
    }).sort_values(by="abs_coeff", ascending=False)

    return model, metrics, feature_importances_df
