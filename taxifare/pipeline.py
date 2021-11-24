from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from taxifare.encoders import DistanceTransformer


def get_pipeline(regressor="random_forest"):
    pipe_distance = make_pipeline(DistanceTransformer(), StandardScaler())

    cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    feateng_blocks = [
        ("distance", pipe_distance, cols),
    ]

    features_encoder = ColumnTransformer(feateng_blocks)

    pipeline = Pipeline(
        steps=[
            ("features", features_encoder),
            ("model", get_model(regressor=regressor)),
        ]
    )

    return pipeline


def get_model(regressor="random_forest"):
    if regressor == "random_forest":
        model_params = dict(n_estimators=100, max_depth=1)

        model = RandomForestRegressor()
        model.set_params(**model_params)
    elif regressor == "linear_regression":
        model = LinearRegression()

    return model
