from taxifare.pipeline import get_pipeline
from taxifare.trainer import Trainer
from sklearn.model_selection import GridSearchCV
from taxifare.data import get_data, clean_df, holdout
from taxifare.utils import save_model


class ParamTrainer:
    def __init__(self):
        pass

    def train(self, params):
        models = {}
        for model, model_params in params.items():
            df = get_data(nrows=model_params["line_count"])
            df_clean = clean_df(df)
            X_train, X_test, y_train, y_test = holdout(df_clean)

            pipeline = get_pipeline(regressor=model)
            grid_search = GridSearchCV(
                pipeline, param_grid=model_params["hyper_params"], cv=5
            )
            grid_search.fit(X_train, y_train)
            models[model] = grid_search
            save_model(grid_search, model)

        return models


if __name__ == "__main__":
    params = dict(
        random_forest=dict(
            line_count=1_000,
            hyper_params=dict(
                features__distance__distancetransformer__distance_type=[
                    "euclidian",
                    "manhattan",
                ],
                features__distance__standardscaler__with_mean=[True, False],
                model__max_depth=[1, 2, 3],
            ),
        ),
        linear_regression=dict(
            line_count=1_000,
            hyper_params=dict(
                features__distance__distancetransformer__distance_type=[
                    "euclidian",
                    "manhattan",
                ],
                features__distance__standardscaler__with_mean=[True, False],
                model__normalize=[True, False],
            ),
        ),
    )
    param_trainer = ParamTrainer()
    models = param_trainer.train(params)
    print(models)
