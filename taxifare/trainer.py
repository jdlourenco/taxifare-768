from taxifare.data import get_data, clean_df, holdout
from taxifare.pipeline import get_pipeline
from taxifare.utils import compute_rmse, save_model
import joblib


class Trainer:
    def __init__(self, **kwargs):
        print(kwargs)
        self.regressor = kwargs.get("regressor", "random_forest")
        self.nrows = kwargs.get("nrows", 1000)
        print(f"regressor={self.regressor}")

    def evaluate(self):  # , pipeline, X_test, y_test):
        # y_pred = pipeline.predict(X_test)
        # rmse = compute_rmse(y_pred, y_test)
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        print(f"rmse={rmse}")
        return rmse

    def train(self):
        # get data
        df = get_data(nrows=self.nrows)
        print(df.shape)

        df_clean = clean_df(df)
        print(df_clean.shape)

        # holdout
        # X_train, X_test, y_train, y_test = holdout(df)
        self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)

        # get pipeline
        # pipeline = get_pipeline()
        self.pipeline = get_pipeline(regressor=self.regressor)
        print(self.pipeline)

        # train model/pipeline
        # pipeline.fit(X_train, y_train)
        self.pipeline.fit(self.X_train, self.y_train)
        # evaluate
        # self.evaluate(pipeline)
        self.evaluate()
        # save model
        save_model(self.pipeline, self.regressor)


if __name__ == "__main__":
    for params in [
        # {"model": "random_forest", "nrows": 1000},
        {"model": "linear_regression", "nrows": 2000},
        # {"model": "random_forest", "nrows": 3000},
        # {"model": "linear_regression", "nrows": 4000},
    ]:
        trainer = Trainer(regressor=params["model"], nrows=params["nrows"])
        fitted_pipeline = trainer.train()
