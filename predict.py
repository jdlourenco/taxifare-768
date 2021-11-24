import pandas as pd
from google.cloud import storage
from pandas.core.algorithms import mode
from taxifare.data import BUCKET_NAME
import joblib

MODEL_BUCKET_PATH = "model2/taxifare"
TEST_SET_PATH = "raw_data/test.csv"

def get_test_set(path):
    df = pd.read_csv(path)
    
    return df

def get_model(model="linear_regression"):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{MODEL_BUCKET_PATH}/{model}.joblib")
    
    local_model_path = f"{model}.joblib"
    blob.download_to_filename(local_model_path)
    model = joblib.load(local_model_path)
    
    return model
    
def get_predictions(model, test_set):
    predictions = model.predict(test_set)

    return predictions


test_set = get_test_set(path=TEST_SET_PATH)
model = get_model()
predictions = get_predictions(model, test_set)

test_set["fare_amount"] = predictions
test_set = test_set[["key", "fare_amount"]]
test_set.to_csv("kaggle_results.csv", index=False)
