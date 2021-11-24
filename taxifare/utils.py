import numpy as np
import joblib
from google.cloud import storage
from taxifare.data import BUCKET_NAME

MODEL_PATH = "model2/taxifare/"

def minkowski_distance(df, p,
                       start_lat="pickup_latitude",
                       start_lon="pickup_longitude",
                       end_lat="dropoff_latitude",
                       end_lon="dropoff_longitude"):
    x1 = df[start_lon]
    x2 = df[end_lon]
    y1 = df[start_lat]
    y2 = df[end_lat]
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)

def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def save_model(pipeline, model):
    model_file_path = f"{model}.joblib"
    joblib.dump(pipeline, model_file_path)
    path_inside_bucket = MODEL_PATH + model + ".joblib"
    
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path_inside_bucket)
    
    print(f"uploading {model_file_path} to {BUCKET_NAME}")
    blob.upload_from_filename(model_file_path)
    