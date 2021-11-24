import pandas as pd
from sklearn.model_selection import train_test_split

BUCKET_NAME = "wagon-768-jdlourenco"
PATH_TO_DATA = "data/train_1k.csv"

def get_data(nrows=1000):
    # url = "s3://wagon-public-datasets/taxi-fare-train.csv"
    url = f"gs://{BUCKET_NAME}/{PATH_TO_DATA}"
    print(f"loading the dataset from {url}")
    df = pd.read_csv(url, nrows=nrows)
    return df

def clean_df(df):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 1]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def holdout(df):
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return (X_train, X_test, y_train, y_test)