from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}


@app.get("/predict")
def predict(pickup_datetime, lon1, lat1, lon2, lat2, passcount):
    # get model
    # format input

    model = joblib.load("model.joblib")
    X = pd.DataFrame({
        "key": ["write_something"],
        "pickup_datetime": [pickup_datetime],
        "pickup_longitude": [float(lon1)],
        "pickup_latitude": [float(lat1)],
        "dropoff_longitude": [float(lon2)],
        "dropoff_latitude": [float(lat2)],
        "passenger_count": [int(passcount)],
    })
    
    prediction = model.predict(X)
    print(prediction)
    
    return {"prediction": prediction[0]}
