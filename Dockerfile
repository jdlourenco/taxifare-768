FROM python:3.8.12-buster

COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
COPY api /api
COPY taxifare /taxifare

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
