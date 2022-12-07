FROM python:3.10.6-buster
COPY requirements.txt /requirements.txt
COPY SeizurePredict /SeizurePredict
COPY setup.py /setup.py
COPY model.joblib /model.joblib

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn SeizurePredict.api.fast:app --host 0.0.0.0 --port $PORT
