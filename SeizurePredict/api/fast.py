from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pyedflib import highlevel

import os

from SeizurePredict.filtering import CustomTranformer
from SeizurePredict.main import preprocess, load_model

from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# signals[16][int(os.environ.get("START_INX_H")):500000]
app.state.model = load_model()

@app.post('/upload_file/')
async def upload_file(file: UploadFile):
    path = Path('/home/lee/code/mariaaraujovitoria/SeizurePredict/raw_data/user_data') / file.filename
    size = path.write_bytes(await file.read())
    return {'file': path, 'bytes': size}

@app.get("/predict/")
def predict_seizure(path: str):
    signals, signal_headers, header = highlevel.read_edf(path)
    X, y = preprocess(path, CustomTranformer(), int(os.environ["PATIENCE_TEST_NUMBER"]), Fourier=False)
    model = app.state.model
    y_pred = model.predict(X)
    raw_signal = signals[16][256*1500:256*1900]
    return {'result':y_pred[(150-1)*2:(190-1)*2:2].tolist(), 'signal': raw_signal.tolist()}

@app.get("/")
def root():
    return {"response":"200"}
