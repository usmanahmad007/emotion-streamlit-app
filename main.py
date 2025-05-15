from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
import numpy as np
import librosa

app = FastAPI()

# Load model and encoder
model = joblib.load("emotion_model.joblib")
encoder = joblib.load("label_encoder.joblib")

# Feature extraction
def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, contrast])

class InputData(BaseModel):
    features: list

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Use file-like object directly with librosa
    features = extract_features(file.file)

    # Reshape for prediction
    features_array = np.array(features).reshape(1, -1)

    # Predict using the model
    prediction = model.predict(features_array)
    label = encoder.inverse_transform(prediction)[0]

    return {"emotion": label}

