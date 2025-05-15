import streamlit as st
import joblib
import numpy as np
import librosa

# Load model and label encoder
model = joblib.load("emotion_model.joblib")
encoder = joblib.load("label_encoder.joblib")

# Extract audio features
def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, contrast])

st.title("🎙️ Emotion Detection from Audio")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    try:
        features = extract_features(uploaded_file)
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        label = encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Emotion: **{label}**")
    except Exception as e:
        st.error(f"Error: {e}")
