# Importing Libraries

from numpy import isscalar
from librosa.feature import spectral_rolloff
from librosa.feature import chroma_stft
from flask import Flask
from flask import render_template
from flask import request
import joblib
import numpy as np
import pandas as pd
import librosa
import os

# Creating a Flask named App

app = Flask(__name__)

# Storing The two pickle

model = joblib.load('rf_genre_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function and loading Audio

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration = 3)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y,sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    harmony, perceptr = librosa.effects.hpss(y)
    tempo,_ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 20)

    # Creating Features

    feature = [
        np.mean(chroma_stft),
        np.var(chroma_stft),
        np.mean(rms),
        np.var(rms),
        np.mean(spec_cent),
        np.var(spec_cent),
        np.mean(spec_bw),
        np.var(spec_bw),
        np.mean(spec_rolloff),
        np.var(spec_rolloff),
        np.mean(zcr),
        np.var(zcr),
        np.mean(harmony),
        np.var(harmony),
        np.mean(perceptr),
        np.var(perceptr),

        tempo if np.isscalar(tempo) else tempo[0]
        ]

    for i in range(20):
        feature.append(np.mean(mfcc[i]))
        feature.append(np.var(mfcc[i]))

    return np.array(feature)

# Creating App

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    file = request.files['file']
    file_path = "temp_audio.wav"
    file.save(file_path)
    features = extract_features(file_path)
    features = features.reshape(1,-1)
    prediction = model.predict(features)
    genre = label_encoder.inverse_transform(prediction)[0]

    os.remove(file_path)
    return render_template('result.html',genre = genre)

# To start the Website

if __name__ == '__main__':
    app.run(debug=True)