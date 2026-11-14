# Music Genre Classification System

An end-to-end audio classification web application that automatically identifies the musical genre of audio files using digital signal processing (DSP) and machine learning.

---

## 1. Overview

This project uses Python, Librosa, Scikit-learn, and Flask to classify music tracks into standard genres. Audio features are extracted from audio waveforms in real time and passed to a trained Random Forest Classifier, which predicts the genre and returns the result through a clean web interface.

The model is trained on acoustic features derived from the standard GTZAN music dataset.

---

## 2. Key Features

- **Automated Feature Extraction**: Analyzes uploaded audio files using Librosa to extract spectral, timbral, and rhythmic characteristics.
- **Robust Machine Learning Model**: Uses a tuned Random Forest Classifier with 200 estimators and depth constraints to prevent overfitting.
- **Fast Real-Time Inference**: Serialized models (`.pkl`) loaded with Joblib allow near-instant predictions.
- **Web Interface**: Simple Flask web application for uploading audio clips and viewing predictions.
- **Temporary File Lifecycle**: Automatically cleans up temporary audio files after inference to save disk space.

---

## 3. Supported Genres

The model classifies audio across 10 distinct genres:
- Blues
- Classical
- Country
- Disco
- Hiphop
- Jazz
- Metal
- Pop
- Reggae
- Rock

---

## 4. Acoustic Features Extracted

The system extracts statistical representations (mean and variance) across multiple audio dimensions:

1. **Chroma STFT** (Pitch & Harmonic Content)
2. **RMS** (Root Mean Square Energy / Loudness)
3. **Spectral Centroid** (Brightness of Sound)
4. **Spectral Bandwidth** (Frequency Spread)
5. **Spectral Roll-off** (Frequency Cutoff Point)
6. **Zero-Crossing Rate (ZCR)** (Percussiveness / Noise)
7. **Harmonic and Percussive Components** (HPSS Separation)
8. **Tempo** (Beats Per Minute)
9. **MFCCs** (20 Mel-Frequency Cepstral Coefficients capturing timbre)

---

## 5. Project Structure

```text
music_project/
│
├── Data/
│   ├── features_3_sec.csv       # Extracted feature dataset (3-second clips)
│   ├── features_30_sec.csv      # Extracted feature dataset (30-second clips)
│   └── genres_original/         # Raw audio tracks categorized by genre
│
├── static/
│   └── style.css                # Styling for the web interface
│
├── templates/
│   ├── index.html               # Audio upload page
│   └── result.html              # Genre prediction result page
│
├── app.py                       # Flask web server and prediction pipeline
├── model.py                     # Model training, evaluation, and export script
├── rf_genre_model.pkl           # Serialized Random Forest model
├── label_encoder.pkl            # Serialized label encoder for genre names
└── README.md                    # Project documentation
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.8 to 3.11
- pip (Python package manager)
- (Optional) Virtual environment tool (`venv` or `conda`)

### Step 1: Clone or Navigate to the Project Directory
```bash
cd music_project
```

### Step 2: Create and Activate a Virtual Environment
- **Windows:**
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Required Dependencies
```bash
pip install flask librosa numpy pandas scikit-learn joblib
```

---

## 7. How to Use

### 1. Training the Model (Optional)
If you want to retrain or fine-tune the classifier:
```bash
python model.py
```
This script will:
- Load `Data/features_3_sec.csv`
- Split the dataset into Training (70%), Validation (20%), and Testing (10%) sets
- Train the `RandomForestClassifier`
- Output the classification report and accuracy metrics
- Save `rf_genre_model.pkl` and `label_encoder.pkl`

### 2. Running the Web Application
Start the Flask application:
```bash
python app.py
```

Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

1. Upload an audio file (`.wav`).
2. Click **Predict** / **Submit**.
3. View the predicted genre on the results page.

---

## 8. Technology Stack

- **Language:** Python
- **Signal Processing:** Librosa
- **Machine Learning:** Scikit-learn
- **Data Handling:** NumPy, Pandas
- **Serialization:** Joblib
- **Web Framework:** Flask
- **Frontend:** HTML, CSS

---

## 9. Future Improvements

- Support for additional audio formats (`.mp3`, `.ogg`, `.flac`).
- Interactive waveform and audio spectrogram visualizations in the browser.
- Displaying prediction probability / confidence scores across all genres.
- Deep Learning experimentation using Convolutional Neural Networks (CNNs) on mel-spectrograms.
