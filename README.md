# Smart IoT-Based Microgrid Monitoring and Management Platform for Agrivoltaics

*(Codename: Taizhi — Mobile Communications Practice Competition, Honorable Mention)*

A Flask web application for monitoring photovoltaic (solar) power plants. It combines
time-series deep learning models for DC power prediction and anomaly detection with a
computer-vision classifier for identifying physical faults on solar panels from camera
images.

## Features

- **Dashboard** — loads inverter and weather CSV data and displays a preview table.
- **Power yield prediction** — a CNN + LSTM model predicts per-inverter daily yield from
  historical inverter and weather data, and plots predicted vs. actual output.
- **Anomaly detection** — an LSTM autoencoder (for feature encoding) feeds an MLP
  regressor that predicts expected DC power; large deviations from the prediction are
  flagged as anomalies using a 3-sigma threshold. Error distribution, error time series,
  and anomaly plots are generated automatically.
- **Visual fault classification** — a ResNet-18 classifier labels captured panel images
  into six conditions: `Bird-drop`, `Clean`, `Dusty`, `Electrical damage`,
  `Physical Damage`, `Snow covered`.
- **Live camera capture** — streams a local webcam feed in the browser and periodically
  captures frames (configurable interval/duration) for fault classification.
- **CSV upload workflow** — upload inverter/weather datasets through the UI to run them
  through the anomaly-detection pipeline.

## Tech Stack

- **Backend**: Flask
- **ML / DL**: PyTorch, torchvision, scikit-learn
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Computer vision**: OpenCV

## Project Structure

```
.
├── app.py                 # Flask application and inference routes
├── train.py                # Training script for the LSTM autoencoder + MLP models
├── train.ipynb              # Notebook version of the training pipeline
├── requirements.txt
├── ip.txt                   # Host interface the app binds to on startup
├── models/                  # Pretrained model weights and the fitted scaler
│   ├── anlaomy_best_model.pth   # ResNet-18 panel fault classifier
│   ├── best_autoencoder.pth     # LSTM autoencoder (encoder/decoder)
│   ├── best_mlp_model.pth       # MLP power regressor
│   ├── best_model.pth           # CNN-LSTM yield prediction model
│   ├── data_best_model.pth
│   └── scaler.save              # Fitted MinMaxScaler used at inference time
├── static/
│   ├── assets/               # UI images and icons
│   ├── css/, js/              # Front-end styling and scripts
│   ├── files/                 # Sample datasets (inverter/weather CSVs), reference PDF, demo video
│   ├── images/                 # Captured camera frames
│   └── plots/                  # Generated result plots
└── templates/                # Jinja2 HTML templates
```

## Models

| Model | File | Purpose |
|---|---|---|
| CNN-LSTM | `models/best_model.pth` | Predicts daily energy yield per inverter from a sliding window of inverter + weather features. |
| LSTM Autoencoder | `models/best_autoencoder.pth` | Learns a compressed representation of recent sensor sequences, used as an engineered feature for anomaly scoring. |
| MLP Regressor | `models/best_mlp_model.pth` | Predicts expected DC power from module temperature, irradiation, and the autoencoder's encoding; prediction error drives anomaly flagging. |
| ResNet-18 | `models/anlaomy_best_model.pth` | Classifies captured panel images into 6 surface/fault conditions. |

`train.py` reproduces the LSTM autoencoder and MLP training pipeline (including the
early-stopping logic and the saved scaler); the ResNet-18 classifier and CNN-LSTM yield
model are expected to already be present under `models/`.

## Setup

**Requirements**: Python 3.9+, a webcam (optional, only needed for the live capture
feature).

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

The app reads the host/interface to bind to from `ip.txt` (defaults to `0.0.0.0` if the
file is missing) and serves on port `5000`.

## Usage

| Route | Method | Description |
|---|---|---|
| `/` | GET | Dashboard with sample inverter/weather data preview |
| `/upload` | GET/POST | Upload inverter + weather CSVs and run anomaly detection |
| `/test_model` | GET/POST | Upload datasets and evaluate the CNN-LSTM yield prediction model |
| `/camera` | GET | Live webcam preview page |
| `/video_feed` | GET | MJPEG video stream endpoint |
| `/start_capture` | POST | Start periodic frame capture (`interval`, `duration`) |
| `/capture_status` | GET | Poll whether capture is currently running |
| `/test_anomaly` | POST | Run the ResNet-18 classifier over captured images |
| `/delete_images` | POST | Clear captured images |

## Notes

- `app.secret_key` in `app.py` is a placeholder — replace it with a securely generated
  value before deploying this app anywhere beyond local use.
- `app.log`, `__pycache__/`, and OS/editor artifacts are excluded via `.gitignore`.
- Several files under `models/` and `static/files/` are large binary assets (trained
  weights, a sample video, a reference PDF); consider Git LFS if you plan to iterate on
  them frequently.
