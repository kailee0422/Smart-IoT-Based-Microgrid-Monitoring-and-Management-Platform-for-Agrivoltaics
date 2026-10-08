# app.py

from flask import Flask, render_template, redirect, url_for, request, flash
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import logging
import joblib
import base64
from torchvision import models, transforms
from flask import Flask, render_template, redirect, url_for, request, flash, Response, jsonify
import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import datetime
import os
import threading
import time
import cv2
import atexit
import logging

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'  # Necessary for flashing messages

app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PLOTS_FOLDER'] = 'static/plots'
app.config['UPLOAD_IMGFOLDER'] = 'static/images'
app.config['DATASET_FOLDER'] = 'static/files'
# Ensure upload and plots directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_IMGFOLDER'], exist_ok=True)
os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)
app.config['UPLOAD_IMGFOLDER'] = 'static/images'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PLOTS_FOLDER'] = 'static/plots'

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

cap = cv2.VideoCapture(0)
# 导入 threading 模块
import threading

# 创建一个全局的线程锁
camera_lock = threading.Lock()
capturing_in_progress = False
capturing_lock = threading.Lock()

# Utility function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Template data function
def template(title="Taizhi", text="Taizhi has started"):
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    timeString = dt_string
    templateDate = {
        'title': title,
        'time': timeString,
        'text': text
    }
    return templateDate
#working main part
# @app.route("/")
# def home():
#     templateData = template(title="Taizhi", text="")
#     return render_template('main.html', **templateData)

import pandas as pd
@app.route("/")
def home():
    try:
        # Paths to the CSV files
        inverter_file = os.path.join(app.config['DATASET_FOLDER'], 'inverter_data.csv')
        weather_file = os.path.join(app.config['DATASET_FOLDER'], 'weather_data.csv')

        # Load CSV files
        inverter_df = pd.read_csv(inverter_file)
        weather_df = pd.read_csv(weather_file)

        # Extract headers and first 10 rows
        inverter_headers = inverter_df.columns.tolist()
        inverter_data = inverter_df.head(10).values.tolist()
        weather_headers = weather_df.columns.tolist()
        weather_data = weather_df.head(10).values.tolist()

        # Render main.html with template data and CSV data
        templateData = template(title="Taizhi", text="")
        return render_template(
            'main.html',
            **templateData,
            inverter_headers=inverter_headers,
            inverter_data=inverter_data,
            weather_headers=weather_headers,
            weather_data=weather_data
        )
    except Exception as e:
        return f"Error loading CSV files: {str(e)}", 500


# @app.route('/')
# def index():
#     try:
#         # Paths to the CSV files
#         inverter_file = os.path.join(app.config['DATASET_FOLDER'], 'inverter_data.csv')
#         weather_file = os.path.join(app.config['DATASET_FOLDER'], 'weather_data.csv')

#         # Load CSV files
#         inverter_df = pd.read_csv(inverter_file)
#         weather_df = pd.read_csv(weather_file)

#         # Extract headers and first 10 rows
#         inverter_headers = inverter_df.columns.tolist()
#         inverter_data = inverter_df.head(10).values.tolist()
#         weather_headers = weather_df.columns.tolist()
#         weather_data = weather_df.head(10).values.tolist()

#         print("Inverter Headers:", inverter_headers)
#         print("Inverter Data (First 10 Rows):", inverter_data)
#         print("Weather Headers:", weather_headers)
#         print("Weather Data (First 10 Rows):", weather_data)

#         # Render main.html with data
#         return render_template(
#             'main.html',
#             inverter_headers=inverter_headers,
#             inverter_data=inverter_data,
#             weather_headers=weather_headers,
#             weather_data=weather_data
#         )
#     except Exception as e:
#         return f"Error loading CSV files: {str(e)}", 500

@app.route("/camera", methods=['GET', 'POST'])
def camera():
    templateData = template(title="Taizhi", text="")
    return render_template("camera.html", **templateData)

# 视频流生成器
def gen_frames():
    while True:
        with camera_lock:
            success, frame = cap.read()
        if not success:
            break
        else:
            # 将图像编码为 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # 使用 multipart/x-mixed-replace 协议传输视频流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 视频流路由
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_images(interval, duration):
    if not cap.isOpened():
        print("Cannot open camera")
        return

    start_time = time.time()
    end_time = start_time + duration
    count = 0

    global capturing_in_progress
    with capturing_lock:
        capturing_in_progress = True

    save_dir = app.config['UPLOAD_IMGFOLDER']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while time.time() < end_time:
        with camera_lock:
            ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        filename = f'capture_{count}.png'
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved {filepath}")
        count += 1
        time.sleep(interval)

    print("Image capturing completed.")
    with capturing_lock:
        capturing_in_progress = False

@app.route('/capture_status')
def capture_status():
    with capturing_lock:
        status = capturing_in_progress
    return jsonify({'capturing': status})

# 开始捕获图像的路由
@app.route('/start_capture', methods=['POST'])
def start_capture():
    data = request.get_json()
    try:
        interval = float(data.get('interval', 0))
        duration = float(data.get('duration', 0))

        if interval <= 0 or duration <= 0:
            return jsonify({'message': 'Please provide valid positive numbers for both fields.'}), 400

        threading.Thread(target=capture_images, args=(interval, duration)).start()
        return jsonify({'message': 'Image capturing started.'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'message': 'An error occurred while starting image capture.'}), 500

# 应用关闭时释放摄像头资源
@atexit.register
def cleanup():
    global cap
    cap.release()
    cv2.destroyAllWindows()

# New route for uploading CSV files
@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    templateData = template(title="Taizhi", text="")
    if request.method == 'POST':
        # Check if both file parts are present in the request
        if 'inverter_file' not in request.files or 'weather_file' not in request.files:
            flash('Both Inverter and Weather CSV files are required.')
            logging.warning("Upload attempted without both files.")
            return redirect(request.url)
        
        inverter_file = request.files['inverter_file']
        weather_file = request.files['weather_file']
        
        # Check if files have been selected
        if inverter_file.filename == '' or weather_file.filename == '':
            flash('No file selected for one or both fields.')
            logging.warning("One or both files not selected.")
            return redirect(request.url)
        
        # Validate file extensions
        if not (allowed_file(inverter_file.filename) and allowed_file(weather_file.filename)):
            flash('Allowed file types are CSV.')
            logging.warning("Uploaded files with invalid extensions.")
            return redirect(request.url)
        
        # Secure the filenames
        inverter_filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S_") + secure_filename(inverter_file.filename)
        weather_filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S_") + secure_filename(weather_file.filename)
        
        # Save the files to the upload folder
        inverter_filepath = os.path.join(app.config['UPLOAD_FOLDER'], inverter_filename)
        weather_filepath = os.path.join(app.config['UPLOAD_FOLDER'], weather_filename)
        inverter_file.save(inverter_filepath)
        weather_file.save(weather_filepath)
        
        flash('Files successfully uploaded.')
        logging.info(f"Files uploaded: {inverter_filename}, {weather_filename}")
        
        try:
            # Process the uploaded files
            results = process_csv(inverter_filepath, weather_filepath)
            logging.info("File processing completed successfully.")
            return render_template('results.html', **results)
        except Exception as e:
            # Handle exceptions raised during processing
            flash(f"Error processing files: {e}")
            logging.error(f"Error processing files: {e}")
            return redirect(request.url)
    
    return render_template('upload.html', **templateData)

#new route for anomaly detection
@app.route("/test_anomaly", methods=['POST'])
def test_anomaly():
    templateData = template(title="Taizhi", text="")
    import glob
    from torchvision import transforms
    from PIL import Image

    # Folder containing captured images
    image_folder = app.config['UPLOAD_IMGFOLDER']

    # Check if the folder contains images
    image_paths = glob.glob(os.path.join(image_folder, "*.png"))
    if not image_paths:
        return "No images found for testing.", 400

    # Load the trained model
    model_path = './models/anlaomy_best_model.pth'
    model = models.resnet18(pretrained=False)
    num_classes = 6
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Class names
    class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical damage', 'Physical Damage', 'Snow covered']

    # Process each image and get predictions
    predictions = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = class_names[predicted.item()]

        # Append file path and prediction
        predictions.append((os.path.basename(image_path), image_path, predicted_class))

    # Return results
    return render_template("anomaly_results.html", predictions=predictions,**templateData)

@app.route('/delete_images', methods=['POST'])
def delete_images():
    try:
        # Folder containing captured images
        image_folder = app.config['UPLOAD_IMGFOLDER']

        # Get all images in the folder
        image_paths = glob.glob(os.path.join(image_folder, "*.png"))

        # Delete each image
        for image_path in image_paths:
            os.remove(image_path)

        logging.info("All images deleted successfully.")
        return jsonify({'message': 'Images deleted successfully.'}), 200
    except Exception as e:
        logging.error(f"Error deleting images: {e}")
        return jsonify({'message': f'Error deleting images: {e}'}), 500
    
class CNNLSTMModel(nn.Module):
    def __init__(self, input_shape, num_inverters):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.dense1 = nn.Linear(50, 50)
        self.dense2 = nn.Linear(50, num_inverters)
    
    def forward(self, x):
        x = self.conv1d(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.batchnorm(x)
        x = x.permute(0, 2, 1)  # Change shape for LSTM
        x, _ = self.lstm(x)
        x = self.dropout2(x[:, -1, :])  # Last time step
        x = torch.relu(self.dense1(x))
        x = self.dense2(x)
        return x



# Plot and save results
def plot_results(y_actual, y_predicted, save_path):
    plt.figure(figsize=(20, 30))
    for i in range(y_actual.shape[1]):
        plt.subplot(11, 2, i + 1)
        plt.plot(y_actual[:, i], label="Actual", color='orange')
        plt.plot(y_predicted[:, i], label="Predicted", linestyle='dashed', color='blue')
        plt.title(f"Inverter {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("DAILY_YIELD (kWh)")
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Flask route for testing the model
@app.route("/test_model", methods=["GET", "POST"])
def test_model():
    templateData = template(title="Taizhi", text="")
    if request.method == "POST":
        try:
            # Verify both datasets are uploaded
            if 'inverter_dataset' not in request.files or 'weather_dataset' not in request.files:
                flash("Please upload both inverter and weather datasets.")
                logging.error("Datasets not uploaded.")
                return redirect(request.url)

            # Save uploaded files
            inverter_file = request.files['inverter_dataset']
            weather_file = request.files['weather_dataset']
            inverter_path = os.path.join(app.config['UPLOAD_FOLDER'], inverter_file.filename)
            weather_path = os.path.join(app.config['UPLOAD_FOLDER'], weather_file.filename)
            inverter_file.save(inverter_path)
            weather_file.save(weather_path)

            # Load datasets
            inverter_data = pd.read_csv(inverter_path)
            weather_data = pd.read_csv(weather_path)

            # Preprocess datasets
            inverter_data.columns = inverter_data.columns.str.strip()
            weather_data.columns = weather_data.columns.str.strip()

            inverter_data['DATE_TIME'] = pd.to_datetime(inverter_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
            weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
            weather_data.rename(columns={'SOURCE_KEY': 'WEATHER_SOURCE_KEY'}, inplace=True)
            # Merge the data
            merged_data = pd.merge(inverter_data, weather_data, on='DATE_TIME', how='outer')
            # Fill missing data
            merged_data = merged_data.ffill().bfill()
            merged_data['DATE'] = merged_data['DATE_TIME'].dt.date
            # Pivot the data
            pivot_data = merged_data.pivot(index='DATE_TIME', columns='SOURCE_KEY', values='DAILY_YIELD')
            pivot_data.fillna(0, inplace=True)
            pivot_data.reset_index(inplace=True)
            # Prepare the input features
            input_features = merged_data[['DATE_TIME', 'DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
            # Merge input features with pivoted data
            input_data = pd.merge(input_features, pivot_data, on='DATE_TIME', how='inner')
            # Split data into train, validation, and test sets (7:2:1)
            # 將70%分給train_data, 30%給test_data
            train_data, test_data = train_test_split(input_data, test_size=0.3, shuffle=False)
            # 再將2/3%分給val_data, 1/3%給test_data
            val_data, test_data = train_test_split(test_data, test_size=0.34, shuffle=False)
            # Function to create dataset
            def create_dataset(data, window_size=5):
                X, y = [], []
                data_values = data.drop(columns=['DATE_TIME']).values  # Exclude 'DATE_TIME'
                for i in range(len(data_values) - window_size):
                    X.append(data_values[i:i + window_size])  # Sequential input data
                    y.append(data_values[i + window_size, -len(pivot_data.columns) + 1:])  # Target values
                    # y.append(data_values[i + window_size, -22:])  # Target (last 22 columns for inverters)
                return np.array(X), np.array(y)

            # Create dataset
            window_size = 5
            X_train, y_train = create_dataset(train_data, window_size)
            X_test, y_test = create_dataset(test_data, window_size)

            # Standardize data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
            X_test = scaler_X.fit_transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
            y_test = scaler_y.fit_transform(y_test)

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

            # Load model
            model_path = './models/best_model.pth'
            model = CNNLSTMModel(input_shape=X_train.shape, num_inverters=22)
            #model = CNNLSTMModel(input_shape=(window_size, X_test.shape[2]), num_inverters=y_test.shape[1])
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Run predictions
            with torch.no_grad():
                predictions = model(X_test_tensor)
            predictions = scaler_y.inverse_transform(predictions.numpy())
            y_test_original = scaler_y.inverse_transform(y_test)

            # Plot results
            plot_path = os.path.join(app.config['PLOTS_FOLDER'], 'Yeild_test_results.png')
            plot_results(y_test_original, predictions, plot_path)

            # Calculate metrics
            mse = mean_squared_error(y_test_original.flatten(), predictions.flatten())
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_original.flatten(), predictions.flatten())
            r2 = r2_score(y_test_original.flatten(), predictions.flatten())

            return render_template(
                "test_results.html",
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2=r2,
                plot_url=url_for('static', filename=f'plots/Yeild_test_results.png'),
                **templateData
            )

        except Exception as e:
            flash(f"Error: {str(e)}")
            logging.error(f"Error during testing: {str(e)}")
            return redirect(request.url)

    return render_template("upload_datasets.html",**templateData)
# Define the LSTM Autoencoder model
class LSTM_Autoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(LSTM_Autoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        # Encoder
        self.encoder_l1 = nn.LSTM(input_size=n_features, hidden_size=16, num_layers=1, batch_first=True)
        self.encoder_dropout1 = nn.Dropout(0.2)
        self.encoder_l2 = nn.LSTM(input_size=16, hidden_size=4, num_layers=1, batch_first=True)
        self.encoder_dropout2 = nn.Dropout(0.2)
        self.encoder_l3 = nn.LSTM(input_size=4, hidden_size=1, num_layers=1, batch_first=True)
        self.encoder_dropout3 = nn.Dropout(0.2)

        # Decoder
        self.decoder_l1 = nn.LSTM(input_size=1, hidden_size=4, num_layers=1, batch_first=True)
        self.decoder_dropout1 = nn.Dropout(0.2)
        self.decoder_l2 = nn.LSTM(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
        self.decoder_dropout2 = nn.Dropout(0.2)
        self.decoder_l3 = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
        self.decoder_dropout3 = nn.Dropout(0.2)

        self.output_layer = nn.Linear(16, n_features)

    def forward(self, x, return_encoding=False):
        # Encoder
        x, _ = self.encoder_l1(x)
        x = self.encoder_dropout1(x)
        x, _ = self.encoder_l2(x)
        x = self.encoder_dropout2(x)
        x, _ = self.encoder_l3(x)
        x = self.encoder_dropout3(x)

        # Get the encoding
        encoding = x[:, -1, :]  # Shape: [batch_size, hidden_size]

        if return_encoding:
            return encoding

        # Prepare for decoder
        x = x[:, -1, :].unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decoder
        x, _ = self.decoder_l1(x)
        x = self.decoder_dropout1(x)
        x, _ = self.decoder_l2(x)
        x = self.decoder_dropout2(x)
        x, _ = self.decoder_l3(x)
        x = self.decoder_dropout3(x)

        # Output layer
        x = self.output_layer(x)
        return x

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        return self.layers(x)

# Dataset classes
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)

class TabularDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Function to process the uploaded CSV files
def process_csv(inverter_filepath, weather_filepath):
    try:
        # Load Inverter Data
        inverter_df = pd.read_csv(inverter_filepath)
    except Exception as e:
        raise Exception(f"Error reading Inverter CSV file: {e}")

    # Check for 'DATE_TIME' column in Inverter CSV
    if 'DATE_TIME' not in inverter_df.columns:
        raise Exception("Inverter CSV file must contain 'DATE_TIME' column.")

    # Parse 'DATE_TIME' in Inverter CSV
    try:
        inverter_df['DATE_TIME'] = pd.to_datetime(
            inverter_df['DATE_TIME'],
            format='%d-%m-%Y %H:%M'
        )
    except Exception as e:
        raise Exception(f"Error parsing 'DATE_TIME' column in Inverter CSV: {e}")

    # Filter data for the first inverter
    inverters = inverter_df['SOURCE_KEY'].unique()
    if len(inverters) == 0:
        raise Exception("No inverters found in the Inverter CSV data.")
    inv_1 = inverter_df[inverter_df['SOURCE_KEY'] == inverters[0]]

    try:
        # Load Weather Data
        weather_df = pd.read_csv(weather_filepath)
    except Exception as e:
        raise Exception(f"Error reading Weather CSV file: {e}")

    # Check for 'DATE_TIME' column in Weather CSV
    if 'DATE_TIME' not in weather_df.columns:
        raise Exception("Weather CSV file must contain 'DATE_TIME' column.")

    # Parse 'DATE_TIME' in Weather CSV
    try:
        weather_df['DATE_TIME'] = pd.to_datetime(
            weather_df['DATE_TIME'],
            format='%Y-%m-%d %H:%M:%S'
        )
    except Exception as e:
        raise Exception(f"Error parsing 'DATE_TIME' column in Weather CSV: {e}")

    # Merge Inverter and Weather Data on 'DATE_TIME'
    merged_df = inv_1.merge(weather_df, on="DATE_TIME", how='left')

    # Select relevant columns
    required_columns = [
        'DATE_TIME', 'DC_POWER',
        'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION'
    ]
    missing_cols = [col for col in required_columns if col not in merged_df.columns]
    if missing_cols:
        raise Exception(f"Missing columns in CSV: {', '.join(missing_cols)}")

    merged_df = merged_df[required_columns]

    # Handle missing values
    merged_df.fillna(method='ffill', inplace=True)  # Forward fill
    merged_df.bfill(inplace=True)                   # Backward fill

    # Set 'DATE_TIME' as index
    merged_df.set_index('DATE_TIME', inplace=True)

    # Load the scaler
    scaler_filename = 'models/scaler.save'
    if not os.path.exists(scaler_filename):
        raise Exception("Scaler file not found. Please train the model first.")
    scaler = joblib.load(scaler_filename)

    # Scale the data
    data_scaled = scaler.transform(merged_df)
    data_scaled = pd.DataFrame(data_scaled, columns=merged_df.columns, index=merged_df.index)

    # Create sequences for LSTM
    T = 5  # Sequence length

    def create_sequences(data, T):
        X = []
        data_values = data.values
        for i in range(T, len(data_values)):
            X.append(data_values[i-T:i, :])
        return np.array(X)

    # Prepare sequences for encoding
    X_seq = create_sequences(data_scaled, T)
    data_seq_index = data_scaled.index[T:]  # Corresponding timestamps

    # Load the trained Autoencoder
    n_features = X_seq.shape[2]
    seq_len = T
    autoencoder = LSTM_Autoencoder(seq_len, n_features)
    ae_model_path = 'models/best_autoencoder.pth'
    if not os.path.exists(ae_model_path):
        raise Exception("Autoencoder model file not found. Please train the model first.")
    autoencoder.load_state_dict(torch.load(ae_model_path))
    autoencoder.eval()

    # Encode data
    dataset_seq = SequenceDataset(X_seq)
    loader_seq = DataLoader(dataset_seq, batch_size=64, shuffle=False)

    encoded_features = []

    with torch.no_grad():
        for seq in loader_seq:
            encoding = autoencoder(seq, return_encoding=True)
            encoded_features.append(encoding.numpy())

    encoded_features = np.concatenate(encoded_features, axis=0)

    # Prepare data for MLP prediction
    data_prepared = data_scaled.iloc[T:].copy()
    data_prepared['encoded'] = encoded_features
    Y = data_prepared[['DC_POWER']].values
    X = data_prepared[['MODULE_TEMPERATURE', 'IRRADIATION']].values
    encoded_features = encoded_features.reshape(-1, 1)
    X = np.concatenate([X, encoded_features], axis=1)

    # Load the trained MLP model
    input_dim = X.shape[1]
    mlp_model = MLP(input_dim)
    mlp_model_path = 'models/best_mlp_model.pth'
    if not os.path.exists(mlp_model_path):
        raise Exception("MLP model file not found. Please train the model first.")
    mlp_model.load_state_dict(torch.load(mlp_model_path))
    mlp_model.eval()

    # Create DataLoader for data
    dataset_mlp = TabularDataset(X, Y)
    loader_mlp = DataLoader(dataset_mlp, batch_size=64, shuffle=False)

    # Make predictions
    preds = []

    with torch.no_grad():
        for X_batch, _ in loader_mlp:
            outputs = mlp_model(X_batch)
            preds.append(outputs.numpy())
    preds = np.concatenate(preds, axis=0)
    preds = preds.reshape(-1)

    # Add predictions to data_prepared
    data_prepared['preds'] = preds

    # Calculate anomaly scores
    scores = data_prepared.copy()
    scores['loss_mae'] = np.abs(scores['DC_POWER'] - scores['preds'])

    # Three sigma method for threshold
    mean_loss = scores['loss_mae'].mean()
    std_loss = scores['loss_mae'].std()
    lower_bound = mean_loss - 3 * std_loss
    upper_bound = mean_loss + 3 * std_loss
    scores['Threshold'] = upper_bound
    scores['Anomaly'] = ((scores['loss_mae'] < lower_bound) | (scores['loss_mae'] > upper_bound)).astype(int)

    # Generate plots for results

    # 1. Error Distribution
    plt.figure(figsize=(10,5))
    sns.histplot(scores['loss_mae'], bins=50, kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Error delta between predicted and real data [DC Power]')
    plt.ylabel('Data point counts')
    error_dist_plot_path = os.path.join(app.config['PLOTS_FOLDER'], 'error_distribution.png')
    plt.savefig(error_dist_plot_path)
    plt.close()

    # 2. Error Time Series and Threshold
    plt.figure(figsize=(10,5))
    plt.plot(scores.index, scores['loss_mae'], label='Loss')
    plt.plot(scores.index, scores['Threshold'], label='Threshold', linestyle='--')
    plt.title('Error Timeseries and Threshold')
    plt.xlabel('DateTime')
    plt.ylabel('Loss')
    plt.legend()
    error_timeseries_plot_path = os.path.join(app.config['PLOTS_FOLDER'], 'error_timeseries.png')
    plt.savefig(error_timeseries_plot_path)
    plt.close()

    # 3. Anomalies Detected
    anomalies = scores[scores['Anomaly'] == 1]
    plt.figure(figsize=(15,5))
    plt.plot(scores.index, scores['DC_POWER'], label='DC Power')
    plt.scatter(anomalies.index, anomalies['DC_POWER'], color='red', label='Anomaly')
    plt.title('Anomalies Detected MLP with LSTM Encoder')
    plt.xlabel('Time')
    plt.ylabel('DC Power')
    plt.legend()
    anomalies_plot_path = os.path.join(app.config['PLOTS_FOLDER'], 'anomalies.png')
    plt.savefig(anomalies_plot_path)
    plt.close()

    # 4. Real vs Predicted DC Power
    plt.figure(figsize=(15,5))
    plt.plot(scores.index, scores['DC_POWER'], color='red', label='Real')
    plt.plot(scores.index, scores['preds'], color='blue', label='Predicted')
    plt.title('DC Power Prediction - MLP with LSTM Encoder')
    plt.xticks(rotation=70)
    plt.xlabel('Time')
    plt.ylabel('DC Power (kW)')
    plt.legend()
    real_vs_pred_plot_path = os.path.join(app.config['PLOTS_FOLDER'], 'real_vs_pred.png')
    plt.savefig(real_vs_pred_plot_path)
    plt.close()

    # Compile results to pass to the template
    results = {
        'error_distribution_plot': url_for('static', filename='plots/error_distribution.png'),
        'error_timeseries_plot': url_for('static', filename='plots/error_timeseries.png'),
        'anomalies_plot': url_for('static', filename='plots/anomalies.png'),
        'real_vs_pred_plot': url_for('static', filename='plots/real_vs_pred.png'),
        'anomalies_count': int(scores['Anomaly'].sum()),
        'total_points': int(len(scores)),
        'anomalies_percentage': round((scores['Anomaly'].sum() / len(scores)) * 100, 2)
    }

    return results




# rout eto display time
@app.route("/time")
def get_time():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"time": current_time}
# Existing routes continued...

# Read IP from 'ip.txt' and run the app
if __name__ == "__main__":
    ip_v = '0.0.0.0'  # Default to all interfaces
    if os.path.exists('ip.txt'):
        with open('ip.txt', 'r') as file:
            ip_v = file.read().strip()
            logging.info(f"Server will run on IP: {ip_v}")
    else:
        logging.warning("ip.txt not found. Using default IP '0.0.0.0'")
    app.run(host=ip_v, port=5000, debug=True)
