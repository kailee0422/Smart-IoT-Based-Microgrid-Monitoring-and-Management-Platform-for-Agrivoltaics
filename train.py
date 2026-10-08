import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import joblib

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
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Function to create sequences
def create_sequences(data, T):
    X, Y = [], []
    data_values = data.values
    for i in range(T, len(data_values)):
        X.append(data_values[i-T:i, :])
        Y.append(data_values[i, :])
    return np.array(X), np.array(Y)

def train_model(inverter_filepath, weather_filepath):

    # Process data
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

    # Split into train and test
    train_size = int(len(merged_df) * 0.7)
    train, _ = merged_df.iloc[:train_size], merged_df.iloc[train_size:]

    # Scale the features
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)

    # Save the scaler
    scaler_filename = 'models/scaler.save'
    joblib.dump(scaler, scaler_filename)

    # Convert scaled data back to DataFrame
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns, index=train.index)

    # Create sequences for LSTM
    T = 5  # Sequence length

    # Create training and validation sequences
    X_train_seq_full, Y_train_seq_full = create_sequences(train_scaled, T)

    # Split into training and validation sets (without shuffling to preserve time order)
    split_index = int(len(X_train_seq_full) * 0.9)
    X_train_seq, X_val_seq = X_train_seq_full[:split_index], X_train_seq_full[split_index:]
    Y_train_seq, Y_val_seq = Y_train_seq_full[:split_index], Y_train_seq_full[split_index:]

    # Define the LSTM Autoencoder model
    seq_len = T
    n_features = X_train_seq.shape[2]

    autoencoder = LSTM_Autoencoder(seq_len, n_features)
    criterion_ae = nn.MSELoss()
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.0001)

    # Prepare DataLoaders
    train_dataset_ae = SequenceDataset(X_train_seq)
    val_dataset_ae = SequenceDataset(X_val_seq)

    train_loader_ae = DataLoader(train_dataset_ae, batch_size=8, shuffle=True)
    val_loader_ae = DataLoader(val_dataset_ae, batch_size=8, shuffle=False)

    # Training loop with early stopping for LSTM Autoencoder
    epochs_ae = 100
    early_stopping_patience = 20
    best_loss_ae = np.inf
    patience_counter_ae = 0

    train_loss_history_ae = []
    val_loss_history_ae = []

    for epoch in range(epochs_ae):
        autoencoder.train()
        train_losses = []
        for seq in train_loader_ae:
            optimizer_ae.zero_grad()
            output = autoencoder(seq)
            loss = criterion_ae(output, seq)
            loss.backward()
            optimizer_ae.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        autoencoder.eval()
        val_losses = []
        with torch.no_grad():
            for seq in val_loader_ae:
                output = autoencoder(seq)
                loss = criterion_ae(output, seq)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        train_loss_history_ae.append(train_loss)
        val_loss_history_ae.append(val_loss)

        print(f"AE Epoch {epoch+1}/{epochs_ae}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_loss_ae:
            best_loss_ae = val_loss
            patience_counter_ae = 0
            # Save the model
            ae_model_path = 'models/best_autoencoder.pth'
            torch.save(autoencoder.state_dict(), ae_model_path)
        else:
            patience_counter_ae += 1
            if patience_counter_ae >= early_stopping_patience:
                print("Early stopping for Autoencoder")
                break

    # Plot training and validation loss for Autoencoder
    plt.figure(figsize=(10,5))
    plt.plot(train_loss_history_ae, label='Train Loss')
    plt.plot(val_loss_history_ae, label='Validation Loss')
    plt.title('LSTM Autoencoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('static/plots/ae_loss.png')
    plt.close()

    # Load the best Autoencoder model
    autoencoder.load_state_dict(torch.load(ae_model_path))
    autoencoder.eval()

    # Encode the full training data
    train_dataset_full = SequenceDataset(X_train_seq_full)
    train_loader_full = DataLoader(train_dataset_full, batch_size=64, shuffle=False)

    encoded_features_full = []

    with torch.no_grad():
        for seq in train_loader_full:
            encoding = autoencoder(seq, return_encoding=True)
            encoded_features_full.append(encoding.numpy())

    encoded_features_full = np.concatenate(encoded_features_full, axis=0)

    # Prepare data for MLP
    train1 = train_scaled.iloc[T:].copy().reset_index(drop=True)
    encoded_features_full = encoded_features_full.reshape(-1, 1)

    # Assign the encoded features to 'train1'
    train1['encoded'] = encoded_features_full

    # Define target and features
    Y1 = train1[['DC_POWER']].values
    X1 = train1[['MODULE_TEMPERATURE', 'IRRADIATION']].values

    # Concatenate encoded features
    X1 = np.concatenate([X1, encoded_features_full], axis=1)

    # Define the MLP model
    input_dim = X1.shape[1]
    mlp_model = MLP(input_dim)
    criterion_mlp = nn.L1Loss()  # MAE loss
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.0001)

    # Split into training and validation sets for MLP (without shuffling)
    split_index_mlp = int(len(X1) * 0.9)
    X_train_mlp, X_val_mlp = X1[:split_index_mlp], X1[split_index_mlp:]
    Y_train_mlp, Y_val_mlp = Y1[:split_index_mlp], Y1[split_index_mlp:]

    # Prepare DataLoaders for MLP
    train_dataset_mlp = TabularDataset(X_train_mlp, Y_train_mlp)
    val_dataset_mlp = TabularDataset(X_val_mlp, Y_val_mlp)

    train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=16, shuffle=True)
    val_loader_mlp = DataLoader(val_dataset_mlp, batch_size=16, shuffle=False)

    # Training loop with early stopping for MLP
    epochs_mlp = 100
    early_stopping_patience_mlp = 20
    best_loss_mlp = np.inf
    patience_counter_mlp = 0

    train_loss_history_mlp = []
    val_loss_history_mlp = []

    for epoch in range(epochs_mlp):
        mlp_model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader_mlp:
            optimizer_mlp.zero_grad()
            outputs = mlp_model(X_batch)
            loss = criterion_mlp(outputs, Y_batch)
            loss.backward()
            optimizer_mlp.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        mlp_model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader_mlp:
                outputs = mlp_model(X_batch)
                loss = criterion_mlp(outputs, Y_batch)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        train_loss_history_mlp.append(train_loss)
        val_loss_history_mlp.append(val_loss)

        print(f"MLP Epoch {epoch+1}/{epochs_mlp}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_loss_mlp:
            best_loss_mlp = val_loss
            patience_counter_mlp = 0
            # Save the model
            mlp_model_path = 'models/best_mlp_model.pth'
            torch.save(mlp_model.state_dict(), mlp_model_path)
        else:
            patience_counter_mlp += 1
            if patience_counter_mlp >= early_stopping_patience_mlp:
                print("Early stopping for MLP")
                break

    # Plot training and validation loss for MLP
    plt.figure(figsize=(10,5))
    plt.plot(train_loss_history_mlp, label='Train Loss')
    plt.plot(val_loss_history_mlp, label='Validation Loss')
    plt.title('MLP with LSTM Encoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('static/plots/mlp_loss.png')
    plt.close()

if __name__ == "__main__":
    # Paths to the CSV files
    
    inverter_filepath = '.\Data\20241120154511_Plant_1_Generation_Data.csv'
    weather_filepath = '.\Data\20241120154511_Plant_1_Weather_Sensor_Data.csv'

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)

    # Train the models
    train_model(inverter_filepath, weather_filepath)
