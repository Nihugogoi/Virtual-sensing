import pandas as pd
import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv


# Initialize MinMaxScaler
scaler = MinMaxScaler()
random_state =42
test_size = 0.1
window_size = 1


# ----------  Function to define model parameters ----------#
def parameters():
    input_size = 1
    d_model = 64# Embedding dimension for transformer
    nhead = 8 # Number of attention heads
    num_encoder_layers = 8 # Number of transformer encoder layers
    dim_feedforward = 2 * d_model  # Feedforward layer dimension
    dropout = 0.15 # Dropout rate
    learning_rate = 0.00001  # Learning rate for stability
    num_epochs = 1000
    max_seq_len = input_size*window_size
    output_size =1
    batch_size = 2
    return (input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, dropout, learning_rate,
            num_epochs, max_seq_len, batch_size)

# ---------- Transformer model ------------#
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_seq_len, target):
        super(TransformerModel, self).__init__()
        self.input = nn.Linear(input_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation = "relu")
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, target)
    def forward(self, src):
        seq_len = src.size(1)
        position_ids = torch.arange(seq_len, device=src.device).unsqueeze(0).repeat(src.size(0), 1)
        position_embeds = self.position_embeddings(position_ids)
        src = self.input(src) + position_embeds
        output = self.transformer_encoder(src)
        output = self.output_dropout(output[:, -1, :])
        output = self.output(output)
        return output


# ---------- Scale, Split and Preprocess Data ------------#
def scale_split_tensorloader(X, y, batch_size, test_size=test_size, random_state=random_state):
    # Scale data
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state)
    ### Concatenate X and y for saving
    train_df = pd.DataFrame(np.hstack([X_train, y_train]), columns=["Temperature", "PH"])
    test_df = pd.DataFrame(np.hstack([X_test, y_test]), columns=["Temperature", "PH"])

    ### Save to CSV (useful for future reconstruction)
    train_df.to_csv("Files/trainset_PH.csv", index=False)
    test_df.to_csv("Files/testset_PH.csv", index=False)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

# ---------- Train model ------------#
def train_and_save_model(X, y, model_filename, test_size=0.1, random_state=42):
    (input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size,
     dropout, learning_rate, num_epochs, max_seq_len, batch_size) = parameters()

    train_loader, test_loader = scale_split_tensorloader(X, y, batch_size, test_size, random_state)

    torch.manual_seed(5)
    model = TransformerModel(input_size, d_model, nhead, num_encoder_layers, dim_feedforward,
                             dropout, max_seq_len, output_size)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    total_params, size_in_MB = calculate_features(model)
    print(f"Total Parameters: {total_params}")
    print(f"Model Size: {size_in_MB:.4f} MB")

    train_losses = []
    val_losses = []

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    min_delta = 0.001
    wait = 0

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in test_loader:
                val_output = model(val_x)
                val_loss += criterion(val_output, val_y.squeeze(-1)).item()
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        # Logging
        if (epoch + 1)/2 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping check
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            wait = 0
            # Save best model temporarily
            torch.save(model.state_dict(), f'{model_filename}')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} — no improvement in val loss for {patience} epochs.")
                break

    # Load best model before returning
    model.load_state_dict(torch.load(f'{model_filename}'))
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")
    train_losses_np = np.array(train_losses)
    val_losses_np = np.array(val_losses)

    # Compute mean and standard deviation
    train_mean = np.mean(train_losses_np)
    train_std = np.std(train_losses_np)

    val_mean = np.mean(val_losses_np)
    val_std = np.std(val_losses_np)
    # ##Plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')

    # Get limits
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Place text in the top-left corner inside the plot
    plt.text(xlim[0] + 7.5, ylim[1] - (0.5 * (ylim[1])),
             'Training period is 38.25 ± 1.04 s ', fontsize=12)
    plt.text(xlim[0] + 7.5, ylim[1] - (0.575 * ylim[1]),
             f"Train Loss: {train_mean:.3f} ± {train_std:.3f}",
             fontsize=12, color='blue')

    plt.text(xlim[0] + 7.5, ylim[1] - (0.65 * ylim[1]),
             f"Val Loss: {val_mean:.3f} ± {val_std:.3f}",
             fontsize=12, color='orange')


    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("MSE Loss", fontsize=14)
    plt.title(f'MSE Loss of PH with Early Epoch at {epoch+1}', fontsize=14)
    plt.xlim(0,30)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_filename}.png')
    plt.show()

    # print(device)
    return model


# ---------- Count model parameters ------------#
def calculate_features(model):
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    param_size = sum(p.element_size() * p.numel() for p in model.parameters())  # Size in bytes
    size_in_MB = param_size / (1024 ** 2)  # Convert to MB
    return total_params, size_in_MB

# ---------- Evaluation of performannce metrics -------------#
def eval_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    error = y_true - y_pred
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)
    nrmse = rmse / (np.max(y_true) - np.min(y_true))  # Range normalization
    
    print(f"R² Score : {r2:.4f}")
    print(f"PCC      : {pcc:.4f}")
    print(f"NRMSE    : {nrmse:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    return  error, y_pred, y_true, r2,mse,rmse,mae, pcc, nrmse
    
 
# ---------- Run trained model and visualise-------------#
def run_and_plot(df,X,y, model_path, source, target):
    # Scale during training
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)

    # Load model with same structure
    (input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size,
     dropout, _, _, max_seq_len, _) = parameters()
    model = TransformerModel(input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_seq_len,
                             output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    # Inverse transform
    predictions_inv = scaler.inverse_transform(predictions)
    y_actual_inv = scaler.inverse_transform(y_scaled)
    error, y_pred, y_true, r2,mse,rmse,mae, pcc, nrmse = eval_metrics(y_actual_inv,predictions_inv)
    # --- Save metrics to csv ---
    eval_file = 'Files/PH_Palma.csv'
    fieldnames = [
        'target', 'd_model', 'nhead', 'num_encoder_layers', 'dim_feedforward',
        'r2', 'mse', 'rmse', 'mae', 'pcc', 'nrmse'
    ]
    write_header = not os.path.exists(eval_file) or os.stat(eval_file).st_size == 0

    with open(eval_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'target', 'd_model', 'nhead', 'num_encoder_layers',  'dim_feedforward',
                'r2', 'mse', 'rmse', 'mae', 'pcc', 'nrmse'
            ])
        writer.writerow([
            f'{target}', f'{d_model}', f'{nhead}', f'{num_encoder_layers}', f'{dim_feedforward}',
            f'{r2:.4f}', f'{mse:.4f}', f'{rmse:.4f}', f'{mae:.4f}', f'{pcc:.4f}', f'{nrmse:.4f}'
        ])

    #Plot
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(df["Time"], df[[source]], label=f'{source}', linewidth=2)
    plt.xlabel('Time',fontsize=16)
    plt.ylabel(source,fontsize=16)
    plt.xlim(df["Time"].min(), df["Time"].max())
    plt.title(f'{source}',fontsize=16)
    plt.legend()
    # plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.scatter(df["Time"], y_actual_inv, label='Actual', linewidth=2)
    plt.scatter(df["Time"], predictions_inv, label='Predicted', linewidth=2)
    plt.xlabel('Time',fontsize=16)
    plt.ylabel(target,fontsize=16)
    plt.title(f'Actual vs Predicted {target}',fontsize=16)
    plt.xlim(df["Time"].min(), df["Time"].max())
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{target}_{num_encoder_layers}_{d_model}_{nhead}_{dim_feedforward}_PH.png')
    plt.show()

    return predictions_inv




