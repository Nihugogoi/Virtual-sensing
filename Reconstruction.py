import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from Functions_VS import TransformerModel
import numpy as np
import matplotlib.dates as mdates
# Load dataset
# df = pd.read_csv("Files/Cabrera_final.csv")
# df["Time"] = pd.to_datetime(df["Time"])
df = pd.read_csv("Files/Daniel2.csv")#daniel
df["Time"] = pd.to_datetime(df["Time"], errors="coerce", dayfirst=True)
df = df.drop_duplicates(subset='Time')
df = df.dropna(subset=['Time'])
data = df.drop(columns=["Time"])

# === Fit scalers ===
scaler_input = MinMaxScaler()
scaled_src = scaler_input.fit_transform(df[["Temperature"]])
Src_tensor = torch.tensor(scaled_src, dtype=torch.float32)

scaler_oxy = MinMaxScaler().fit(df[["Oxygen"]])
scaler_ph = MinMaxScaler().fit(df[["PH"]])

# === Model hyperparameters ===
input_size = 1
output_size = 1
nhead = 8
dropout = 0.15
max_seq_len = 1

# Oxygen model
d_model_oxy = 64
num_encoder_layers_oxy = 8
dim_feedforward_oxy = 128

# pH model
d_model_ph = 32
num_encoder_layers_ph = 6
dim_feedforward_ph = 32

# === Load models ===
model_oxy = TransformerModel(input_size, d_model_oxy, nhead, num_encoder_layers_oxy,
                             dim_feedforward_oxy, dropout, max_seq_len, output_size)
model_oxy.load_state_dict(torch.load("Models/pretrained_temp2oxy.pth", weights_only=True))
model_oxy.eval()

model_ph = TransformerModel(input_size, d_model_ph, nhead, num_encoder_layers_ph,
                            dim_feedforward_ph, dropout, max_seq_len, output_size)
model_ph.load_state_dict(torch.load("Models/pretrained_temp2ph.pth"))
model_ph.eval()

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_oxy.to(device)
model_ph.to(device)
Src_tensor = Src_tensor.to(device)

# === Prediction loop ===
oxy_preds = []
ph_preds = []

with torch.no_grad():
    for i in range(Src_tensor.shape[0]):
        input_seq = Src_tensor[i].unsqueeze(0)  # [1, seq_len, input_size]
        output_oxy = model_oxy(input_seq)
        output_ph = model_ph(input_seq)

        oxy_preds.append(output_oxy.cpu().numpy().squeeze())
        ph_preds.append(output_ph.cpu().numpy().squeeze())

# === Inverse transform predictions ===
oxy_preds = scaler_oxy.inverse_transform(pd.DataFrame(oxy_preds))
ph_preds = scaler_ph.inverse_transform(pd.DataFrame(ph_preds))

oxy_preds_flat = oxy_preds.flatten()
ph_preds_flat = ph_preds.flatten()
temperature = df['Temperature'].values

# Mask NaNs for oxygen correlation
mask_oxy = ~np.isnan(temperature) & ~np.isnan(oxy_preds_flat)
Corr_Oxygen = np.corrcoef(temperature[mask_oxy], oxy_preds_flat[mask_oxy])[0, 1]

# Mask NaNs for pH correlation
mask_ph = ~np.isnan(temperature) & ~np.isnan(ph_preds_flat)
Corr_PH = np.corrcoef(temperature[mask_ph], ph_preds_flat[mask_ph])[0, 1]

print("Correlation (Temperature vs Predicted DO):", Corr_Oxygen)
print("Correlation (Temperature vs Predicted pH):", Corr_PH)

###=== Plotting ===
plt.figure(figsize=(8, 8))
# date_format = mdates.DateFormatter('%b %Y')
# Temperature
plt.subplot(3, 1, 1)
plt .plot(df["Time"], df["Temperature"], color='orange')
plt.ylabel("Temperature (°C)", fontsize=13)
plt.title("Source: Temperature", fontsize=13)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(fontsize=11,rotation=30)
plt.yticks(fontsize=11)

# Oxygen
plt.subplot(3, 1, 2)
plt .plot(df["Time"], df["Oxygen"], label='Actual DO',color='navy')
plt .plot(df["Time"], oxy_preds, label='Predicted DO', color='cornflowerblue')
plt.ylabel("DO ($\mu$molkg$^{-1}$)", fontsize=13)
# plt.ylabel("DO (% saturation)", fontsize=13)
plt.title("Target: DO", fontsize=13)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.legend()
plt.xticks(fontsize=11,rotation=30)
plt.yticks(fontsize=11)

# pH
plt.subplot(3, 1, 3)
plt .plot(df["Time"], df["PH"], label='Actual pH', color='darkgreen')
plt .plot(df["Time"], ph_preds, label='Predicted pH', color='darkturquoise')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.ylabel("PH", fontsize=13)
plt.xlabel("Time", fontsize=13)
plt.title("Target: PH", fontsize=13)
plt.xticks(fontsize=11,rotation=30)
plt.yticks(fontsize=11)
plt.legend()

plt.tight_layout()
plt.savefig('Figs/fig_RDB_1.png')
plt.show()
