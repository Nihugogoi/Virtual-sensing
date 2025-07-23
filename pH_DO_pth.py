import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import psutil
import time
from datetime import datetime
from Functions_VS import TransformerModel

# Load dataset
df = pd.read_csv("Files/Cabrera.csv")
df = df.dropna(subset=['Time'])
df["Time"] = pd.to_datetime(df["Time"])

# Drop time for processing
data = df.drop(columns=["Time"])

# === Fit scalers ===
# Input scaler for Temperature (used for both models)
scaler_input = MinMaxScaler()
scaled_src = scaler_input.fit_transform(df[["Temperature"]])
Src_tensor = torch.tensor(scaled_src, dtype=torch.float32)

# Output scalers for targets
scaler_oxy = MinMaxScaler()
scaler_oxy.fit(df[["Oxygen"]])

scaler_ph = MinMaxScaler()
scaler_ph.fit(df[["PH"]])

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

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_oxy.to(device)
model_ph.to(device)
Src_tensor = Src_tensor.to(device)


# === Optional: CPU temperature (Raspberry Pi only) ===
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except:
        return None

# === Logging ===
log_file = "Files/rpi_pth_pred.csv"
with open(log_file, "w") as f:
    f.write("Index,Timestamp,CPU_Usage(%),RAM_Usage(MB),CPU_Temp(C),T_oxy,Predicted_DO,T_pH,Predicted_pH\n")

# === Prediction Loop ===
for i in range(1, Src_tensor.shape[0]):
    input_seq = Src_tensor[i].unsqueeze(0)  # shape [1, seq_len, input_size]

    with torch.no_grad():
        start = time.time()
        output_oxy = model_oxy(input_seq)
        mid = time.time()
        output_ph = model_ph(input_seq)
        end = time.time()

    time_oxy = (mid - start) * 1000  # in ms
    time_ph = (end - mid) * 1000

    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
    cpu_temp = get_cpu_temperature()

    # Get predicted values
    oxy_scaled = output_oxy.cpu().numpy()
    ph_scaled = output_ph.cpu().numpy()

    oxy_actual = scaler_oxy.inverse_transform(oxy_scaled)[0][0]
    ph_actual = scaler_ph.inverse_transform(ph_scaled)[0][0]

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(
            f"{i},{current_time},{cpu_usage},{ram_usage:.2f},{cpu_temp},{time_oxy:.2f},{oxy_actual:.3f},{time_ph:.2f},{ph_actual:.3f}\n")

    print(
        f"[{current_time}] DO: {oxy_actual:.3f}, PH: {ph_actual:.3f}, CPU: {cpu_usage}%, RAM: {ram_usage:.2f}MB,Time_PH: {time_ph:.2f}ms, Time_oxy: {time_oxy:.2f}ms")

    time.sleep(10)  # Change to 30 if needed
