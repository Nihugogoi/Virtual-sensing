import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
import psutil
import time
from datetime import datetime

# Load dataset
df = pd.read_csv("Files/Cabrera.csv")
df = df.dropna(subset=['Time'])
df["Time"] = pd.to_datetime(df["Time"])
data = df.drop(columns=["Time"])

# === Fit scalers ===
scaler_input = MinMaxScaler()
scaled_src = scaler_input.fit_transform(df[["Temperature"]])

scaler_oxy = MinMaxScaler()
scaler_oxy.fit(df[["Oxygen"]])

scaler_ph = MinMaxScaler()
scaler_ph.fit(df[["PH"]])

# === Load ONNX models ===
session_oxy = ort.InferenceSession("Models/temp2oxy.onnx")
session_ph = ort.InferenceSession("Models/temp2ph.onnx")

input_name_oxy = session_oxy.get_inputs()[0].name
input_name_ph = session_ph.get_inputs()[0].name

# === Optional: CPU temperature (Raspberry Pi only) ===
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except:
        return None

# === Logging ===
log_file = "Files/rpi_onnx_pred.csv"
with open(log_file, "w") as f:
    f.write("Index,Timestamp,CPU_Usage(%),RAM_Usage(MB),CPU_Temp(C),T_oxy,Predicted_DO,T_pH,Predicted_pH\n")

# === Prediction Loop ===
for i in range(1, 300):
    input_seq = scaled_src[i].reshape(1, 1, 1).astype(np.float32)

    start = time.time()
    output_oxy = session_oxy.run(None, {input_name_oxy: input_seq})[0]
    mid = time.time()
    output_ph = session_ph.run(None, {input_name_ph: input_seq})[0]
    end = time.time()

    time_oxy = (mid - start) * 1000  # in ms
    time_ph = (end - mid) * 1000

    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
    cpu_temp = get_cpu_temperature()

    oxy_actual = scaler_oxy.inverse_transform(output_oxy)[0][0]
    ph_actual = scaler_ph.inverse_transform(output_ph)[0][0]

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(
            f"{i},{current_time},{cpu_usage},{ram_usage:.2f},{cpu_temp},{time_oxy:.2f},{oxy_actual:.3f},{time_ph:.2f},{ph_actual:.3f}\n")

    print(
        f"[{current_time}] DO: {oxy_actual:.3f}, PH: {ph_actual:.3f}, CPU: {cpu_usage}%, RAM: {ram_usage:.2f}MB, Time_DO: {time_oxy:.2f}ms, Time_PH: {time_ph:.2f}ms")

    time.sleep(10)  # Change to 30 if needed
