import pandas as pd
from Functions_VS import eval_metrics
import matplotlib.pyplot as plt
# Load and preprocess data
df = pd.read_csv("Files/Cabrera.csv")
# df= df.iloc[:299]
df = df.dropna(subset=['Temperature'])
df_ph = df["PH"]
df_oxy = df["Oxygen"]



rpi = pd.read_csv("rpi_onnx_19June.csv")
# rpi = rpi.iloc[:299]
rpi_ph = rpi["Predicted_pH"]
rpi_oxy = rpi["Predicted_DO"]
print("------------PH-------------")
eval_metrics(df_ph[:-1],rpi_ph)
print("-----------Oxygen------------")
eval_metrics(df_oxy[:-1],rpi_oxy)

