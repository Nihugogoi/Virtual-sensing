import pandas as pd
import matplotlib.pyplot as plt
# Load and preprocess data
rpi_onnx = pd.read_csv("rpi_onnx_19June.csv")
rpi_pth = pd.read_csv("rpi_pth_19June.csv")
rpi_onnx = rpi_onnx.iloc[:299]
rpi_pth = rpi_pth.iloc[:299]

#Dropping first row because of booting the inference takes time than usual
rpi_onnx = rpi_onnx.iloc[2:].reset_index(drop=True)
rpi_pth= rpi_pth.iloc[2:].reset_index(drop=True)

valid_mask = ~rpi_onnx["Predicted_DO"].isna()

# Apply the mask to all relevant data
pred_DO_onnx = rpi_onnx["Predicted_DO"][valid_mask]
T_DO_onnx = rpi_onnx["T_oxy"][valid_mask]
T_pH_onnx = rpi_onnx["T_pH"][valid_mask]
T_DO_pth = rpi_pth["T_oxy"][valid_mask]
T_pH_pth = rpi_pth["T_pH"][valid_mask]
x = range(0, len(pred_DO_onnx))
# Plot
plt.figure(figsize=(8,5))
plt.scatter(x, T_DO_onnx, label="T_DO (.onnx)", color='darkred',s=20)
plt.scatter(x, T_DO_pth, label="T_DO (.pth)", color='rosybrown',s=20)
plt.scatter(x, T_pH_onnx, label="T_PH (.onnx)", color='salmon',s=20)
plt.scatter(x, T_pH_pth, label="T_PH (.pth)", color='darkorchid', s=20)
plt.xlabel("Predictions",fontsize=16)
plt.ylabel("Inference (ms)",fontsize=16)
plt.title("Inference of PH and DO",fontsize=16)
plt.xticks(fontsize=14)  # X-axis tick size
plt.yticks(fontsize=14)
plt.xlim(0,320)
plt.legend(loc ='upper right')
# plt.grid(True)
plt.tight_layout()
plt.savefig("InferenceDOPH.png")
plt.show()

