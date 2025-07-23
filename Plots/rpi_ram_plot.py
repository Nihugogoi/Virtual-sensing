import pandas as pd
import matplotlib.pyplot as plt
# Load and preprocess data
rpi_onnx = pd.read_csv("rpi_onnx_17June.csv")
rpi_pth = pd.read_csv("rpi_pth_17June.csv")
##selecting only first 300 items
rpi_onnx = rpi_onnx.iloc[:299]
rpi_pth = rpi_pth.iloc[:299]

#Dropping first row because of booting the inference takes time than usual
rpi_onnx = rpi_onnx.iloc[2:].reset_index(drop=True)
rpi_pth= rpi_pth.iloc[2:].reset_index(drop=True)

valid_mask = ~rpi_onnx["Predicted_DO"].isna()

# Apply the mask to all relevant data
pred_DO = rpi_onnx["Predicted_DO"][valid_mask]
RAM_pth = rpi_pth["RAM_Usage(MB)"][valid_mask]
RAM_onnx = rpi_onnx["RAM_Usage(MB)"][valid_mask]
x = range(0, len(pred_DO))

mean_oxy_onnx = RAM_onnx.mean()
std_oxy_onnx = RAM_onnx.std()

mean_oxy_pth = RAM_pth.mean()
std_oxy_pth = RAM_pth.std()

ram_onnx_text = f": {mean_oxy_onnx:.2f} ± {std_oxy_onnx:.2f} MB"
ram_pth_text = f": {mean_oxy_pth:.2f} ± {std_oxy_pth:.2f} MB"
print(ram_onnx_text, ram_pth_text)

labels = ['.pth', '.onnx']
colors = ['mistyrose', 'sienna']
means = [float(f"{mean_oxy_onnx:.2f}"),  float(f"{mean_oxy_pth:.2f}")]
errors = [float(f"{std_oxy_onnx:.2f}"),  float(f"{std_oxy_pth:.2f}")]
# Plot
plt.figure(figsize=(5,5))
bars = plt.bar(labels, means, yerr=errors, capsize=3, color=colors, edgecolor='black')#, width=0.5)
plt.ylabel("RAM Usage (MB)",fontsize=15)
plt.title("Memory Usage of .pth and .onnx models",fontsize=15)
plt.xticks(fontsize=15)  # X-axis tick size
plt.yticks(fontsize=15)  # Y-axis tick size

plt.tight_layout()
plt.savefig("RAM_DOPH.png")
plt.show()

