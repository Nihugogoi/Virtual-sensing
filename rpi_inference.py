import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load and preprocess data
rpi_onnx = pd.read_csv("rpi_onnx_19June.csv")
rpi_onnx = rpi_onnx[rpi_onnx["Predicted_DO"].notna()]
rpi_pth = pd.read_csv("rpi_pth_19June.csv")
rpi_pth = rpi_pth[rpi_pth["Oxygen"].notna()]
#Dropping first row because of booting the inference takes time than usual
rpi_onnx = rpi_onnx.iloc[2:].reset_index(drop=True)
rpi_pth= rpi_pth.iloc[2:].reset_index(drop=True)

####ONNX
mean_oxy_onnx = rpi_onnx["T_oxy"].mean()
std_oxy_onnx = rpi_onnx["T_oxy"].std()
print(f"Average Inference Time (DO): {mean_oxy_onnx:.2f} ± {std_oxy_onnx:.2f} ms (ONNX)")

mean_ph_onnx = rpi_onnx["T_pH"].mean()
std_ph_onnx = rpi_onnx["T_pH"].std()
print(f"Average Inference Time (pH): {mean_ph_onnx:.2f} ± {std_ph_onnx:.2f} ms (ONNX)")

###PYTORCH
mean_oxy_pth = rpi_pth["T_oxy"].mean()
std_oxy_pth = rpi_pth["T_oxy"].std()
print(f"Average Inference Time (DO): {mean_oxy_pth:.2f} ± {std_oxy_pth:.2f} ms (Pytorch)")

mean_ph_pth = rpi_pth["T_pH"].mean()
std_ph_pth = rpi_pth["T_pH"].std()
print(f"Average Inference Time (pH): {mean_ph_pth:.2f} ± {std_ph_pth:.2f} ms (Pytorch)")

# Data
labels = ['PH(.pth)','DO(.pth)', 'PH(.onnx)', 'DO(.onnx)']
colors = ['darkturquoise', 'darkviolet','lawngreen', 'cornflowerblue']
means = [float(f"{mean_ph_pth:.2f}"),   float(f"{mean_oxy_pth:.2f}"), float(f"{mean_ph_onnx:.2f}"), float(f"{mean_oxy_onnx:.2f}")]
errors = [float(f"{std_ph_pth:.2f}"),  float(f"{std_oxy_pth:.2f}"),float(f"{std_ph_onnx:.2f}"), float(f"{std_oxy_onnx:.2f}")]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, means, yerr=errors, capsize=3, color=colors, edgecolor='black')

# Add annotations for sums
sum_1 = means[0] + means[1]  # DO(.pth) + PH(.onnx)
sum_2 = means[2] + means[3]  # PH(.onnx) + DO(.onnx)

# Position above relevant bars
plt.text(
    0.5, max(means) + 2.5,
    f"Total inference time in\n.pth version is {sum_1:.2f} ms",
    ha='center', fontsize=10, color='black',
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
)

plt.text(
    2.5, max(means) - 15,
    f"Total inference time in\n.onnx version is {sum_2:.2f} ms",
    ha='center', fontsize=10, color='black',
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
)


plt.ylabel('Inference time (ms)', fontsize=13)
plt.title('Inference time of PH and DO', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,27)
plt.tight_layout()
plt.savefig('Figs/Inference_time.png')
plt.show()
