import pandas as pd
from Functions_VS import eval_metrics
import matplotlib.pyplot as plt

df = pd.read_csv("Files/Cabrera.csv")
df = df.dropna(subset=['Temperature'])
df = df.iloc[:299]

rpi_onnx = pd.read_csv("rpi_onnx_17June.csv")
rpi_pth = pd.read_csv("rpi_pth_17June.csv")
##selecting only first 300 items
rpi_onnx = rpi_onnx.iloc[:299]
rpi_pth = rpi_pth.iloc[:299]

##DO
df_oxy = df["Oxygen"]
onnx_DO = rpi_onnx["Predicted_DO"]
pth_DO = rpi_pth["Predicted_DO"]

print("----------DO ONNX------------")
error_DO_onnx, DO_pred_onnx, DO_true_onnx, _, _, _, DO_MAE_onnx, _, _ = eval_metrics(df_oxy, onnx_DO)

print("---------DO .pth-----------")
error_DO_pth, DO_pred_pth, DO_true_pth, _, _, _, DO_MAE_pth, _, _ = eval_metrics(df_oxy, pth_DO)

###PH
df_PH = df["PH"]
onnx_PH = rpi_onnx["Predicted_pH"]
pth_PH = rpi_pth["Predicted_pH"]

print("-----------PH ONNX------------")
error_PH_onnx, PH_pred_onnx, PH_true_onnx, _, _, _, PH_MAE_onnx, _, _  = eval_metrics(df_PH, onnx_PH)

print("----------PH .pth------------")
error_PH_pth, PH_pred_pth, PH_true_pth, _, _, _, PH_MAE_pth, _, _= eval_metrics(df_PH, pth_PH)


#####----Figure plot---#####
# # Plot actual and predicted DO
# x = range(0,len(DO_pred_pth))
# fig, ax1 = plt.subplots(1,figsize=(8, 5))
# ax1.scatter(x, DO_true_pth, label="Actual DO", s=10, color='navy')
#
# ax1.scatter(x, DO_pred_pth, label="Predicted DO (.pth)", s=10, color='darkviolet')
# ax1.scatter(x, DO_pred_onnx, label="Predicted DO (.onnx)", s=10, color='cornflowerblue')
# ax1.set_xlabel("Predictions", fontsize=16)
# ax1.set_ylabel("DO", fontsize=16)
# ax1.set_xlim(0,320)
# ax1.legend(loc="upper left")
#
# # # Create secondary y-axis for error
# ax2 = ax1.twinx()
# # ONNX errors as arrows
# ax2.quiver(x, 0, [0]*len(x), error_DO_onnx, angles='xy', scale_units='xy',
#            scale=1, color='brown', label=f"Error (.onnx),\nMAE = {DO_MAE_onnx:.3f}", width=0.002)
# # PTH errors as arrows
# ax2.quiver(x, 0, [0]*len(x), error_DO_pth, angles='xy', scale_units='xy',
#            scale=1, color='orange', label=f"Error (.pth),\nMAE = {DO_MAE_pth:.3f}", width=0.002)
# ax2.set_ylim(-50, 55)
# ax2.set_ylabel("Prediction Error", fontsize=16)
# ax2.legend(bbox_to_anchor=(0, 0.8),loc="upper left")
# plt.xticks(fontsize=14)  # X-axis tick size
# ax1.tick_params(axis='y', labelsize=14)
# ax1.tick_params(axis='x', labelsize=14)
# ax2.tick_params(axis='y', labelsize=14)
#
# plt.title("Actual vs Predicted DO with Error", fontsize=16)
# plt.tight_layout()
# plt.xlim(0,250)
# plt.savefig("Errors_DO_Deploy")
# plt.show()
#
# ### Plot actual and predicted PH
x = range(0,len(df_PH))
x1 = range(0,len(PH_pred_pth))
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(x, df_PH, label="Actual PH", s=10, color='darkgreen')
ax1.scatter(x1, PH_pred_onnx, label="Predicted PH (.onnx)", s=10, color='lawngreen')
ax1.scatter(x1, PH_pred_pth, label="Predicted PH (.pth)", s=10, color='darkturquoise')
ax1.set_xlabel("Predictions", fontsize=16)
ax1.set_ylabel("PH", fontsize=16)
ax1.legend(loc="upper left")

# # Create secondary y-axis for error
ax2 = ax1.twinx()
# ONNX errors as arrows
ax2.quiver(x1, 0, [0]*len(x1), error_PH_onnx, angles='xy', scale_units='xy',
           scale=1, color='brown', label=f"Error (.onnx),\nMAE = {PH_MAE_onnx:.3f}", width=0.002)
# PTH errors as arrows
ax2.quiver(x1, 0, [0]*len(x1), error_PH_pth, angles='xy', scale_units='xy',
           scale=1, color='orange', label=f"Error (.pth),\nMAE = {PH_MAE_pth:.3f}", width=0.002)
ax2.set_ylim(-0.2, 0.2)
ax2.set_ylabel("Prediction Error", fontsize=16)
ax2.legend(bbox_to_anchor=(0, 0.8),loc="upper left")
plt.xticks(fontsize=14)  # X-axis tick size
ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
plt.title("Actual vs Predicted PH with Error", fontsize=16)
plt.tight_layout()
plt.xlim(0,250)
plt.savefig("Errors_PH_Deploy")
plt.show()