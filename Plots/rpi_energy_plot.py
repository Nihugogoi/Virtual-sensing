import matplotlib.pyplot as plt

# Energy per inference in mJ
energy_values = [26.98 , 41.45 ,  4.24, 6.54]
labels = ['PH(.pth)','DO(.pth)', 'PH(.onnx)', 'DO(.onnx)']
colors = ['darkturquoise', 'darkviolet','lawngreen', 'cornflowerblue']
errors = [0.0141,0.0021,0.0141,0.0021]

# Plot
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, energy_values, yerr=errors, capsize=3, color=colors, edgecolor='black')

# Labeling
plt.ylabel('Energy per Inference (mJ)', fontsize=13)
plt.title('Energy Consumption per Inference', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,60)
# Add annotations for sums
sum_1 = energy_values[0] + energy_values[1]  # DO(.pth) + PH(.onnx)
sum_2 = energy_values[2] + energy_values[3]  # PH(.onnx) + DO(.onnx)

# Position above relevant bars
plt.text(
    0.5, max(energy_values) + 1.5,
    f"Energy inference in\n.pth version is {sum_1:.2f} mJ",
    ha='center', fontsize=10, color='black',
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
)

plt.text(
    2.5, max(energy_values) - 30,
    f"Energy inference in\n.onnx version is {sum_2:.2f} mJ",
    ha='center', fontsize=10, color='black',
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
)


plt.savefig("energy.png")
plt.show()
