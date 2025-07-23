import matplotlib.pyplot as plt

# Model sizes in kilobytes (kB)
model_size = [185.8, 1111.7, 344.1, 1322.7]
labels = ['PH(.pth)', 'DO(.pth)', 'PH(.onnx)', 'DO(.onnx)']
colors = ['darkturquoise', 'darkviolet', 'lawngreen', 'cornflowerblue']

# Plot
plt.figure(figsize=(6, 4))
plt.bar(labels, model_size, color=colors, edgecolor='black')

# Labeling
plt.ylabel('Model Size (kB)', fontsize=13)
plt.title('Size of PH and DO Models', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.ylim(0,1300)
plt.savefig("model_size.png")
plt.show()
