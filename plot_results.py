import matplotlib.pyplot as plt
import numpy as np

# Data extracted from notebook logs
labels = ['Raw ResNet50\n(10 Epochs)', 'ResNet50 + Random Data\n(5 Epochs)', 'ResNet50 + Temporal Data\n(5 Epochs)']
test_accs = [54.98, 60.36, 65.35]

epochs_raw = np.arange(1, 11)
train_raw = [32.45, 38.24, 44.80, 48.68, 51.47, 53.92, 55.98, 57.81, 59.67, 61.73]
val_raw = [34.48, 42.69, 44.67, 46.70, 50.74, 53.38, 54.82, 55.66, 56.62, 55.37]

epochs_ft = np.arange(11, 16)
train_rnd = [64.52, 66.06, 66.68, 66.95, 67.16]
val_rnd = [59.98, 59.57, 60.57, 59.77, 60.39]

train_tmp = [66.68, 68.63, 69.30, 69.52, 69.91]
val_tmp = [63.13, 64.70, 65.36, 65.15, 65.35]

# 1. Bar Chart for Test Accuracies
plt.figure(figsize=(9, 6))
# A nice set of colors (Blue, Orange, Green)
colors = ['#4A90E2', '#F5A623', '#7ED321']
bars = plt.bar(labels, test_accs, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim(0, 80)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f'{yval:.2f}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig('test_accuracy_bar_chart.png', dpi=300)
plt.close()

# 2. Line Chart for Train and Val Accuracies
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Train Accuracy
axes[0].plot(epochs_raw, train_raw, marker='o', linestyle='-', linewidth=2, markersize=6, label='Raw ResNet50')
axes[0].plot(epochs_ft, train_rnd, marker='s', linestyle='--', linewidth=2, markersize=6, label='ResNet50 + Random Data')
axes[0].plot(epochs_ft, train_tmp, marker='^', linestyle='-', linewidth=2, markersize=6, label='ResNet50 + Temporal Data')
axes[0].set_title('Training Accuracy vs Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend(fontsize=11)
axes[0].set_xticks(np.arange(1, 16))
axes[0].set_ylim(30, 75)

# Subplot 2: Validation Accuracy
axes[1].plot(epochs_raw, val_raw, marker='o', linestyle='-', linewidth=2, markersize=6, label='Raw ResNet50')
axes[1].plot(epochs_ft, val_rnd, marker='s', linestyle='--', linewidth=2, markersize=6, label='ResNet50 + Random Data')
axes[1].plot(epochs_ft, val_tmp, marker='^', linestyle='-', linewidth=2, markersize=6, label='ResNet50 + Temporal Data')
axes[1].set_title('Validation Accuracy vs Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend(fontsize=11)
axes[1].set_xticks(np.arange(1, 16))
axes[1].set_ylim(30, 75)

plt.tight_layout()
plt.savefig('train_val_accuracy_lines.png', dpi=300)
plt.close()

print("Visualizations generated and saved locally!")
