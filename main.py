import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('NicolasWalking.csv')

data = dataset.iloc[:, :]
labels = dataset.columns.tolist()



windowSize = 100
datasetRoll = dataset.rolling(windowSize).mean()

# Create subplots for X, Y, and Z axes
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Plot accelerometer data for X-axis
axs[0].plot(datasetRoll[labels[0]], datasetRoll[labels[1]], label='X-axis', color='blue')
#axs[0].plot(dataset31[labels[0]], dataset[labels[1]], label='X-axis', color='orange')
axs[0].set_title('Accelerometer Data for Walking - X-axis')
axs[0].set_xlabel('Timestamp')
axs[0].set_ylabel('Acceleration')
axs[0].legend()

# Plot accelerometer data for Y-axis
axs[1].plot(datasetRoll[labels[0]], datasetRoll[labels[2]], label='Y-axis', color='blue')
#axs[1].plot(dataset31[labels[0]], dataset[labels[2]], label='Y-axis', color='orange')
axs[1].set_title('Accelerometer Data for Walking - Y-axis')
axs[1].set_xlabel('Timestamp')
axs[1].set_ylabel('Acceleration')
axs[1].legend()

# Plot accelerometer data for Z-axis
axs[2].plot(datasetRoll[labels[0]], datasetRoll[labels[3]], label='Z-axis', color='blue')
#axs[2].plot(dataset31[labels[0]], dataset[labels[3]], label='Z-axis', color='orange')
axs[2].set_title('Accelerometer Data for Walking - Z-axis')
axs[2].set_xlabel('Timestamp')
axs[2].set_ylabel('Acceleration')
axs[2].legend()

plt.tight_layout()
plt.show()