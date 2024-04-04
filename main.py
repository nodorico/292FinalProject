import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler

# Load training and testing data from the HDF5 file
with h5py.File('dataset.h5', 'r') as f:
    train_data = f['dataset/train/train_data'][:]
    test_data = f['dataset/testing/test_data'][:]

# Separate walking and jumping data for training and testing sets
train_labels = train_data[:, -1]
train_walking_data = train_data[train_labels == 1][:, :-1]  # Assuming 1 corresponds to walking
train_jumping_data = train_data[train_labels == 0][:, :-1]  # Assuming 0 corresponds to jumping

test_labels = test_data[:, -1]
test_walking_data = test_data[test_labels == 1][:, :-1]    # Assuming 1 corresponds to walking
test_jumping_data = test_data[test_labels == 0][:, :-1]

# Create rolling mean dataset for visualization
windowSize = 500
train_walking_roll = pd.DataFrame(train_walking_data).rolling(windowSize).mean()
train_jumping_roll = pd.DataFrame(train_jumping_data).rolling(windowSize).mean()
test_walking_roll = pd.DataFrame(test_walking_data).rolling(windowSize).mean()
test_jumping_roll = pd.DataFrame(test_jumping_data).rolling(windowSize).mean()

# Apply preprocessing steps
scaler = StandardScaler()

# Normalize the data
train_walking_normalized = scaler.fit_transform(train_walking_roll)
train_jumping_normalized = scaler.fit_transform(train_jumping_roll)
test_walking_normalized = scaler.fit_transform(test_walking_roll)
test_jumping_normalized = scaler.fit_transform(test_jumping_roll)

# Create subplots for X, Y, and Z axes
fig, axs = plt.subplots(3, 2, figsize=(50, 30))

axs[0, 0].plot(train_walking_normalized[:, 0], label='Walking', color='blue')
axs[0, 0].plot(train_jumping_normalized[:, 0], label='Jumping', color='orange')
axs[0, 0].set_title('Accelerometer Data for Training Set - X-axis')
axs[0, 0].set_xlabel('Timestamp')
axs[0, 0].set_ylabel('Acceleration')
axs[0, 0].legend()

# Plot accelerometer data for testing set - X-axis
axs[0, 1].plot(test_walking_normalized[:, 0], label='Walking', color='blue')
axs[0, 1].plot(test_jumping_normalized[:, 0], label='Jumping', color='orange')
axs[0, 1].set_title('Accelerometer Data for Testing Set - X-axis')
axs[0, 1].set_xlabel('Timestamp')
axs[0, 1].set_ylabel('Acceleration')
axs[0, 1].legend()

# Plot accelerometer data for training set - Y-axis
axs[1, 0].plot(train_walking_normalized[:, 1], label='Walking', color='blue')
axs[1, 0].plot(train_jumping_normalized[:, 1], label='Jumping', color='orange')
axs[1, 0].set_title('Accelerometer Data for Training Set - Y-axis')
axs[1, 0].set_xlabel('Timestamp')
axs[1, 0].set_ylabel('Acceleration')
axs[1, 0].legend()

# Plot accelerometer data for testing set - Y-axis
axs[1, 1].plot(test_walking_normalized[:, 1], label='Walking', color='blue')
axs[1, 1].plot(test_jumping_normalized[:, 1], label='Jumping', color='orange')
axs[1, 1].set_title('Accelerometer Data for Testing Set - Y-axis')
axs[1, 1].set_xlabel('Timestamp')
axs[1, 1].set_ylabel('Acceleration')
axs[1, 1].legend()

# Plot accelerometer data for training set - Z-axis
axs[2, 0].plot(train_walking_normalized[:, 2], label='Walking', color='blue')
axs[2, 0].plot(train_jumping_normalized[:, 2], label='Jumping', color='orange')
axs[2, 0].set_title('Accelerometer Data for Training Set - Z-axis')
axs[2, 0].set_xlabel('Timestamp')
axs[2, 0].set_ylabel('Acceleration')
axs[2, 0].legend()

# Plot accelerometer data for testing set - Z-axis
axs[2, 1].plot(test_walking_normalized[:, 2], label='Walking', color='blue')
axs[2, 1].plot(test_jumping_normalized[:, 2], label='Jumping', color='orange')
axs[2, 1].set_title('Accelerometer Data for Testing Set - Z-axis')
axs[2, 1].set_xlabel('Timestamp')
axs[2, 1].set_ylabel('Acceleration')
axs[2, 1].legend()

plt.show()
