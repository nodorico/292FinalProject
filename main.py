import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

# Load training and testing data from the HDF5 file
with h5py.File('data.hdf5', 'r') as f:
    train_data = f['dataset']['train'][:]
    test_data = f['dataset']['test'][:]

# Separate walking and jumping data for training and testing sets
train_walking_data = train_data[train_data[:, -1] == b'walking'][:, :-1]
train_jumping_data = train_data[train_data[:, -1] == b'jumping'][:, :-1]
test_walking_data = test_data[test_data[:, -1] == b'walking'][:, :-1]
test_jumping_data = test_data[test_data[:, -1] == b'jumping'][:, :-1]

# Create rolling mean dataset for visualization
windowSize = 50
train_walking_roll = pd.DataFrame(train_walking_data).rolling(windowSize).mean()
train_jumping_roll = pd.DataFrame(train_jumping_data).rolling(windowSize).mean()
test_walking_roll = pd.DataFrame(test_walking_data).rolling(windowSize).mean()
test_jumping_roll = pd.DataFrame(test_jumping_data).rolling(windowSize).mean()

# Create subplots for X, Y, and Z axes
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Plot accelerometer data for training set - X-axis
axs[0, 0].plot(train_walking_roll.iloc[:, 0], train_walking_roll.iloc[:, 1], label='Walking', color='blue')
axs[0, 0].plot(train_jumping_roll.iloc[:, 0], train_jumping_roll.iloc[:, 1], label='Jumping', color='orange')
axs[0, 0].set_title('Accelerometer Data for Training Set - X-axis')
axs[0, 0].set_xlabel('Timestamp')
axs[0, 0].set_ylabel('Acceleration')
axs[0, 0].legend()

# Plot accelerometer data for testing set - X-axis
axs[0, 1].plot(test_walking_roll.iloc[:, 0], test_walking_roll.iloc[:, 1], label='Walking', color='blue')
axs[0, 1].plot(test_jumping_roll.iloc[:, 0], test_jumping_roll.iloc[:, 1], label='Jumping', color='orange')
axs[0, 1].set_title('Accelerometer Data for Testing Set - X-axis')
axs[0, 1].set_xlabel('Timestamp')
axs[0, 1].set_ylabel('Acceleration')
axs[0, 1].legend()

# Plot accelerometer data for training set - Y-axis
axs[1, 0].plot(train_walking_roll.iloc[:, 0], train_walking_roll.iloc[:, 2], label='Walking', color='blue')
axs[1, 0].plot(train_jumping_roll.iloc[:, 0], train_jumping_roll.iloc[:, 2], label='Jumping', color='orange')
axs[1, 0].set_title('Accelerometer Data for Training Set - Y-axis')
axs[1, 0].set_xlabel('Timestamp')
axs[1, 0].set_ylabel('Acceleration')
axs[1, 0].legend()

# Plot accelerometer data for testing set - Y-axis
axs[1, 1].plot(test_walking_roll.iloc[:, 0], test_walking_roll.iloc[:, 2], label='Walking', color='blue')
axs[1, 1].plot(test_jumping_roll.iloc[:, 0], test_jumping_roll.iloc[:, 2], label='Jumping', color='orange')
axs[1, 1].set_title('Accelerometer Data for Testing Set - Y-axis')
axs[1, 1].set_xlabel('Timestamp')
axs[1, 1].set_ylabel('Acceleration')
axs[1, 1].legend()

# Plot accelerometer data for training set - Z-axis
axs[2, 0].plot(train_walking_roll.iloc[:, 0], train_walking_roll.iloc[:, 3], label='Walking', color='blue')
axs[2, 0].plot(train_jumping_roll.iloc[:, 0], train_jumping_roll.iloc[:, 3], label='Jumping', color='orange')
axs[2, 0].set_title('Accelerometer Data for Training Set - Z-axis')
axs[2, 0].set_xlabel('Timestamp')
axs[2, 0].set_ylabel('Acceleration')
axs[2, 0].legend()

# Plot accelerometer data for testing set - Z-axis
axs[2, 1].plot(test_walking_roll.iloc[:, 0], test_walking_roll.iloc[:, 3], label='Walking', color='blue')
axs[2, 1].plot(test_jumping_roll.iloc[:, 0], test_jumping_roll.iloc[:, 3], label='Jumping', color='orange')
axs[2, 1].set_title('Accelerometer Data for Testing Set - Z-axis')
axs[2, 1].set_xlabel('Timestamp')
axs[2, 1].set_ylabel('Acceleration')
axs[2, 1].legend()

plt.tight_layout()
plt.show()


# show metadata 
