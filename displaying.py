import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Load training and testing data from the HDF5 file
with h5py.File('dataset.h5', 'r') as f:
    train_data = f['dataset/train/train_data'][:]
    test_data = f['dataset/testing/test_data'][:]

# Separate walking and jumping data for training and testing sets
train_labels = train_data[:, -1]
train_walking_data = train_data[train_labels == 1][:, :-1]
train_jumping_data = train_data[train_labels == 0][:, :-1]

test_labels = test_data[:, -1]
test_walking_data = test_data[test_labels == 1][:, :-1]
test_jumping_data = test_data[test_labels == 0][:, :-1]

# Create rolling mean dataset for visualization
windowSize = 500

train_walking_roll = pd.DataFrame(train_walking_data).rolling(windowSize).mean().dropna()
train_jumping_roll = pd.DataFrame(train_jumping_data).rolling(windowSize).mean().dropna()
test_walking_roll = pd.DataFrame(test_walking_data).rolling(windowSize).mean().dropna()
test_jumping_roll = pd.DataFrame(test_jumping_data).rolling(windowSize).mean().dropna()

# train_walking_roll = train_walking_roll.iloc[:,]
# train_jumping_roll = train_jumping_roll.iloc[:,]
# test_walking_roll = test_walking_roll.iloc[:,]
# test_jumping_roll = test_jumping_roll.iloc[:,]

# Apply preprocessing steps
scaler = StandardScaler()

# Normalize the data
train_walking_normalized = scaler.fit_transform(train_walking_roll)
train_jumping_normalized = scaler.fit_transform(train_jumping_roll)
test_walking_normalized = scaler.fit_transform(test_walking_roll)
test_jumping_normalized = scaler.fit_transform(test_jumping_roll)


def grapher(trainW, trainJ, testW, testJ):
    # Create subplots for X, Y, and Z axes
    fig, axs = plt.subplots(4, 2, figsize=(20, 20))

    axs[0, 0].plot(trainW[:, 0], label='Walking', color='blue')
    axs[0, 0].plot(trainJ[:, 0], label='Jumping', color='orange')
    axs[0, 0].set_title('Accelerometer Data for Training Set - X-axis')
    axs[0, 0].set_xlabel('Timestamp')
    axs[0, 0].set_ylabel('Acceleration')
    axs[0, 0].legend()

    # Plot accelerometer data for testing set - X-axis
    axs[0, 1].plot(testW[:, 0], label='Walking', color='blue')
    axs[0, 1].plot(testJ[:, 0], label='Jumping', color='orange')
    axs[0, 1].set_title('Accelerometer Data for Testing Set - X-axis')
    axs[0, 1].set_xlabel('Timestamp')
    axs[0, 1].set_ylabel('Acceleration')
    axs[0, 1].legend()

    # Plot accelerometer data for training set - Y-axis
    axs[1, 0].plot(trainW[:, 1], label='Walking', color='blue')
    axs[1, 0].plot(trainJ[:, 1], label='Jumping', color='orange')
    axs[1, 0].set_title('Accelerometer Data for Training Set - Y-axis')
    axs[1, 0].set_xlabel('Timestamp')
    axs[1, 0].set_ylabel('Acceleration')
    axs[1, 0].legend()

    # Plot accelerometer data for testing set - Y-axis
    axs[1, 1].plot(testW[:, 1], label='Walking', color='blue')
    axs[1, 1].plot(testJ[:, 1], label='Jumping', color='orange')
    axs[1, 1].set_title('Accelerometer Data for Testing Set - Y-axis')
    axs[1, 1].set_xlabel('Timestamp')
    axs[1, 1].set_ylabel('Acceleration')
    axs[1, 1].legend()

    # Plot accelerometer data for training set - Z-axis
    axs[2, 0].plot(trainW[:, 2], label='Walking', color='blue')
    axs[2, 0].plot(trainJ[:, 2], label='Jumping', color='orange')
    axs[2, 0].set_title('Accelerometer Data for Training Set - Z-axis')
    axs[2, 0].set_xlabel('Timestamp')
    axs[2, 0].set_ylabel('Acceleration')
    axs[2, 0].legend()

    # Plot accelerometer data for testing set - Z-axis
    axs[2, 1].plot(testW[:, 2], label='Walking', color='blue')
    axs[2, 1].plot(testJ[:, 2], label='Jumping', color='orange')
    axs[2, 1].set_title('Accelerometer Data for Testing Set - Z-axis')
    axs[2, 1].set_xlabel('Timestamp')
    axs[2, 1].set_ylabel('Acceleration')
    axs[2, 1].legend()

    axs[3, 0].plot(trainW[:, 3], label='Walking', color='blue')
    axs[3, 0].plot(trainJ[:, 3], label='Jumping', color='orange')
    axs[3, 0].set_title('Accelerometer Data for Training Set - Absolute')
    axs[3, 0].set_xlabel('Timestamp')
    axs[3, 0].set_ylabel('Acceleration')
    axs[3, 0].legend()

    # Plot accelerometer data for testing set - Z-axis
    axs[3, 1].plot(testW[:, 3], label='Walking', color='blue')
    axs[3, 1].plot(testJ[:, 3], label='Jumping', color='orange')
    axs[3, 1].set_title('Accelerometer Data for Testing Set - Absolute')
    axs[3, 1].set_xlabel('Timestamp')
    axs[3, 1].set_ylabel('Acceleration')
    axs[3, 1].legend()

    plt.show()


print("Shape of train_walking_data:", train_walking_data.shape)
print("Shape of train_walking_roll:", train_walking_roll.shape)
print("Shape of train_walking_normalized:", train_walking_normalized.shape)

grapher(train_walking_data, train_jumping_data, test_walking_data, test_jumping_data)



train_walking_roll = np.array(train_walking_roll)
train_jumping_roll = np.array(train_jumping_roll)
test_walking_roll = np.array(test_walking_roll)
test_jumping_roll = np.array(test_jumping_roll)

#grapher(train_walking_roll, train_jumping_roll, test_walking_roll, test_jumping_roll)
#grapher(train_walking_normalized, train_jumping_normalized, test_walking_normalized, test_jumping_normalized)

def rollingGrapher(windowSize):
    train_walking_roll = pd.DataFrame(train_walking_data).rolling(windowSize).mean().dropna()
    train_jumping_roll = pd.DataFrame(train_jumping_data).rolling(windowSize).mean().dropna()
    test_walking_roll = pd.DataFrame(test_walking_data).rolling(windowSize).mean().dropna()
    test_jumping_roll = pd.DataFrame(test_jumping_data).rolling(windowSize).mean().dropna()
    train_walking_roll = np.array(train_walking_roll)
    train_jumping_roll = np.array(train_jumping_roll)
    test_walking_roll = np.array(test_walking_roll)
    test_jumping_roll = np.array(test_jumping_roll)

    grapher(train_walking_roll, train_jumping_roll, test_walking_roll, test_jumping_roll)

rollingGrapher(10)
rollingGrapher(50)
rollingGrapher(100)
rollingGrapher(500)
rollingGrapher(1000)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D acceleration
ax.plot(test_jumping_normalized[:, 3], test_jumping_normalized[:, 1], test_jumping_normalized[:, 2], label='Jumping Data')
ax.plot(test_walking_normalized[:, 3], test_walking_normalized[:, 1], test_walking_normalized[:, 2], label='Walking Data', color = 'orange')
# Set labels and title
ax.set_xlabel('Acceleration Z')
ax.set_ylabel('Acceleration Y')
ax.set_zlabel('Acceleration X')
ax.set_title('3D Acceleration Plot')

# Add legend
ax.legend()

# Show plot
plt.show()




