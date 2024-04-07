import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
windowSize = 350

train_walking_roll = pd.DataFrame(train_walking_data).rolling(windowSize).mean().dropna()
train_jumping_roll = pd.DataFrame(train_jumping_data).rolling(windowSize).mean().dropna()
test_walking_roll = pd.DataFrame(test_walking_data).rolling(windowSize).mean().dropna()
test_jumping_roll = pd.DataFrame(test_jumping_data).rolling(windowSize).mean().dropna()

train_walking_roll = train_walking_roll.iloc[:,]
train_jumping_roll = train_jumping_roll.iloc[:,]
test_walking_roll = test_walking_roll.iloc[:,]
test_jumping_roll = test_jumping_roll.iloc[:,]

# Apply preprocessing steps
scaler = StandardScaler()

# Normalize the data
train_walking_normalized = scaler.fit_transform(train_walking_roll)
train_jumping_normalized = scaler.fit_transform(train_jumping_roll)
test_walking_normalized = scaler.fit_transform(test_walking_roll)
test_jumping_normalized = scaler.fit_transform(test_jumping_roll)

# Create subplots for X, Y, and Z axes
fig, axs = plt.subplots(4, 2, figsize=(50, 30))

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

axs[3, 0].plot(train_walking_normalized[:, 3], label='Walking', color='blue')
axs[3, 0].plot(train_jumping_normalized[:, 3], label='Jumping', color='orange')
axs[3, 0].set_title('Accelerometer Data for Training Set - Absolute')
axs[3, 0].set_xlabel('Timestamp')
axs[3, 0].set_ylabel('Acceleration')
axs[3, 0].legend()

# Plot accelerometer data for testing set - Z-axis
axs[3, 1].plot(test_walking_normalized[:, 3], label='Walking', color='blue')
axs[3, 1].plot(test_jumping_normalized[:, 3], label='Jumping', color='orange')
axs[3, 1].set_title('Accelerometer Data for Testing Set - Absolute')
axs[3, 1].set_xlabel('Timestamp')
axs[3, 1].set_ylabel('Acceleration')
axs[3, 1].legend()

#Step 5 below:

#make a function to extract features from the segments. The features i want to extract are mean, std, max, min, variance, skewness, kurtosis, sma, autocorrelation, and cross-axis correlation
def extract_features(segment):
    features = [
        np.max(segment),
        np.min(segment),
        np.ptp(segment),
        np.mean(segment),
        np.median(segment),
        np.var(segment),
        np.std(segment),
        np.mean(np.absolute(segment - np.mean(segment)))
    ]
    return features

# Extract features for all segments
features_wtr = np.array([extract_features(segment) for segment in train_walking_normalized])
features_jtr = np.array([extract_features(segment) for segment in train_jumping_normalized])
features_wte = np.array([extract_features(segment) for segment in test_walking_normalized])
features_jte = np.array([extract_features(segment) for segment in test_jumping_normalized])


# Normalize features
w_n_train = scaler.fit_transform(features_wtr)
j_n_train = scaler.fit_transform(features_jtr)
w_n_test = scaler.fit_transform(features_wte)
j_n_test = scaler.fit_transform(features_jte)



#____________________________________________________________________________________

#Step 6 below:

# Create training and testing sets
X_train = np.concatenate((w_n_train, j_n_train))
y_train = np.concatenate((np.ones(len(w_n_train)), np.zeros(len(j_n_train))))
X_test = np.concatenate((w_n_test, j_n_test))
y_test = np.concatenate((np.ones(len(w_n_test)), np.zeros(len(j_n_test))))

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
#do accuracy test for the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# _____________________________________________________


#____________________________________________________________________________________________________

# plt.show()