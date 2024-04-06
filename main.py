import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
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
windowSize = 5

train_walking_roll = pd.DataFrame(train_walking_data).rolling(windowSize).mean()
train_jumping_roll = pd.DataFrame(train_jumping_data).rolling(windowSize).mean()
test_walking_roll = pd.DataFrame(test_walking_data).rolling(windowSize).mean()
test_jumping_roll = pd.DataFrame(test_jumping_data).rolling(windowSize).mean()

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

#Step 5 below:
def normalize_features(df):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
    return normalized_df

features_w_train = pd.DataFrame(columns=['mean', 'std', 'max', 'min', 'variance', 'skewness', 'kurtosis'])

features_w_train['mean'] = train_walking_roll.mean()
features_w_train['std'] = train_walking_roll.std()
features_w_train['max'] = train_walking_roll.max()
features_w_train['min'] = train_walking_roll.min()
features_w_train['variance'] = train_walking_roll.var()
features_w_train['skewness'] = train_walking_roll.skew()
features_w_train['kurtosis'] = train_walking_roll.kurt()

features_w_test = pd.DataFrame(columns=['mean', 'std', 'max', 'min', 'variance', 'skewness', 'kurtosis'])

features_w_test['mean'] = test_walking_roll.mean()
features_w_test['std'] = test_walking_roll.std()
features_w_test['max'] = test_walking_roll.max()
features_w_test['min'] = test_walking_roll.min()
features_w_test['variance'] = test_walking_roll.var()
features_w_test['skewness'] = test_walking_roll.skew()
features_w_test['kurtosis'] = test_walking_roll.kurt()

features_j_train = pd.DataFrame(columns=['mean', 'std', 'max', 'min', 'variance', 'skewness', 'kurtosis'])

features_j_train['mean'] = train_jumping_roll.mean()
features_j_train['std'] = train_jumping_roll.std()
features_j_train['max'] = train_jumping_roll.max()
features_j_train['min'] = train_jumping_roll.min()
features_j_train['variance'] = train_jumping_roll.var()
features_j_train['skewness'] = train_jumping_roll.skew()
features_j_train['kurtosis'] = train_jumping_roll.kurt()

features_j_test = pd.DataFrame(columns=['mean', 'std', 'max', 'min', 'variance', 'skewness', 'kurtosis'])

features_j_test['mean'] = test_jumping_roll.mean()
features_j_test['std'] = test_jumping_roll.std()
features_j_test['max'] = test_jumping_roll.max()
features_j_test['min'] = test_jumping_roll.min()
features_j_test['variance'] = test_jumping_roll.var()
features_j_test['skewness'] = test_jumping_roll.skew()
features_j_test['kurtosis'] = test_jumping_roll.kurt()




train_features_w_normalized = normalize_features(features_w_train)
test_features_w_normalized = normalize_features(features_w_test)

train_features_j_normalized = normalize_features(features_j_train)
test_features_j_normalized = normalize_features(features_j_test)

# def print_summary_statistics(df, title=""):#for testing the features and normalized features...double check everything is right
#     print(title)
#     print("Mean:\n", df.mean())
#     print("\nStandard Deviation:\n", df.std())
# 
# print_summary_statistics(features_w_train, "Original Walking Training Features Summary Statistics")
# print_summary_statistics(train_features_w_normalized, "Normalized Walking Training Features Summary Statistics")
#end of step 5

# Step 6
# Combine features and labels for training and testing sets
train_walking_normalized = scaler.fit_transform(train_walking_data)
train_jumping_normalized = scaler.fit_transform(train_jumping_data)
test_walking_normalized = scaler.fit_transform(test_walking_data)
test_jumping_normalized = scaler.fit_transform(test_jumping_data)

# Recalculate rolling means after normalization
train_walking_roll = pd.DataFrame(train_walking_normalized).rolling(windowSize).mean().dropna()
train_jumping_roll = pd.DataFrame(train_jumping_normalized).rolling(windowSize).mean().dropna()
test_walking_roll = pd.DataFrame(test_walking_normalized).rolling(windowSize).mean().dropna()
test_jumping_roll = pd.DataFrame(test_jumping_normalized).rolling(windowSize).mean().dropna()

# Concatenate DataFrames with continuous index
train_features = pd.concat([train_walking_roll, train_jumping_roll], ignore_index=True)
test_features = pd.concat([test_walking_roll, test_jumping_roll], ignore_index=True)

# Calculate labels
y_train = np.concatenate([np.ones(len(train_walking_roll)), np.zeros(len(train_jumping_roll))])
y_test = np.concatenate([np.ones(len(test_walking_roll)), np.zeros(len(test_jumping_roll))])

# Select only the desired columns for X_train and X_test
X_train_selected = train_features.iloc[:, 1:3]  # Selecting columns 1 and 2 (second and third columns)
X_test_selected = test_features.iloc[:, 1:3]  # Selecting columns 1 and 2 (second and third columns)

# Initialize Logistic Regression model
logistic_model = LogisticRegression()

# Train the model
logistic_model.fit(X_train_selected, y_train)

# Predict on the test set
y_pred = logistic_model.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

plt.show()