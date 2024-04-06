import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Read walking and jumping data for each individual
nicolasWalking = pd.read_csv('NicolasWalking.csv')
brandonWalking = pd.read_csv('BrandonWalking.csv')
rylanWalking = pd.read_csv('RylanWalking.csv')

nicolasJumping = pd.read_csv('NicolasJumping.csv')
brandonJumping = pd.read_csv('BrandonJumping.csv')
rylanJumping = pd.read_csv('RylanJumping.csv')

# Create an HDF5 file and store the datasets
with h5py.File('./dataset.h5', 'w') as hdf:
    nicolas = hdf.create_group('/Nicolas')
    nicolas.create_dataset('nicolasWalking', data=nicolasWalking)
    nicolas.create_dataset('nicolasJumping', data=nicolasJumping)

    brandon = hdf.create_group('/Brandon')
    brandon.create_dataset('brandonWalking', data=brandonWalking.values)
    brandon.create_dataset('brandonJumping', data=brandonJumping.values)

    rylan = hdf.create_group('/Rylan')
    rylan.create_dataset('rylanWalking', data=rylanWalking.values)
    rylan.create_dataset('rylanJumping', data=rylanJumping.values)

# Combine walking and jumping data from all individuals
walking_data = pd.concat([nicolasWalking, brandonWalking, rylanWalking], ignore_index=True)
jumping_data = pd.concat([nicolasJumping, brandonJumping, rylanJumping], ignore_index=True)

# Create a new column for activity (0 for jumping, 1 for walking)
walking_data['Activity'] = 1
jumping_data['Activity'] = 0

# Concatenate walking and jumping data
combinedDatasets = pd.concat([walking_data, jumping_data], ignore_index=True)

# Set up parameters for segmenting the data
sampling_frequency = 100  # Hz
window_size_seconds = 5
window_size_samples = int(window_size_seconds * sampling_frequency)

# Find the index of the time corresponding to the end of a 5-second window
time_seconds = combinedDatasets.iloc[:, 0]
window_end_indices = np.arange(window_size_samples, len(time_seconds), window_size_samples)

# Segment the data into 5-second windows
segmented = [combinedDatasets.iloc[i:j] for i, j in zip([0] + list(window_end_indices), list(window_end_indices) + [len(combinedDatasets)])]
np.random.shuffle(segmented)

# Split the data into training and testing sets
train_size = int(0.9 * len(segmented))
train_segments = segmented[:train_size]
test_segments = segmented[train_size:]
print(len(segmented))

# Concatenate the segments into train and test data arrays
train_data = np.concatenate([segment.values for segment in train_segments])
test_data = np.concatenate([segment.values for segment in test_segments])

# Convert the data to float64
train_data = train_data.astype(np.float64)
test_data = test_data.astype(np.float64)

# Append train_data and test_data to the HDF5 file
with h5py.File('./dataset.h5', 'a') as hdf:
    train_group = hdf.create_group('dataset/train')
    test_group = hdf.create_group('dataset/testing')

    train_group.create_dataset('train_data', data=train_data)
    test_group.create_dataset('test_data', data=test_data)

# Print the shape of datasets in the HDF5 file
with h5py.File('./dataset.h5', 'r') as hdf:
    def print_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        else:
            print(f"Unknown object: {name}")

    hdf.visititems(print_dataset)
