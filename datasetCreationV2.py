import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

nicolasWalking = pd.read_csv('NicolasWalking.csv')
brandonWalking = pd.read_csv('BrandonWalking.csv')
rylanWalking = pd.read_csv('RylanWalking.csv')

nicolasJumping = pd.read_csv('NicolasJumping.csv')
brandonJumping = pd.read_csv('BrandonJumping.csv')
rylanJumping = pd.read_csv('RylanJumping.csv')

with h5py.File('./dataset.h5','w') as hdf:
    nicolas = hdf.create_group('/Nicolas')
    nicolas.create_dataset('nicolasWalking', data=nicolasWalking)
    nicolas.create_dataset('nicolasJumping', data=nicolasJumping)

    brandon = hdf.create_group('/Brandon')
    brandon.create_dataset('brandonWalking', data=brandonWalking.values)
    brandon.create_dataset('brandonJumping', data=brandonJumping.values)

    rylan = hdf.create_group('/Rylan')
    rylan.create_dataset('rylanWalking', data=rylanWalking.values)
    rylan.create_dataset('rylanJumping', data=rylanJumping.values)

walking_data = pd.concat([nicolasWalking, brandonWalking, rylanWalking], ignore_index=True)
jumping_data = pd.concat([nicolasJumping, brandonJumping, rylanJumping], ignore_index=True)
# create new column for activity
walking_data['Activity'] = '0'
jumping_data['Activity'] = '1'

combinedDatasets = pd.concat([walking_data, jumping_data], ignore_index=True)

time_seconds = combinedDatasets.iloc[:, 0]

sampling_frequency = 100  # Hz
window_size_seconds = 5
window_size_samples = int(window_size_seconds * sampling_frequency)

# Find the index of the time corresponding to the end of a 5-second window
window_end_indices = np.arange(window_size_samples, len(time_seconds), window_size_samples)

# Segment the data into 5-second windows
segmented = [combinedDatasets.iloc[i:j] for i, j in zip([0] + list(window_end_indices), list(window_end_indices) + [len(combinedDatasets)])]
np.random.shuffle(segmented)

train_size = int(0.9 * len(segmented))

print(len(segmented))
train_segments = segmented[:train_size]
test_segments = segmented[train_size:]

train_data = np.concatenate([segment.values for segment in train_segments])
test_data = np.concatenate([segment.values for segment in test_segments])

train_data = train_data.astype(np.float64)
test_data = test_data.astype(np.float64)


with h5py.File('./dataset.h5', 'a') as hdf:

    train_group = hdf.create_group('dataset/train')
    test_group = hdf.create_group('dataset/testing')

    # Store the combined training data as a single dataset
    train_group.create_dataset('train_data', data=train_data)

    # Store the combined testing data as a single dataset
    test_group.create_dataset('test_data', data=test_data)

with h5py.File('./dataset.h5', 'r') as hdf:
    def print_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        else:
            print(f"Unknown object: {name}")

    hdf.visititems(print_dataset)