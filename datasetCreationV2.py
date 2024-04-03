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

nicolasWalking['Activity'] = 1
brandonWalking['Activity'] = 1
rylanWalking['Activity'] = 1
nicolasJumping['Activity'] = 0
brandonJumping['Activity'] = 0
rylanJumping['Activity'] = 0

with h5py.File('./dataset.h5','w') as hdf:
    nicolas = hdf.create_group('/Nicolas')
    nicolas.create_dataset('nicolasWalking', data=nicolasWalking.values)
    nicolas.create_dataset('nicolasJumping', data=nicolasJumping.values)

    brandon = hdf.create_group('/Brandon')
    brandon.create_dataset('brandonWalking', data=brandonWalking.values)
    brandon.create_dataset('brandonJumping', data=brandonJumping.values)

    rylan = hdf.create_group('/Rylan')
    rylan.create_dataset('rylanWalking', data=rylanWalking.values)
    rylan.create_dataset('rylanJumping', data=rylanJumping.values)

walking_data = pd.concat([nicolasWalking, brandonWalking, rylanWalking], ignore_index=True)
jumping_data = pd.concat([nicolasJumping, brandonJumping, rylanJumping], ignore_index=True)
combinedDatasets = pd.concat([walking_data, jumping_data], ignore_index=True)

# Add labels to combined dataset
combinedDatasets['Activity'] = combinedDatasets['Activity'].astype(int)


time_seconds = combinedDatasets.iloc[:, 0]

# Find the index of the time corresponding to the end of a 5-second window
window_size_seconds = 5
window_end_indices = time_seconds.searchsorted(
    time_seconds.iloc[0] + np.arange(window_size_seconds, time_seconds.max(), window_size_seconds))

# Segment the data into 5-second windows
segmented = [combinedDatasets.iloc[i:j] for i, j in
                  zip([0] + list(window_end_indices), list(window_end_indices) + [len(combinedDatasets)])]

np.random.shuffle(segmented)

train_size = int(0.9 * len(segmented))
train_segments = segmented[:train_size]
test_segments = segmented[train_size:]

train_data = np.concatenate([segment['Activity'].values.reshape(-1, 1) for segment in train_segments])
test_data = np.concatenate([segment['Activity'].values.reshape(-1, 1) for segment in test_segments])


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