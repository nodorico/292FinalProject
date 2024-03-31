import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def segmentData(data):
    # Convert the time column to seconds
    time_seconds = data.iloc[:, 0]

    # Find the index of the time corresponding to the end of a 5-second window
    window_size_seconds = 5
    window_end_indices = time_seconds.searchsorted(
        time_seconds.iloc[0] + np.arange(window_size_seconds, time_seconds.max(), window_size_seconds))

    # Segment the data into 5-second windows
    segmented_data = [data.iloc[i:j] for i, j in
                      zip([0] + list(window_end_indices), list(window_end_indices) + [len(data)])]

    np.random.shuffle(segmented_data)
    return segmented_data

NdataWalking = segmentData(pd.read_csv('NicolasWalking.csv'))
BdataWalking = segmentData(pd.read_csv('BrandonWalking.csv'))
dataWalking = NdataWalking + BdataWalking

NdataJumping = segmentData(pd.read_csv('NicolasJumping.csv'))
BdataJumping = segmentData(pd.read_csv('BrandonJumping.csv'))
dataJumping = NdataJumping + BdataJumping

for segment in dataWalking:
    segment['Label'] = 'walking'

for segment in dataJumping:
    segment['Label'] = 'jumping'

allData = dataWalking + dataJumping

train_data, test_data = train_test_split(allData, test_size=0.1)

train_df = pd.concat(train_data)
test_df = pd.concat(test_data)

with h5py.File('data.hdf5','a') as f:
    dataset = f.create_dataset('292', dtype='d')
    subgroups = ['dataset', 'Nicolas', 'Rylan', 'Brandon']
    for subgroup_name in subgroups:
        subgroup = f.create_group(subgroup_name)

    # Create sub-subgroup datasets within the 'dataset' subgroup
    dataset_subgroups = ['test', 'train']
    for subgroup_name in dataset_subgroups:
        subgroup = f['dataset'].create_group(subgroup_name)

    subgroup = f['Nicolas']
    df = pd.read_csv('NicolasWalking.csv')
    subgroup.create_dataset(name='NicolasWalking', data=df)

    df = pd.read_csv('NicolasJumping.csv')
    subgroup.create_dataset(name='NicolasJumping', data=df)

    subgroup = f['Brandon']
    df = pd.read_csv('BrandonWalking.csv')
    subgroup.create_dataset(name='BrandonWalking', data=df)

    df = pd.read_csv('BrandonJumping.csv')
    subgroup.create_dataset(name='BrandonJumping', data=df)

    with pd.HDFStore('data.hdf5') as store:
        # Store train_df and test_df in the HDF5 file
        store.put('dataset/train', train_df)
        store.put('dataset/test', test_df)