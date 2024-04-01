import h5py
import pandas as pd

# Open the HDF5 file in read mode
with h5py.File('data.hdf5', 'r') as f:
    # Access the 'train' group within the 'dataset' subgroup
    train_group = f['dataset']['train']

    # Extract the components of the DataFrame
    axis0 = train_group['axis0'][:]
    axis1 = train_group['axis1'][:]
    block0_items = train_group['block0_items'][:]
    block0_values = train_group['block0_values'][:]
    block1_items = train_group['block1_items'][:]
    block1_values = train_group['block1_values'][:]

    # Reconstruct the DataFrame
    train_df = pd.DataFrame(data=block0_values, columns=block0_items)
    train_df.columns = ['Time (s)', 'Acceleration (m/s^2)', 'X', 'Y', 'Z']

# Now you can use 'train_df' as a pandas DataFrame containing the 'train' dataset
print(train_df)