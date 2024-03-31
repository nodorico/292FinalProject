import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("NicolasWalking.csv")


def segmentData(data):
# Convert the time column to seconds
    time_seconds = data.iloc[:, 0]

    # Find the index of the time corresponding to the end of a 5-second window
    window_size_seconds = 5
    window_end_indices = time_seconds.searchsorted(time_seconds.iloc[0] + np.arange(window_size_seconds, time_seconds.max(), window_size_seconds))

    # Segment the data into 5-second windows
    segmented_data = [data.iloc[i:j] for i, j in zip([0] + list(window_end_indices), list(window_end_indices) + [len(data)])]

    np.random.shuffle(segmented_data)
    return segmented_data
