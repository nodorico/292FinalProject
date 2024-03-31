import h5py
import numpy as np
import pandas as pd





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

