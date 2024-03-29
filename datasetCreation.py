import h5py

with h5py.File('data.hdf5','a') as f:
    dataset = f.create_dataset('292', dtype='d')
    subgroups = ['dataset', 'Nicolas', 'Rylan', 'Brandon']
    for subgroup_name in subgroups:
        subgroup = f.create_group(subgroup_name)

    # Create sub-subgroup datasets within the 'dataset' subgroup
    dataset_subgroups = ['test', 'train']
    for subgroup_name in dataset_subgroups:
        subgroup = f['dataset'].create_group(subgroup_name)