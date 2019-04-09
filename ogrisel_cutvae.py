import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_cut_data(cut_idx=45, voxel_standardize=True):
    subject_100307_folder = Path("~/data/HCP_100307").expanduser()
    rest_filepaths = sorted(subject_100307_folder.glob("*REST*.npy"))

    slice_folder = subject_100307_folder / f"slice_{cut_idx:03d}"
    slice_folder.mkdir(exist_ok=True)

    slice_filepaths = []
    for rest_filepath in rest_filepaths:
        slice_filepath = slice_folder / rest_filepath.name
        slice_filepaths.append(slice_filepath)
        if not slice_filepath.exists():
            print(f"Saving {slice_filepath}...")
            data = np.load(str(rest_filepath), mmap_mode="r")
            np.save(str(slice_filepath), data[:, :, cut_idx, :])

    data_train = np.concatenate(
        [np.load(str(f)) for f in slice_filepaths[:-1]], axis=-1)
    data_test = np.load(str(slice_filepaths[-1]))

    if voxel_standardize:
        # Voxel-wise standardization on the training set
        mean_train = data_train.mean(axis=-1)
        std_train = data_train.std(axis=-1)

        def standarize(data):
            data = data - mean_train[..., None]
            inv_scale = std_train
            inv_scale[std_train == 0] = 1.
            return data / inv_scale[..., None]

        data_train = standarize(data_train)
        data_test = standarize(data_test)
    return data_train, data_test


data_train_raw, data_test_raw = load_cut_data(voxel_standardize=False)
plt.figure(figsize=(16, 8))
plt.imshow(data_train_raw[:, :, 0], cmap=plt.cm.viridis)
plt.colorbar()
plt.title("Frame 0 of original data")

data_train_raw, data_test_raw = load_cut_data(voxel_standardize=False)
plt.figure(figsize=(16, 8))
plt.imshow(data_train_raw.std(axis=-1), cmap=plt.cm.viridis)
plt.colorbar()
plt.title("Voxel-wise standard dev of original data")

data_train, data_test = load_cut_data()
print(f"Train data shape: {data_train.shape}")
print(f"Test data shape: {data_test.shape}")

plt.figure(figsize=(16, 8))
plt.imshow(data_train[:, :, 10], cmap=plt.cm.RdBu)
plt.colorbar()
plt.title("Frame 0 of voxel-wise standardized data")
