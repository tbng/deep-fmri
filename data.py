import glob

import numpy as np
import torch
from os.path import expanduser, join
from skimage.morphology import dilation, binary_dilation
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path


class NumpyDataset(Dataset):
    def __init__(self, filename, return_2d=False, cut_idx=40):
        self.filename = filename
        self.return_2d = return_2d
        self.cut_idx = cut_idx

    def __len__(self):
        return np.load(self.filename, mmap_mode='r').shape[-1]

    def __getitem__(self, index):
        data = np.load(self.filename, mmap_mode='r')
        if self.return_2d:
            return torch.Tensor(data[None, :, :, self.cut_idx, index])
        else:
            return torch.Tensor(data[None, :, :, :, index])


class NumpyDatasetMem(Dataset):
    def __init__(self, filename, return_2d=False, cut_idx=40):
        temp = np.load(filename, mmap_mode='r')
        if return_2d:
            self.data = torch.Tensor(temp[:, :, cut_idx, :])
        else:
            self.data = torch.Tensor(temp)
        self.return_2d = return_2d
        self.cut_idx = cut_idx

    def __len__(self):
        return self.data.shape[-1]

    def __getitem__(self, index):
        if self.return_2d:
            return self.data[None, :, :, index]
        else:
            return self.data[None, :, :, :, index]


# TODO: Fix this to take into account of 2D mask
def get_dataset(subject=100307, data_dir=None, in_memory=False,
                return_2d=True, cut_idx=40):
    if in_memory:
        dataset_type = NumpyDatasetMem
    else:
        dataset_type = NumpyDataset
    if data_dir is None:
        data_dir = expanduser('~/data/HCP_masked')
    datasets = []
    for filename in glob.glob(join(data_dir, '%s_REST*.npy' % subject)):
        datasets.append(dataset_type(filename, return_2d=return_2d,
                                     cut_idx=cut_idx))
    train_dataset = ConcatDataset(datasets[:-1])
    test_dataset = datasets[-1]
    mask = np.load(expanduser('~/data/HCP_masked/%s_mask.npy'
                              % subject))
    if return_2d:
        mask = mask[:, :, cut_idx]
    mask = mask.astype('bool')
    print('Mask', mask.astype('float').sum(), 'voxels')
    for i in range(2):
        mask = binary_dilation(mask)
    mask = mask.astype('uint8')
    print('Dilated mask', mask.astype('float').sum(), 'voxels')
    mask = torch.from_numpy(mask).byte()

    return train_dataset, test_dataset, mask


def load_cut_data(cut_idx=45, voxel_standardize=True):
    subject_100307_folder = Path("~/data/HCP_masked").expanduser()
    rest_filepaths = sorted(subject_100307_folder.glob("*100307_REST*.npy"))

    slice_folder = Path("~/build/").expanduser() / f"slice_{cut_idx:03d}"
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
    mask = np.load(expanduser('~/data/HCP_masked/100307_mask.npy'))
    mask = mask[:, :, cut_idx]
    mask = mask.astype('bool')
    print('Mask', mask.astype('float').sum(), 'voxels')
    for i in range(2):
        mask = binary_dilation(mask)
    mask = mask.astype('uint8')
    print('Dilated mask', mask.astype('float').sum(), 'voxels')
    mask = torch.from_numpy(mask).byte()

    return data_train, data_test, mask


class NumpyDataset2d(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[-1]

    def __getitem__(self, index):
        return torch.Tensor(self.dataset[None, :, :, index])
