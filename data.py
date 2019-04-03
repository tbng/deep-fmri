import glob

import numpy as np
import torch
from os.path import expanduser, join
from skimage.morphology import dilation, binary_dilation
from torch.utils.data import Dataset, ConcatDataset


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

