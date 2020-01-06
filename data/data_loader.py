"""
This file defines data loader for some benchmarked remote sensing datasets
"""


# Built-in
import os

# Libs
import h5py
import numpy as np
from torch.utils import data

# Own modules
from mrs_utils import misc_utils


def get_file_paths(parent_path, file_list):
    """
    Parse the paths into absolute paths
    :param parent_path: the parent paths of all the data files
    :param file_list: the list of files
    :return:
    """
    img_list = []
    lbl_list = []
    for fl in file_list:
        img_filename, lbl_filename = [os.path.join(parent_path, a) for a in fl.strip().split(' ')]
        img_list.append(img_filename)
        lbl_list.append(lbl_filename)
    return img_list, lbl_list


class RSDataLoader(data.Dataset):
    def __init__(self, parent_path, file_list, transforms=None):
        """
        A data reader for the remote sensing dataset
        The dataset storage structure should be like
        /parent_path
            /patches
                img0.png
                img1.png
            file_list.txt
        Normally the downloaded remote sensing dataset needs to be preprocessed
        :param parent_path: path to a preprocessed remote sensing dataset
        :param file_list: a text file where each row contains rgb and gt files separated by space
        :param transforms: albumentation transforms
        """
        try:
            file_list = misc_utils.load_file(file_list)
            self.img_list, self.lbl_list = get_file_paths(parent_path, file_list)
        except OSError:
            file_list = eval(file_list)
            parent_path = eval(parent_path)
            self.img_list, self.lbl_list = [], []
            for fl, pp in zip(file_list, parent_path):
                img_list, lbl_list = get_file_paths(pp, misc_utils.load_file(fl))
                self.img_list.extend(img_list)
                self.lbl_list.extend(lbl_list)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        rgb = misc_utils.load_file(self.img_list[index])
        lbl = misc_utils.load_file(self.lbl_list[index])
        if self.transforms:
            for tsfm in self.transforms:
                tsfm_image = tsfm(image=rgb, mask=lbl)
                rgb = tsfm_image['image']
                lbl = tsfm_image['mask']
        return rgb, lbl


class HDF5DataLoader(data.Dataset):
    def __init__(self, parent_path, file_list, transforms=None):
        """
        A data reader for the remote sensing dataset in hdf5 format
        Training with hdf5 data is generally >10% faster
        Normally the downloaded remote sensing dataset needs to be preprocessed
        :param parent_path: path to a preprocessed remote sensing dataset
        :param file_list: a text file where each row contains rgb and gt files separated by space
        :param transforms: albumentation transforms
        """
        self.file_path = os.path.join(parent_path, file_list)
        self.transforms = transforms
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = file['img'].shape[0]

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        rgb = self.dataset['img'][index, ...]
        gt = self.dataset['lbl'][index, ...]
        if self.transforms:
            for tsfm in self.transforms:
                tsfm_image = tsfm(image=rgb, mask=gt)
                rgb = tsfm_image['image']
                gt = tsfm_image['mask']
        return rgb, gt


def get_loader(data_path, file_name, transforms=None):
    """
    Get the appropriate loader with the given file type
    :param data_path: path to a preprocessed remote sensing dataset
    :param file_name: name of the data file, could be a text file or hdf5 file
    :param transforms: albumentation transforms
    :return: the corresponding loader
    """
    if file_name[-3:] == 'txt':
        return RSDataLoader(data_path, file_name, transforms)
    elif file_name[-4:] == 'hdf5':
        return HDF5DataLoader(data_path, file_name, transforms)
    elif file_name[-1] == ']':
        # multi dataset
        return RSDataLoader(data_path, file_name, transforms)
    else:
        raise NotImplementedError('File extension {} is not supportted yet'.format(os.path.splitext(file_name))[-1])


class MixedBatchSampler(data.sampler.Sampler):
    def __init__(self, ds_len, ratio):
        """
        Mixed batch sampler where each batch has fixed number of samples from each data source
        :param ds_len: length of each dataset
        :param ratio: #samples from each dataset, it should have same number of elements ds_len
        """
        super(MixedBatchSampler, self).__init__(ds_len)
        assert len(ds_len) == len(ratio)
        self.ds_len = ds_len
        self.ratio = ratio
        self.batch_size = sum(ratio)
        self.offset = [int(np.sum(self.ds_len[:a])) for a in range(len(ds_len))]

    def __len__(self):
        return self.ds_len[0]

    def __iter__(self):
        rand_idx = [np.random.permutation(np.arange(x)) for x in self.ds_len]
        for cnt in range(self.ds_len[0] // self.batch_size):
            for ds_cnt, n_sample in enumerate(self.ratio):
                for curr_cnt in range(n_sample):
                    yield self.offset[ds_cnt] + rand_idx[ds_cnt][(cnt*n_sample+curr_cnt) % self.ds_len[ds_cnt]]


if __name__ == '__main__':
    import albumentations as A

    tsfms = [A.RandomCrop(512, 512)]
    ds1 = RSDataLoader(r'/hdd/mrs/inria/ps512_pd0_ol0/patches', r'/hdd/mrs/inria/ps512_pd0_ol0/file_list_train_kt.txt', transforms=tsfms)
    ds2 = RSDataLoader(r'/hdd/mrs/synthinel_1/patches', r'/hdd/mrs/synthinel_1/file_list_train.txt', transforms=tsfms)
    ds3 = RSDataLoader(r'/hdd/mrs/inria/ps512_pd0_ol0/patches', r'/hdd/mrs/inria/ps512_pd0_ol0/file_list_valid_a.txt', transforms=tsfms)
    ds = data.ConcatDataset([ds1, ds2, ds3])
    loader = data.DataLoader(ds, sampler=MixedBatchSampler((len(ds1), len(ds2), len(ds3)), (5, 0, 0)), batch_size=5)

    for cnt, (rgb, gt) in enumerate(loader):
        from mrs_utils import vis_utils
        vis_utils.compare_figures(rgb, (1, 5), fig_size=(12, 6))
