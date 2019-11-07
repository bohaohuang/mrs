"""
This file defines data loader for some benchmarked remote sensing datasets
"""


# Built-in
import os

# Libs
import h5py
from torch.utils import data

# Own modules
from data import data_utils
from mrs_utils import misc_utils


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
        self.file_list = misc_utils.load_file(file_list)
        self.parent_path = parent_path
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        rgb_filename, gt_filename = [os.path.join(self.parent_path, a)
                                     for a in self.file_list[index].strip().split(' ')]
        rgb = misc_utils.load_file(rgb_filename)
        gt = misc_utils.load_file(gt_filename)
        if self.transforms:
            for tsfm in self.transforms:
                tsfm_image = tsfm(image=rgb, mask=gt)
                rgb = tsfm_image['image']
                gt = tsfm_image['mask']
        return rgb, gt


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
    else:
        raise NotImplementedError('File extension {} is not supportted yet'.format(os.path.splitext(file_name))[-1])


if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    tsfm = A.Compose([
        A.Flip(),
        A.RandomRotate90(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    ds = RSDataLoader(r'/hdd/mrs/inria/ps512_pd0_ol0/patches', r'/hdd/mrs/inria/ps512_pd0_ol0/file_list_valid.txt', transforms=tsfm)
    # ds = RSDataLoader(r'/hdd/pgm/nz_ps512_ol0/patches', r'/hdd/pgm/nz_ps512_ol0/file_list_valid.txt', transforms=tsfm)
    for cnt, (rgb, gt) in enumerate(ds):
        data_utils.visualize(rgb, gt)

        if cnt == 10:
            break
