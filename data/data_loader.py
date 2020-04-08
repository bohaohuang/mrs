"""
This file defines data loader for some benchmarked remote sensing datasets
"""


# Built-in
import os

# Libs
import torch
import h5py
import numpy as np
from torch.utils import data

# Own modules
from mrs_utils import misc_utils


def get_file_paths(parent_path, file_list, with_label=True):
    """
    Parse the paths into absolute paths
    :param parent_path: the parent paths of all the data files
    :param file_list: the list of files
    :param with_label: if True, the label files will also be returned
    :return:
    """
    img_list = []
    lbl_list = []
    for fl in file_list:
        if with_label:
            img_filename, lbl_filename = [os.path.join(parent_path, a) for a in fl.strip().split(' ')[:2]]
            lbl_list.append(lbl_filename)
        else:
            img_filename = [os.path.join(parent_path, a) for a in fl.strip().split(' ')[:1]][0]
        img_list.append(img_filename)
    return img_list, lbl_list


def one_hot(class_n, x):
    """
    Make scalar into one-hot vector
    :param class_n: number of classes
    :param x: the scalar
    :return: converted one-hot vector
    """
    return torch.eye(class_n)[x]


class RSDataLoader(data.Dataset):
    def __init__(self, parent_path, file_list, transforms=None, n_class=2, with_label=True, with_aux=False):
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
        :param n_class: if greater than 0, will yield a #classes dimension vector where 1 indicates corresponding class exist
        :param with_label: if True, label files will be read, otherwise label files will be ignored
        :param with_aux: if True, auxiliary classification label will be returned
        """
        self.with_label = with_label
        try:
            file_list = misc_utils.load_file(file_list)
            self.img_list, self.lbl_list = get_file_paths(parent_path, file_list, self.with_label)
        except OSError:
            file_list = eval(file_list)
            parent_path = eval(parent_path)
            self.img_list, self.lbl_list = [], []
            for fl, pp in zip(file_list, parent_path):
                img_list, lbl_list = get_file_paths(pp, misc_utils.load_file(fl))
                self.img_list.extend(img_list)
                self.lbl_list.extend(lbl_list)
        self.transforms = transforms
        self.n_class = n_class
        self.with_aux = with_aux

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        output_dict = dict()
        output_dict['image'] = misc_utils.load_file(self.img_list[index])
        if self.with_label:
            output_dict['mask'] = misc_utils.load_file(self.lbl_list[index])
        if self.transforms:
            for tsfm in self.transforms:
                tsfm_image = tsfm(**output_dict)
                for key, val in tsfm_image.items():
                    output_dict[key] = val
        if self.with_aux:
            if len(output_dict['mask'].shape) == 2:
                cls = int(torch.mean(output_dict['mask'].type(torch.float)) > 0)
                cls = one_hot(self.n_class, cls).type(torch.float)
            else:
                cls = (torch.sum(output_dict['mask'], dim=-1) > 0).type(torch.float)
            output_dict['cls'] = cls
        return output_dict


def infi_loop_loader(dl):
    """
    An iterator that reloads after reaching to the end
    :param dl: data loader
    :return: an endless iterator
    """
    while True:
        for x in dl: yield x


class HDF5DataLoader(data.Dataset):
    def __init__(self, parent_path, file_list, transforms=None, n_class=0):
        """
        A data reader for the remote sensing dataset in hdf5 format
        Training with hdf5 data is generally >10% faster
        Normally the downloaded remote sensing dataset needs to be preprocessed
        :param parent_path: path to a preprocessed remote sensing dataset
        :param file_list: a text file where each row contains rgb and gt files separated by space
        :param transforms: albumentation transforms
        :param n_class: if greater than 0, will yield a #classes dimension vector where 1 indicates corresponding class exist
        """
        self.file_path = os.path.join(parent_path, file_list)
        self.transforms = transforms
        self.dataset = None
        self.n_class = n_class
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = file['img'].shape[0]

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        rgb = self.dataset['img'][index, ...]
        lbl = self.dataset['lbl'][index, ...]
        if self.transforms:
            for tsfm in self.transforms:
                tsfm_image = tsfm(image=rgb, mask=lbl)
                rgb = tsfm_image['image']
                lbl = tsfm_image['mask']
        if self.n_class:
            if len(lbl.shape) == 2:
                cls = int(torch.mean(lbl.type(torch.float)) > 0)
                cls = one_hot(self.n_class, cls).type(torch.float)
            else:
                cls = (torch.sum(lbl, dim=-1) > 0).type(torch.float)
            return rgb, lbl, cls
        else:
            return rgb, lbl


def get_loader(data_path, file_name, transforms=None, n_class=2, with_aux=False):
    """
    Get the appropriate loader with the given file type
    :param data_path: path to a preprocessed remote sensing dataset
    :param file_name: name of the data file, could be a text file or hdf5 file
    :param transforms: albumentation transforms
    :param aux_loss: if > 0, the dataloader will return patch-wise classification label
    :return: the corresponding loader
    """
    if file_name[-3:] == 'txt':
        return RSDataLoader(data_path, file_name, transforms, n_class, with_aux=with_aux)
    elif file_name[-4:] == 'hdf5':
        return HDF5DataLoader(data_path, file_name, transforms, n_class)
    elif file_name[-1] == ']':
        # multi dataset
        return RSDataLoader(data_path, file_name, transforms, n_class, with_aux=with_aux)
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
    from data import data_utils
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    tsfms = [A.RandomCrop(512, 512), ToTensorV2()]
    ds1 = RSDataLoader(r'/hdd/mrs/inria/ps512_pd0_ol0/patches', r'/hdd/mrs/inria/ps512_pd0_ol0/file_list_train_kt.txt', transforms=tsfms, n_class=2)

    for cnt, data_dict in enumerate(ds1):
        from mrs_utils import vis_utils
        rgb, gt, cls = data_dict['image'], data_dict['mask'], data_dict['cls']
        print(cls)
        vis_utils.compare_figures([data_utils.change_channel_order(rgb.cpu().numpy()),
                                   gt.cpu().numpy()], (1, 2), fig_size=(12, 5))
