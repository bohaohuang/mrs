"""
This file defines data loader for some benchmarked remote sensing datasets
"""


# Built-in

# Libs
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Own modules
from data import patch_extractor
from mrs_utils import misc_utils


class JointFlip(object):
    """
    Random horizontal or vertical flip of a given image pair
    """
    def __call__(self, ftr, lbl):
        method = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        rand = np.random.randint(3)
        if rand != 2:
            ftr = ftr.transpose(method[rand])
            lbl = lbl.transpose(method[rand])
        return ftr, lbl


class JointRotate(object):
    """
    Random clockwise rotation of a given image pair
    """
    def __call__(self, ftr, lbl):
        method = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        rand = np.random.randint(4)
        if rand != 3:
            ftr = ftr.transpose(method[rand])
            lbl = lbl.transpose(method[rand])
        return ftr, lbl


class JointToTensor(object):
    """
    Cast given image pair to pytorch tensors
    Note: this function take cares of channel switching and data rescaling
    """
    def __call__(self, ftr, lbl):
        tt = torchvision.transforms.ToTensor()
        return tt(ftr), torch.from_numpy(np.array(lbl))


class JointCompose(object):
    """
    Compose torchvision transforms for a pair of given images
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ftr, lbl):
        for tsfm in self.transforms:
            ftr, lbl = tsfm(ftr, lbl)
        return ftr, lbl


class RemoteSensingDataset(Dataset):
    """Remote sensing dataset, the patch data files can be created by patch_extractor.py"""
    def __init__(self, txt_file=None, file_list=None, transform=None, transform_ftr=None):
        """
        Initialize a remote sensing dataset class
        :param txt_file: a file stores data files, each line is a sample, if this is None, file_,list cannot be None
        :param file_list: a list of lists that contains data files, each sublist is a sample
        :param transform: torchvision transforms or other compatible composed transforms, they will be applied to both
        feature and label
        :param transform_ftr: transforms (e.g., normalize) that only will be applied to feature not label
        """
        if not file_list:
            assert txt_file
            self.file_list = misc_utils.load_file(txt_file)
        else:
            self.file_list = file_list
        self.transform = transform
        self.transform_ftr = transform_ftr

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def stem_file_names(line):
        """
        Get rgb and gt name from a file string
        :param line: a string contains files for one sample
        :return: rgb file name and gt file name
        """
        return [misc_utils.stem_string(a, lower=False) for a in line.split(' ')]

    def __getitem__(self, idx):
        rgb_name, gt_name = self.stem_file_names(self.file_list[idx])
        rgb, gt = misc_utils.load_file(rgb_name, pil=True), misc_utils.load_file(gt_name, pil=True)
        if self.transform:
            rgb, gt = self.transform(rgb, gt)
        if self.transform_ftr:
            rgb = self.transform_ftr(rgb)
        return rgb, gt


class TileDataset(Dataset):
    """Read patches from a tile"""
    def __init__(self, data, input_size=(224, 224), pad=0, transform=None):
        """
        Patch reader of a given tile
        :param data: tile image
        :param input_size: size of the output shape
        :param pad: #pixels will be padded around the tile
        :param transform: torchvision transforms or other compatible composed transforms
        """
        self.data = data
        self.transform = transform
        self.input_size = input_size
        self.pad = pad
        if self.pad > 0:
            self.data = patch_extractor.pad_image(self.data, self.pad)
        tile_size = data.shape[:2]
        self.grid = patch_extractor.make_grid(tile_size, self.input_size, overlap=0)

    def __len__(self):
        return len(self.grid)

    def get_patch(self, patch_id):
        """
        Get patch of given patch_id, patch id created by patch_extractor.make_grid function
        :param patch_id: id of the given patch
        :return:
        """
        y, x = self.grid[patch_id]
        patch = patch_extractor.crop_image(self.data, y, x, *self.input_size)
        return patch

    def __getitem__(self, idx):
        patch = self.get_patch(idx)
        if self.transform:
            patch = self.transform(patch)
        return patch


class InfiniteDataLoader(DataLoader):
    """
    DataLoader wrapper, this will keep read data and never ends
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    file_list_name = r'/hdd/mrs/inria/file_list.txt'
    rsd = RemoteSensingDataset(file_list_name)
    composed = JointCompose([
        JointFlip(),
        JointRotate(),
        JointToTensor(),
    ])

    for rgb, gt in rsd:
        print(rgb.size, gt.size)

        print(np.unique(np.array(gt)))

        ftr, lbl = composed(rgb, gt)
        ftr = np.rollaxis(np.array(ftr.data.numpy()), 0, 3)
        lbl = np.rollaxis(np.array(lbl.data.numpy()), 0, 3)[:, :, 0]
        print(ftr.shape, lbl.shape)

        plt.subplot(221)
        plt.imshow(rgb)
        plt.subplot(222)
        plt.imshow(np.array(gt))
        plt.subplot(223)
        plt.imshow(ftr)
        plt.subplot(224)
        plt.imshow(lbl)
        plt.tight_layout()
        plt.show()
