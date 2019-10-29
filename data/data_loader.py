"""
This file defines data loader for some benchmarked remote sensing datasets
"""


# Built-in
import os

# Libs
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


class RSDataLoaderMB(data.Dataset):
    def __init__(self, parent_path, file_list, batch_ratio, transforms=None):
        """
        A mixed batch reader which reads data from two separate datasets with a certain ratio
        E.g., n data from dataset 1 and m data from dataset 2 for each batch where n+m=batch size
        Each remote sensing dataset structure should be like
        /parent_path
            /patches
                img0.png
                img1.png
            file_list.txt
        WARNING: DO NOT SHUFFLE THE DATALOADER SINCE IT WILL DESTROY THE SAMPLE DISTRIBUTION
        Normally the downloaded remote sensing dataset needs to be preprocessed
        :param parent_path: a list of paths to a preprocessed remote sensing dataset
        :param file_list: a list of text files where each row contains rgb and gt files separated by space
        :param batch_ratio: a list of integers where each item is the #samples from corresponding dataset in a batch
        :param transforms: albumentation transforms
        """
        self.file_list = []
        for fl in file_list:
            self.file_list.append(misc_utils.load_file(fl))
        self.parent_path = parent_path
        self.transforms = transforms
        self.batch_ratio = batch_ratio

    def __len__(self):
        return len(self.file_list)

    def arrange_order(self):
        pass

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
