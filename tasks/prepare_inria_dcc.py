"""

"""


# Built-in
import os

# Libs
from glob import glob
from natsort import natsorted

# Own modules
from data import patch_extractor


def get_city_name(file_name):
    return os.path.basename(file_name).split('_')[0]


def get_file_list(data_dir, gt_data_dir):
    file_list = []
    gt_files = natsorted(glob(os.path.join(gt_data_dir, '*GT_Divide.tif')))
    for gt_file in gt_files:
        rgb_file = os.path.join(data_dir, '{}_RGB.tif'.format(get_city_name(gt_file)))
        file_list.append([rgb_file, gt_file])
    return file_list


if __name__ == '__main__':
    patch_size = (224, 224)
    pad = 0
    overlap = 0
    save_path = r'/work/bh163/mrs/inria'
    data_dir = r'/work/bh163/uab_datasets/inria/data/Original_Tiles'
    gt_data_dir = r'/work/bh163/uab_datasets/inria/data/TilePreproc/MultChanOp_chans3_Divide_dF255p000'
    file_list = get_file_list(data_dir, gt_data_dir)
    patch_extractor.patch_extractor(file_list, ['jpg', 'png'], patch_size, pad, overlap, save_path, force_run=True)
