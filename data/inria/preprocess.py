"""
This file preprocess the raw Inria Data and extract the images into small patches
"""


# Built-in
import os

# Libs
import h5py
import numpy as np
from tqdm import tqdm

# Own modules
from data import data_utils
from mrs_utils import misc_utils

# Settings
CITY_NAMES = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna',
              'bellingham', 'bloomington', 'innsbruck', 'sfo', 'tyrol-e']
CITY_IDS = [0, 1, 2, 3, 4]
SAVE_CITY = [CITY_NAMES[a] for a in CITY_IDS]
VAL_CITY = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
VAL_IDS = list(range(1, 6))


def patch_inria(data_dir, save_dir, patch_size, pad, overlap):
    """
    Preprocess the standard inria dataset
    :param data_dir: path to the original inria dataset
    :param save_dir: directory to save the extracted patches
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :param pad: #pixels to be padded around each tile, should be either one element or four elements
    :param overlap: #overlapping pixels between two patches in both vertical and horizontal direction
    :return:
    """
    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(save_dir, 'file_list_valid.txt'), 'w+')
    # get rgb and gt files
    for city_name in tqdm(SAVE_CITY, desc='City-wise'):
        for tile_id in tqdm(range(1, 37), desc='Tile-wise', leave=False):
            rgb_filename = os.path.join(data_dir, 'image', '{}{}.tif'.format(city_name, tile_id))
            gt_filename = os.path.join(data_dir, 'truth', '{}{}.tif'.format(city_name, tile_id))
            for rgb_patch, gt_patch, y, x in data_utils.patch_tile(rgb_filename, gt_filename, patch_size, pad, overlap):
                rgb_patchname = '{}{}_y{}x{}.jpg'.format(city_name, tile_id, int(y), int(x))
                gt_patchname = '{}{}_y{}x{}.png'.format(city_name, tile_id, int(y), int(x))
                misc_utils.save_file(os.path.join(patch_dir, rgb_patchname), rgb_patch.astype(np.uint8))
                misc_utils.save_file(os.path.join(patch_dir, gt_patchname), (gt_patch/255).astype(np.uint8))
                if city_name in VAL_CITY and tile_id in VAL_IDS:
                    record_file_valid.write('{} {}\n'.format(rgb_patchname, gt_patchname))
                else:
                    record_file_train.write('{} {}\n'.format(rgb_patchname, gt_patchname))
    record_file_train.close()
    record_file_valid.close()


def get_images(data_dir, city_ids=tuple(range(5)), tile_ids=tuple(range(1, 6))):
    rgb_files = []
    gt_files = []
    for city_name in [CITY_NAMES[i] for i in city_ids]:
        for tile_id in tile_ids:
            rgb_filename = os.path.join(data_dir, 'image', '{}{}.tif'.format(city_name, tile_id))
            gt_filename = os.path.join(data_dir, 'truth', '{}{}.tif'.format(city_name, tile_id))
            if city_name in VAL_CITY and tile_id in VAL_IDS:
                rgb_files.append(rgb_filename)
                gt_files.append(gt_filename)
    return rgb_files, gt_files


if __name__ == '__main__':
    ps = 512
    pd = 0
    ol = 0
    save_dir = os.path.join(r'/hdd/mrs/inria', 'ps{}_pd{}_ol{}'.format(ps, pd, ol))
    misc_utils.make_dir_if_not_exist(save_dir)
    patch_inria(data_dir=r'/media/ei-edl01/data/remote_sensing_data/inria',
                save_dir=save_dir,
                patch_size=(ps, ps),
                pad=pd, overlap=ol)
