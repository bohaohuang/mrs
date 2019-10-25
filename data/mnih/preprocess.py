"""
This file preprocess the raw MNIH Massachusetts Roads Dataset and extract the images into small patches
"""


# Built-in
import os
import csv

# Libs
import numpy as np
from tqdm import tqdm

# Own modules
from data import data_utils
from mrs_utils import misc_utils

# Settings
DATA_DIR = '/data/users/wh145/mass_roads/'
SPLITS = os.listdir(DATA_DIR) # train, valid, test
MODES = os.listdir(os.path.join(DATA_DIR, SPLITS[0])) # sat (ipt), map(tgt)
TILES_PER_FILE = 5
print(MODES)

def patch_tile(rgb_file, gt_file, patch_size, pad, overlap):
    """
    Extract the given rgb and gt tiles into patches
    :param rgb_file: path to the rgb file
    :param gt_file: path to the gt file
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :param pad: #pixels to be padded around each tile, should be either one element or four elements
    :param overlap: #overlapping pixels between two patches in both vertical and horizontal direction
    :return: rgb and gt patches as well as coordinates
    """
    rgb = misc_utils.load_file(rgb_file)
    gt = misc_utils.load_file(gt_file)
    np.testing.assert_array_equal(rgb.shape[:2], gt.shape)
    grid_list = data_utils.make_grid(np.array(rgb.shape[:2]) + 2 * pad, patch_size, overlap)
    if pad > 0:
        rgb = data_utils.pad_image(rgb, pad)
        gt = data_utils.pad_image(gt, pad)
    for y, x in grid_list:
        rgb_patch = data_utils.crop_image(rgb, y, x, patch_size[0], patch_size[1])
        gt_patch = data_utils.crop_image(gt, y, x, patch_size[0], patch_size[1])
        yield rgb_patch, gt_patch, y, x


def patch_mnih(data_dir, save_dir, patch_size, pad, overlap):
    """
    Preprocess the standard mnih dataset
    :param data_dir: path to the original mnih dataset
    :param save_dir: directory to save the extracted patches
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :param pad: #pixels to be padded around each tile, should be either one element or four elements
    :param overlap: #overlapping pixels between two patches in both vertical and horizontal direction
    :return:
    """
    
    for dataset in tqdm(SPLITS, desc='Train-valid-test split'):
        FILENAMES = [
            fname.split('.')[0] for fname in os.listdir(os.path.join(DATA_DIR, dataset, MODES[0]))
        ]
        NUM_FILES = len(FILENAMES)    
        # create folders and files
        patch_dir = os.path.join(save_dir, dataset)
        misc_utils.make_dir_if_not_exist(patch_dir)
        record_file = open(os.path.join(save_dir, 'file_list_{}.txt'.format(dataset)), 'w+')

        # get rgb and gt files
        for fname in tqdm(FILENAMES, desc='File-wise'):
#             for tile_id in tqdm(range(TILES_PER_FILE-1), desc='Tile-wise', leave=False):
            rgb_filename = os.path.join(DATA_DIR, dataset, 'sat', fname+'.tiff')
            gt_filename = os.path.join(DATA_DIR, dataset, 'map', fname+'.tif')
            for rgb_patch, gt_patch, y, x in patch_tile(rgb_filename, gt_filename, patch_size, pad, overlap):
                rgb_patchname = '{}_y{}x{}.jpg'.format(fname, int(y), int(x))
                gt_patchname = '{}_y{}x{}.png'.format(fname, int(y), int(x))
                misc_utils.save_file(os.path.join(patch_dir, rgb_patchname), rgb_patch.astype(np.uint8))
                misc_utils.save_file(os.path.join(patch_dir, gt_patchname), (gt_patch/255).astype(np.uint8))
                record_file.write('{} {}\n'.format(rgb_patchname, gt_patchname))
        record_file.close()


def get_images(data_dir, dataset='test'):
    """
    Stand-alone function to be used in evaluate.py.
    :param data_dir
    :param dataset: name of the dataset/split
    """    
    rgb_files = []
    gt_files = []
    with open(os.path.join(data_dir, 'file_list_{}.txt'.format(dataset)), 'r') as f:
        for _, line in enumerate(f):
            rgb, gt = line.replace('\n', '').split(' ')
            rgb_files.append(os.path.join(data_dir, rgb))
            gt_files.append(os.path.join(data_dir, gt))
        f.close()
    return rgb_files, gt_files

if __name__ == '__main__':
    ps = 512
    pd = 0
    ol = 0
    save_dir = r'/data/users/wh145/processed_mass_roads/'
    misc_utils.make_dir_if_not_exist(save_dir)
    patch_mnih(data_dir=r'/data/users/wh145/mass_roads',
                save_dir=save_dir,
                patch_size=(ps, ps),
                pad=pd, overlap=ol)
