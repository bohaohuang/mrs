"""
This file preprocess the raw MNIH Massachusetts Roads Dataset and extract the images into small patches
"""


# Built-in
import os

# Libs
import numpy as np
from tqdm import tqdm

# Own modules
from data import data_utils
from mrs_utils import misc_utils

# Settings
DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data/mnih' # '/data/users/wh145/mnih/'
SPLITS = ['train', 'valid'] # test set will be grabbed by get_images() and processed during testing
MODES = os.listdir(os.path.join(DATA_DIR, SPLITS[0])) # sat (input), map (target)
MEAN = (0.4251811, 0.42812928, 0.39143909)
STD = (0.22423858, 0.21664895, 0.22102307)

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
    
    for dataset in tqdm(SPLITS, desc='Train-valid split'):
        FILENAMES = [
            fname.split('.')[0] for fname in os.listdir(os.path.join(DATA_DIR, dataset, MODES[0]))
        ]
        # create folders and files
        patch_dir = os.path.join(save_dir, 'patches')
        misc_utils.make_dir_if_not_exist(patch_dir)
        record_file = open(os.path.join(save_dir, 'file_list_{}.txt'.format(dataset)), 'w+')

        # get rgb and gt files
        for fname in tqdm(FILENAMES, desc='File-wise'):
            rgb_filename = os.path.join(DATA_DIR, dataset, 'sat', fname+'.tiff')
            gt_filename = os.path.join(DATA_DIR, dataset, 'map', fname+'.tif')
            for rgb_patch, gt_patch, y, x in patch_tile(rgb_filename, gt_filename, patch_size, pad, overlap):
                rgb_patchname = '{}_y{}x{}.jpg'.format(fname, int(y), int(x))
                gt_patchname = '{}_y{}x{}.png'.format(fname, int(y), int(x))
                misc_utils.save_file(os.path.join(patch_dir, rgb_patchname), rgb_patch.astype(np.uint8))
                misc_utils.save_file(os.path.join(patch_dir, gt_patchname), (gt_patch/255).astype(np.uint8))
                record_file.write('{} {}\n'.format(rgb_patchname, gt_patchname))
        record_file.close()


def get_images(data_dir=DATA_DIR, dataset='test'):
    """
    Stand-alone function to be used in evaluate.py.
    :param data_dir
    :param dataset: name of the dataset/split
    """    
    rgb_files = []
    gt_files = []
    file_list = os.listdir(os.path.join(data_dir, dataset, 'map'))
    for fname in file_list:
        gt_files.append(os.path.join(data_dir, dataset, 'map' ,fname))
        rgb_files.append(os.path.join(data_dir, dataset, 'sat' ,fname.replace('tif', 'tiff')))
    return rgb_files, gt_files


def get_stats(img_dir):
    from data import data_utils
    from glob import glob
    rgb_imgs = []
    for set_name in ['train', 'valid', 'test']:
        rgb_imgs.extend(glob(os.path.join(img_dir, set_name, 'sat', '*.tiff')))
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    print('Mean: {}'.format(ds_mean))
    print('Std: {}'.format(ds_std))


if __name__ == '__main__':
    '''ps = 512
    pd = 0
    ol = 0
    save_dir = r'/data/users/wh145/processed_mnih/'
    misc_utils.make_dir_if_not_exist(save_dir)
    patch_mnih(data_dir=DATA_DIR,
               save_dir=save_dir,
               patch_size=(ps, ps),
               pad=pd, overlap=ol)'''

    get_stats(r'/media/ei-edl01/data/remote_sensing_data/mnih')
