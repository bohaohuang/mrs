"""
Process deepglobe raod extraction data
The raw images are in 1024*1024 pixels
The gt images have class 1 = 255
"""


# Built-in
import os

# Libs
import numpy as np
from tqdm import tqdm

# Own modules
from data import data_utils
from mrs_utils import misc_utils, process_block

# Settings
DS_NAME = 'DeepGlobeRoad'
MEAN = (0.40994515, 0.38314009, 0.28864455)
STD = (0.12889884, 0.10563929, 0.09726452)


def decode_map(gt_map):
    return gt_map[:, :, 0]


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
    gt = misc_utils.load_file(gt_file)[:, :, 0]
    np.testing.assert_array_equal(rgb.shape[:2], gt.shape)
    grid_list = data_utils.make_grid(np.array(rgb.shape[:2]) + 2 * pad, patch_size, overlap)
    if pad > 0:
        rgb = data_utils.pad_image(rgb, pad)
        gt = data_utils.pad_image(gt, pad)
    for y, x in grid_list:
        rgb_patch = data_utils.crop_image(rgb, y, x, patch_size[0], patch_size[1])
        gt_patch = data_utils.crop_image(gt, y, x, patch_size[0], patch_size[1])

        yield rgb_patch, gt_patch, y, x


def patch_deepgloberoad(data_dir, save_dir, patch_size, pad, overlap, valid_percent=0.14):
    dirs = ['road_trainv1/train', 'road_trainv2/train']

    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(save_dir, 'file_list_valid.txt'), 'w+')

    # make folds
    files = []
    for dir_ in dirs:
        files.extend(data_utils.get_img_lbl(os.path.join(data_dir, dir_), 'sat.jpg', 'mask.png'))
    valid_size = int(len(files) * valid_percent)

    for cnt, (img_file, lbl_file) in enumerate(tqdm(files)):
        city_name = os.path.splitext(os.path.basename(img_file))[0].split('_')[0]
        for rgb_patch, gt_patch, y, x in patch_tile(img_file, lbl_file, patch_size, pad, overlap):
            img_patchname = '{}_y{}x{}.jpg'.format(city_name, int(y), int(x))
            lbl_patchname = '{}_y{}x{}.png'.format(city_name, int(y), int(x))
            misc_utils.save_file(os.path.join(patch_dir, img_patchname), rgb_patch.astype(np.uint8))
            misc_utils.save_file(os.path.join(patch_dir, lbl_patchname), (gt_patch / 255).astype(np.uint8))

            if cnt < valid_size:
                record_file_valid.write('{} {}\n'.format(img_patchname, lbl_patchname))
            else:
                record_file_train.write('{} {}\n'.format(img_patchname, lbl_patchname))
    record_file_train.close()
    record_file_valid.close()


def get_stats(img_dir):
    from data import data_utils
    dirs = ['road_trainv1/train', 'road_trainv2/train']
    rgb_imgs = []
    for dir_ in dirs:
        rgb_imgs.extend([a[0] for a in data_utils.get_img_lbl(os.path.join(img_dir, dir_), 'sat.jpg', 'mask.png')])
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    return np.stack([ds_mean, ds_std], axis=0)


def get_images(data_dir, valid_percent=0.14):
    dirs = ['road_trainv1/train', 'road_trainv2/train']
    files = []
    for dir_ in dirs:
        files.extend(data_utils.get_img_lbl(os.path.join(data_dir, dir_), 'sat.jpg', 'mask.png'))
    valid_size = int(len(files) * valid_percent)
    rgb_files, gt_files = [], []
    for cnt, (img_file, lbl_file) in enumerate(files):
        if cnt < valid_size:
            rgb_files.append(img_file)
            gt_files.append(lbl_file)
    return rgb_files, gt_files


def get_stats_pb(img_dir):
    val = process_block.ValueComputeProcess(DS_NAME, os.path.join(os.path.dirname(__file__), '../stats/builtin'),
                                            os.path.join(os.path.dirname(__file__),
                                                         '../stats/builtin/{}.npy'.format(DS_NAME)),
                                            func=get_stats).\
        run(img_dir=img_dir).val
    val_test = val
    return val, val_test


if __name__ == '__main__':
    ps = 1024
    pd = 0
    ol = 0
    save_dir = os.path.join(r'/hdd/mrs/deepgloberoad', 'ps{}_pd{}_ol{}'.format(ps, pd, ol))
    patch_deepgloberoad(data_dir=r'/media/ei-edl01/data/remote_sensing_data/DeepGlobeRoad',
                        save_dir=save_dir,
                        patch_size=(ps, ps),
                        pad=pd, overlap=ol)
