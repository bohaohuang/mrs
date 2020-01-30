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
from mrs_utils import misc_utils

# Settings
MEAN = (0.40851371, 0.37964116, 0.28266888)
STD = (0.12667853, 0.10076384, 0.08919973)
CLASS_NAMES = ['Urbanland', 'Agricultureland', 'Rangeland', 'Forestland', 'Water', 'Barrenland', 'Unknown']

# Decoder
DECODER = {
    255255: 0,          # Urban land
    255255000: 1,       # Agriculture land
    255000255: 2,       # Rangeland
    255000: 3,          # Forest land
    255: 4,             # Water
    255255255: 5,       # Barren land
    0: 6                # Unknown
}

ENCODER = {
    0: (0, 255, 255),           # Urban land
    1: (255, 255, 0),           # Agriculture land
    2: (255, 0, 255),           # Rangeland
    3: (0, 255, 0),             # Forest land
    4: (0, 0, 255),             # Water
    5: (255, 255, 255),         # Barren land
    6: (0, 0, 0)                # Unknown
}


def decode_map(gt_map, decoder=DECODER):
    dc_func = lambda x: decoder[x]
    gt_map = gt_map.astype(np.float)
    new_map = gt_map[:, :, 0] * 1000000 + gt_map[:, :, 1] * 1000 + gt_map[:, :, 2]
    return np.vectorize(dc_func)(new_map.astype(np.long))


def encode_map(gt, encoder=ENCODER):
    dc_func = lambda x: encoder[x]
    return np.stack(np.vectorize(dc_func)(gt.astype(np.long)), axis=-1)


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
    gt_mask = misc_utils.load_file(gt_file)
    gt = decode_map(gt_mask, DECODER)

    np.testing.assert_array_equal(rgb.shape[:2], gt.shape[:2])
    grid_list = data_utils.make_grid(np.array(rgb.shape[:2]) + 2 * pad, patch_size, overlap)
    if pad > 0:
        rgb = data_utils.pad_image(rgb, pad)
        gt = data_utils.pad_image(gt, pad)
    for y, x in grid_list:
        rgb_patch = data_utils.crop_image(rgb, y, x, patch_size[0], patch_size[1])
        gt_patch = data_utils.crop_image(gt, y, x, patch_size[0], patch_size[1])

        yield rgb_patch, gt_patch, y, x


def patch_deepglobeland(data_dir, save_dir, patch_size, pad, overlap, valid_percent=0.14):
    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(save_dir, 'file_list_valid.txt'), 'w+')

    # make folds
    files = data_utils.get_img_lbl(os.path.join(data_dir, 'land-train', 'land-train'), 'sat.jpg', 'mask.png')
    valid_size = int(len(files) * valid_percent)

    for cnt, (img_file, lbl_file) in enumerate(tqdm(files)):
        city_name = os.path.splitext(os.path.basename(img_file))[0].split('_')[0]
        for rgb_patch, gt_patch, y, x in patch_tile(img_file, lbl_file, patch_size, pad, overlap):
            img_patchname = '{}_y{}x{}.jpg'.format(city_name, int(y), int(x))
            lbl_patchname = '{}_y{}x{}.png'.format(city_name, int(y), int(x))
            misc_utils.save_file(os.path.join(patch_dir, img_patchname), rgb_patch.astype(np.uint8))
            misc_utils.save_file(os.path.join(patch_dir, lbl_patchname), gt_patch.astype(np.uint8))

            if cnt < valid_size:
                record_file_valid.write('{} {}\n'.format(img_patchname, lbl_patchname))
            else:
                record_file_train.write('{} {}\n'.format(img_patchname, lbl_patchname))
    record_file_train.close()
    record_file_valid.close()


def get_stats(img_dir):
    from data import data_utils
    dirs = ['land-train/land-train',]
    rgb_imgs = []
    for dir_ in dirs:
        rgb_imgs.extend([a[0] for a in data_utils.get_img_lbl(os.path.join(img_dir, dir_), 'sat.jpg', 'mask.png')])
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    print('Mean: {}'.format(ds_mean))
    print('Std: {}'.format(ds_std))


def get_images(data_dir, valid_percent=0.14):
    files = data_utils.get_img_lbl(os.path.join(data_dir, 'land-train', 'land-train'), 'sat.jpg', 'mask.png')
    valid_size = int(len(files) * valid_percent)
    rgb_files, gt_files = [], []
    for cnt, (img_file, lbl_file) in enumerate(files):
        if cnt < valid_size:
            rgb_files.append(img_file)
            gt_files.append(lbl_file)
    return rgb_files, gt_files


if __name__ == '__main__':
    ps = 512
    pd = 0
    ol = 0
    save_dir = os.path.join(r'/hdd/mrs/deepglobeland', 'ps{}_pd{}_ol{}'.format(ps, pd, ol))
    patch_deepglobeland(data_dir=r'/media/ei-edl01/data/remote_sensing_data/DGLand',
                        save_dir=save_dir,
                        patch_size=(ps, ps),
                        pad=pd, overlap=ol)
    get_stats(r'/media/ei-edl01/data/remote_sensing_data/DGLand')
