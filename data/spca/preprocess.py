"""

"""


# Built-in
import os
from glob import glob

# Libs
import numpy as np
from tqdm import tqdm
from natsort import natsorted

# Own modules
from data import data_utils
from mrs_utils import misc_utils, process_block

# Settings
DS_NAME = 'spca'


def get_images(data_dir, valid_percent=0.5, split=False):
    rgb_files = natsorted(glob(os.path.join(data_dir, '*RGB.jpg')))
    lbl_files = natsorted(glob(os.path.join(data_dir, '*GT.png')))

    '''ind = np.arange(len(rgb_files))
    np.random.shuffle(ind)
    rgb_files = [rgb_files[a] for a in ind]
    lbl_files = [lbl_files[a] for a in ind]'''

    assert len(rgb_files) == len(lbl_files)
    city_names = ['Fresno', 'Modesto', 'Stockton']
    city_files = {city_name: [(rgb_file, lbl_file) for (rgb_file, lbl_file) in zip(rgb_files, lbl_files)
                              if city_name in rgb_file] for city_name in city_names}
    train_files, valid_files = [], []
    for city_name, file_pairs in city_files.items():
        valid_size = int(valid_percent * len(file_pairs))
        train_files.extend(file_pairs[valid_size:])
        valid_files.extend(file_pairs[:valid_size])
    if split:
        return train_files, valid_files
    else:
        return [a[0] for a in valid_files], [a[1] for a in valid_files]


def create_dataset(data_dir, save_dir, patch_size, pad, overlap, valid_percent=0.2, visualize=False):
    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(save_dir, 'file_list_train_{}_2.txt').format(
        misc_utils.float2str(valid_percent)), 'w+')
    record_file_valid = open(os.path.join(save_dir, 'file_list_valid_{}_2.txt').format(
        misc_utils.float2str(valid_percent)), 'w+')
    train_files, valid_files = get_images(data_dir, valid_percent, split=True)

    for img_file, lbl_file in tqdm(train_files):
        city_name = os.path.splitext(os.path.basename(img_file))[0].split('_')[0]
        for rgb_patch, gt_patch, y, x in data_utils.patch_tile(img_file, lbl_file, patch_size, pad, overlap):
            if visualize:
                from mrs_utils import vis_utils
                vis_utils.compare_figures([rgb_patch, gt_patch], (1, 2), fig_size=(12, 5))
            img_patchname = '{}_y{}x{}.jpg'.format(city_name, int(y), int(x))
            lbl_patchname = '{}_y{}x{}.png'.format(city_name, int(y), int(x))
            # misc_utils.save_file(os.path.join(patch_dir, img_patchname), rgb_patch.astype(np.uint8))
            # misc_utils.save_file(os.path.join(patch_dir, lbl_patchname), gt_patch.astype(np.uint8))
            record_file_train.write('{} {}\n'.format(img_patchname, lbl_patchname))

    for img_file, lbl_file in tqdm(valid_files):
        city_name = os.path.splitext(os.path.basename(img_file))[0].split('_')[0]
        for rgb_patch, gt_patch, y, x in data_utils.patch_tile(img_file, lbl_file, patch_size, pad, overlap):
            if visualize:
                from mrs_utils import vis_utils
                vis_utils.compare_figures([rgb_patch, gt_patch], (1, 2), fig_size=(12, 5))
            img_patchname = '{}_y{}x{}.jpg'.format(city_name, int(y), int(x))
            lbl_patchname = '{}_y{}x{}.png'.format(city_name, int(y), int(x))
            # misc_utils.save_file(os.path.join(patch_dir, img_patchname), rgb_patch.astype(np.uint8))
            # misc_utils.save_file(os.path.join(patch_dir, lbl_patchname), gt_patch.astype(np.uint8))
            record_file_valid.write('{} {}\n'.format(img_patchname, lbl_patchname))


def get_stats(img_dir):
    from data import data_utils
    from glob import glob
    rgb_imgs = glob(os.path.join(img_dir, '*RGB.jpg'))
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    return np.stack([ds_mean, ds_std], axis=0)


def get_stats_pb(img_dir):
    val = process_block.ValueComputeProcess(DS_NAME, os.path.join(os.path.dirname(__file__), '../stats/builtin'),
                                            os.path.join(os.path.dirname(__file__), '../stats/builtin/{}.npy'.format(DS_NAME)), func=get_stats).\
        run(img_dir=img_dir).val
    val_test = val
    return val, val_test


if __name__ == '__main__':
    img_files = natsorted(glob(os.path.join(r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles', '*RGB.jpg')))

    np.random.seed(931004)
    ps = 512
    ol = 0
    pd = 0
    create_dataset(data_dir=r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles',
                   save_dir=r'/hdd/mrs/spca', patch_size=(ps, ps), pad=pd, overlap=ol, visualize=False, valid_percent=0.5)

    # val = get_stats_pb(r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles')[0]

    # data_utils.patches_to_hdf5(r'/hdd/mrs/spca', r'/hdd/mrs/spca/ps512_pd0_ol0_hdf5')
