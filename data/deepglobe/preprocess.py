"""

"""


# Built-in
import os
import subprocess
from glob import glob

# Libs
import numpy as np
from tqdm import tqdm
from natsort import natsorted

# Own modules
from mrs_utils import misc_utils, process_block

# Settings
DS_NAME = 'DeepGlobe'
CITY_DICT = {'Vegas': '2_Vegas', 'Paris': '3_Paris', 'Shanghai': '4_Shanghai', 'Khartoum': '5_Khartoum'}
MEAN = (0.34391829, 0.41294382, 0.45617783)
STD = (0.10493991, 0.09446405, 0.08782307)


def get_image_gt(data_dir, city_names, target='buildings', valid_percent=0.4):
    ds_train = []
    ds_valid = []

    for city_name in city_names:
        rgb_dir = os.path.join(data_dir, 'AOI_{}_Train'.format(CITY_DICT[city_name]), 'RGB-PanSharpen')
        gt_dir = os.path.join(data_dir, 'AOI_{}_Train'.format(CITY_DICT[city_name]), 'geojson')
        rgb_files = natsorted(glob(os.path.join(rgb_dir, '*.tif')))
        gt_files = natsorted(glob(os.path.join(gt_dir, target, '*.geojson')))

        assert len(rgb_files) == len(gt_files)
        valid_size = int(len(rgb_files) * valid_percent)
        for cnt, (rgb_file, gt_file) in enumerate(zip(rgb_files, gt_files)):
            if cnt < valid_size:
                ds_valid.append([rgb_file, gt_file])
            else:
                ds_train.append([rgb_file, gt_file])

    return ds_train, ds_valid


def convert_gtif_to_8bit(src_raster_path, dst_raster_path):
    """
    Convert geotiff images (11bits) into 8bit uint8 images
    This function comes from https://github.com/motokimura/spacenet_building_detection/
    :param src_raster_path:
    :param dst_raster_path:
    :return:
    """
    from osgeo import gdal

    srcRaster = gdal.Open(src_raster_path)

    outputPixType = 'Byte'
    outputFormat = 'JPEG'

    cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat, '-co', '"PHOTOMETRIC=rgb"', '-q']
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId + 1
        band = srcRaster.GetRasterBand(bandId)
        min = band.GetMinimum()
        max = band.GetMaximum()

        # if not exist minimum and maximum values
        if min is None or max is None:
            (min, max) = band.ComputeRasterMinMax(1)

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(max))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(src_raster_path)
    cmd.append(dst_raster_path)
    return subprocess.call(cmd)


def check_blank_region(img):
    h, w, _ = img.shape
    return np.sum(np.sum(img, axis=-1) == 0) / (h * w)


def make_dataset(ds_train, ds_valid, save_dir, th=0.5):
    import solaris as sol

    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(save_dir, 'file_list_valid.txt'), 'w+')

    # remove counting
    remove_train_cnt = 0
    remove_valid_cnt = 0

    # make dataset
    ds_dict = {
        'train': {'ds': ds_train, 'record': record_file_train, 'remove_cnt': remove_train_cnt},
        'valid': {'ds': ds_valid, 'record': record_file_valid, 'remove_cnt': remove_valid_cnt}
    }

    # valid ds
    for phase in ['valid', 'train']:
        for rgb_file, gt_file in tqdm(ds_dict[phase]['ds']):
            img_save_name = os.path.join(patch_dir, '{}.jpg'.format(os.path.splitext(os.path.basename(rgb_file))[0]))
            lbl_save_name = os.path.join(patch_dir, '{}.png'.format(os.path.splitext(os.path.basename(rgb_file))[0]))
            convert_gtif_to_8bit(rgb_file, img_save_name)
            img = misc_utils.load_file(img_save_name)
            lbl = sol.vector.mask.footprint_mask(df=gt_file, reference_im=rgb_file)

            # from mrs_utils import vis_utils
            # vis_utils.compare_figures([img, lbl], (1, 2), fig_size=(12, 5))

            blank_region = check_blank_region(img)
            if blank_region > th:
                ds_dict[phase]['remove_cnt'] += 1
                os.remove(img_save_name)
            else:
                if img.shape[0] != lbl.shape[0] or img.shape[1] != lbl.shape[1]:
                    assert np.unique(lbl) == np.array([0])
                    lbl = lbl[:img.shape[0], :img.shape[1]]
                misc_utils.save_file(os.path.join(patch_dir, lbl_save_name), (lbl / 255).astype(np.uint8))
                ds_dict[phase]['record'].write('{} {}\n'.format(os.path.basename(img_save_name),
                                                                os.path.basename(lbl_save_name)))
        ds_dict[phase]['record'].close()
        print('{} set: {:.2f}% data removed with threshold of {}'.format(
            phase, ds_dict[phase]['remove_cnt']/len(ds_dict[phase]['ds']), th))
        print('\t kept patches: {}'.format(len(ds_dict[phase]['ds']) - ds_dict[phase]['remove_cnt']))

        files_remove = glob(os.path.join(patch_dir, '*.aux.xml'))
        for f in files_remove:
            os.remove(f)


def get_stats(img_dir):
    from data import data_utils
    rgb_imgs = natsorted(glob(os.path.join(img_dir, '*.jpg')))
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    return np.stack([ds_mean, ds_std], axis=0)


def get_stats_pb(img_dir=r'/hdd/mrs/deepglobe/14p_pd0_ol0/patches'):
    val = process_block.ValueComputeProcess(
        DS_NAME, os.path.join(os.path.dirname(__file__), '../stats/builtin'),
        os.path.join(os.path.dirname(__file__), '../stats/builtin/{}.npy'.format(DS_NAME)), func=get_stats).\
        run(img_dir=img_dir).val
    val_test = val
    return val, val_test


def get_images(data_dir):
    record_file_valid = os.path.join(data_dir, 'file_list_valid.txt')
    file_list = misc_utils.load_file(record_file_valid)
    rgb_files, gt_files = [], []
    for line in file_list:
        rgb_file, gt_file = line.strip().split(' ')
        rgb_files.append(os.path.join(data_dir, 'patches', rgb_file))
        gt_files.append(os.path.join(data_dir, 'patches', gt_file))
    return rgb_files, gt_files


if __name__ == '__main__':
    save_dir = os.path.join(r'/hdd/mrs/deepglobe', 'pd{}_ol{}'.format(0, 0))
    train, valid = get_image_gt(r'/hdd/deepglobe', ['Vegas', 'Paris', 'Shanghai', 'Khartoum'], valid_percent=0.4)
    make_dataset(train, valid, save_dir)
