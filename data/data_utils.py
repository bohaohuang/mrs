"""

"""


# Built-in
import os
from glob import glob

# Libs
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from torchvision import transforms

# Own modules
from mrs_utils import misc_utils, process_block


def make_grid(tile_size, patch_size, overlap):
    """
    Extract patches at fixed locations. Output coordinates for Y,X as a list (not two lists)
    :param tile_size: size of the tile (input image)
    :param patch_size: size of the output patch
    :param overlap: #overlapping pixels
    :return:
    """
    max_h = tile_size[0] - patch_size[0]
    max_w = tile_size[1] - patch_size[1]
    if max_h > 0 and max_w > 0:
        h_step = np.ceil(tile_size[0] / (patch_size[0] - overlap))
        w_step = np.ceil(tile_size[1] / (patch_size[1] - overlap))
    else:
        h_step = 1
        w_step = 1
    patch_grid_h = np.floor(np.linspace(0, max_h, h_step)).astype(np.int32)
    patch_grid_w = np.floor(np.linspace(0, max_w, w_step)).astype(np.int32)

    y, x = np.meshgrid(patch_grid_h, patch_grid_w)
    return list(zip(y.flatten(), x.flatten()))


def pad_image(img, pad, mode='reflect'):
    """
    Symmetric pad pixels around images
    :param img: image to pad
    :param pad: list of #pixels pad around the image, if it is a scalar, it will be assumed to pad same number
                number of pixels around 4 directions
    :param mode: padding mode
    :return: padded image
    """
    if type(pad) is not list:
        pad = [pad for i in range(4)]
    assert len(pad) == 4
    if len(img.shape) == 2:
        return np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])), mode)
    else:
        h, w, c = img.shape
        pad_img = np.zeros((h + pad[0] + pad[1], w + pad[2] + pad[3], c))
        for i in range(c):
            pad_img[:, :, i] = np.pad(img[:, :, i], ((pad[0], pad[1]), (pad[2], pad[3])), mode)
    return pad_img


def crop_image(img, y, x, h, w):
    """
    Crop the image with given top-left anchor and corresponding width & height
    :param img: image to be cropped
    :param y: height of anchor
    :param x: width of anchor
    :param h: height of the patch
    :param w: width of the patch
    :return:
    """
    if len(img.shape) == 2:
        return img[y:y+w, x:x+h]
    else:
        return img[y:y+w, x:x+h, :]


def change_channel_order(data, to_channel_last=True):
    """
    Switch the image type from channel first to channel last
    :param data: the data to switch the channels
    :param to_channel_last: if True, switch the first channel to the last
    :return: the channel switched data
    """
    if to_channel_last:
        if len(data.shape) == 3:
            return np.rollaxis(data, 0, 3)
        else:
            return np.rollaxis(data, 1, 4)
    else:
        if len(data.shape) == 3:
            return np.rollaxis(data, 2, 0)
        else:
            return np.rollaxis(data, 3, 1)


def visualize(rgb, gt, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.255)):
    """
    Visualize a given pair of image and mask normalized tensors
    :param rgb: the image tensor with shape [c, h, w]
    :param gt: the mask tensor with shape [1, h, w]ps512_pd0_ol0
    :param mean: the mean used to normalize the input
    :param std: the std used to normalize the input
    :return:
    """
    mean = [-a/b for a, b in zip(mean, std)]
    std = [1/a for a in std]
    inv_normalize = transforms.Normalize(
        mean=mean,
        std=std
    )
    rgb = inv_normalize(rgb)
    rgb, gt = rgb.numpy(), gt.numpy()
    rgb = change_channel_order(rgb, True)
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.imshow((rgb*255).astype(np.uint8))
    plt.subplot(122)
    plt.imshow(gt.astype(np.uint8))
    plt.tight_layout()
    plt.show()


def get_img_lbl(data_dir, img_ext, lbl_ext):
    """
    Get image and label pair in a given pair
    The image and label are identified by their given extensions
    :param data_dir: the directory of the dataset
    :param img_ext: extension of the image files
    :param lbl_ext: extension of the label files
    :return: list of image and label pairs
    """
    img_files = natsorted(glob(os.path.join(data_dir, '*{}'.format(img_ext))))
    lbl_files = natsorted(glob(os.path.join(data_dir, '*{}'.format(lbl_ext))))
    assert len(img_files) == len(lbl_files)
    return [(img_file, lbl_file) for (img_file, lbl_file) in zip(img_files, lbl_files)]


def get_ds_stats(img_files):
    """
    Get the dataset mean and standard deviation, this would be used for augmentation in data reader
    :param img_files: list of image files to compute the mean and standard deviation
    :return:
    """
    ds_mean = np.zeros(3)
    ds_std = np.zeros(3)

    for file in tqdm(img_files):
        img = misc_utils.load_file(file).astype(np.float32) / 255
        ds_mean = ds_mean + np.mean(img, axis=(0, 1))[:3]
        ds_std = ds_std + np.std(img, axis=(0, 1))[:3]

    ds_mean = ds_mean / len(img_files)
    ds_std = ds_std / len(img_files)

    return ds_mean, ds_std


def patch_tile(rgb_file, gt_file, patch_size, pad, overlap):
    """
    Extract the given rgb and gt tiles into patches
    :param rgb_file: path to the rgb file or the rgb imagery
    :param gt_file: path to the gt file or the gt mask
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :param pad: #pixels to be padded around each tile, should be either one element or four elements
    :param overlap: #overlapping pixels between two patches in both vertical and horizontal direction
    :return: rgb and gt patches as well as coordinates
    """
    if isinstance(rgb_file, str) and isinstance(gt_file, str):
        rgb = misc_utils.load_file(rgb_file)
        gt = misc_utils.load_file(gt_file)
    else:
        rgb = rgb_file
        gt = gt_file
    np.testing.assert_array_equal(rgb.shape[:2], gt.shape)
    grid_list = make_grid(np.array(rgb.shape[:2]) + 2 * pad, patch_size, overlap)
    if pad > 0:
        rgb = pad_image(rgb, pad)
        gt = pad_image(gt, pad)
    for y, x in grid_list:
        rgb_patch = crop_image(rgb, y, x, patch_size[0], patch_size[1])
        gt_patch = crop_image(gt, y, x, patch_size[0], patch_size[1])
        yield rgb_patch, gt_patch, y, x


def get_custom_ds_stats(ds_name, img_dir):
    def get_stats(img_dir):
        rgb_imgs = natsorted(glob(os.path.join(img_dir, '*.jpg')))
        ds_mean, ds_std = get_ds_stats(rgb_imgs)
        return np.stack([ds_mean, ds_std], axis=0)

    val = process_block.ValueComputeProcess(
        ds_name, os.path.join(os.path.dirname(__file__), './stats/custom'),
        os.path.join(os.path.dirname(__file__), './stats/custom/{}.npy'.format(ds_name)), func=get_stats). \
        run(img_dir=img_dir).val
    val_test = val

    return val, val_test


def create_toy_set(data_dir, train_file='file_list_train.txt', valid_file='file_list_valid.txt',
                   n_train=0.2, n_valid=0.2, random_seed=1, move_dir=None):
    np.random.seed(random_seed)
    train_list = misc_utils.load_file(os.path.join(data_dir, train_file))
    valid_list = misc_utils.load_file(os.path.join(data_dir, valid_file))
    origin_train_len = len(train_list)
    origin_valid_len = len(valid_list)
    if n_train < 1:
        n_train = int(origin_train_len * n_train)
    if n_valid < 1:
        n_valid = int(origin_valid_len * n_valid)

    n_train_select = np.sort(np.random.choice(origin_train_len, n_train, replace=False))
    n_valid_select = np.sort(np.random.choice(origin_valid_len, n_valid, replace=False))
    train_select = [train_list[a] for a in n_train_select]
    valid_select = [valid_list[a] for a in n_valid_select]
    misc_utils.save_file(os.path.join(data_dir, 'toy_'+train_file), train_select)
    misc_utils.save_file(os.path.join(data_dir, 'toy_'+valid_file), valid_select)

    if move_dir:
        from shutil import copyfile
        # need to move data to a new directory
        print('Moving toy dataset into {} ...'.format(move_dir))
        misc_utils.make_dir_if_not_exist(move_dir)
        patch_dir = os.path.join(move_dir, 'patches')
        misc_utils.make_dir_if_not_exist(patch_dir)
        misc_utils.save_file(os.path.join(move_dir, train_file), train_select)
        misc_utils.save_file(os.path.join(move_dir, valid_file), valid_select)

        for item in train_select:
            rgb, lbl = item.strip().split(' ')
            copyfile(os.path.join(data_dir, 'patches', rgb), os.path.join(patch_dir, rgb))
            copyfile(os.path.join(data_dir, 'patches', lbl), os.path.join(patch_dir, lbl))

        for item in valid_select:
            rgb, lbl = item.strip().split(' ')
            copyfile(os.path.join(data_dir, 'patches', rgb), os.path.join(patch_dir, rgb))
            copyfile(os.path.join(data_dir, 'patches', lbl), os.path.join(patch_dir, lbl))


def patches_to_hdf5(data_dir, save_dir, patch_size=None):
    """
    Convert a normal dataset into hdf5 format
    Reading files from hdf5 is faster due to faster decoding and none-scatter file reading
    :param data_dir: The data directory of a dataset created by preprocess.py
    :param save_dir: The directory to save the hdf5 dataset files
    :param patch_size: The size of the patches, every patch should have the same size, if None is given, the size of the
                       first element in file_list_train.txt will be used
    :return:
    """
    # create folders and files
    misc_utils.make_dir_if_not_exist(save_dir)
    hdf5_train = h5py.File(os.path.join(save_dir, 'train.hdf5'), mode='w')
    hdf5_valid = h5py.File(os.path.join(save_dir, 'valid.hdf5'), mode='w')

    # read train, valid files
    train_file_list = misc_utils.load_file(os.path.join(data_dir, 'file_list_train.txt'))
    valid_file_list = misc_utils.load_file(os.path.join(data_dir, 'file_list_valid.txt'))

    if not patch_size:
        # read first elemet in file_list_train.txt
        patch_size = misc_utils.load_file(os.path.join(data_dir, 'patches',
                                                       train_file_list[0].strip().split(' ')[0])).shape[:2]
    train_img_shape = (len(train_file_list), *patch_size, 3)
    train_lbl_shape = (len(train_file_list), *patch_size)
    valid_img_shape = (len(valid_file_list), *patch_size, 3)
    valid_lbl_shape = (len(valid_file_list), *patch_size)

    # create dataset
    hdf5_train.create_dataset('img', train_img_shape, np.uint8)
    hdf5_train.create_dataset('lbl', train_lbl_shape, np.uint8)
    hdf5_valid.create_dataset('img', valid_img_shape, np.uint8)
    hdf5_valid.create_dataset('lbl', valid_lbl_shape, np.uint8)

    # write data
    patch_dir = os.path.join(data_dir, 'patches')
    for cnt, line in enumerate(tqdm(train_file_list)):
        img_file, lbl_file = line.strip().split(' ')
        img, lbl = misc_utils.load_file(os.path.join(patch_dir, img_file)), \
                   misc_utils.load_file(os.path.join(patch_dir, lbl_file))
        np.testing.assert_array_equal(img.shape[:2], patch_size)
        np.testing.assert_array_equal(img.shape[:2], lbl.shape[:2])
        hdf5_train['img'][cnt, ...] = img
        hdf5_train['lbl'][cnt, ...] = lbl
    for cnt, line in enumerate(tqdm(valid_file_list)):
        img_file, lbl_file = line.strip().split(' ')
        img, lbl = misc_utils.load_file(os.path.join(patch_dir, img_file)), \
                   misc_utils.load_file(os.path.join(patch_dir, lbl_file))
        np.testing.assert_array_equal(img.shape[:2], patch_size)
        np.testing.assert_array_equal(img.shape[:2], lbl.shape[:2])
        hdf5_valid['img'][cnt, ...] = img
        hdf5_valid['lbl'][cnt, ...] = lbl

    hdf5_train.close()
    hdf5_valid.close()


if __name__ == '__main__':
    create_toy_set(r'/hdd/mrs/deepglobeland/ps512_pd0_ol0', move_dir='/hdd/mrs/deepglobeland/toyset',
                   n_train=0.1, n_valid=0.05)
    # patches_to_hdf5(r'/hdd/mrs/deepglobe/14p_pd0_ol0',
    #                 r'/hdd/mrs/deepglobe/14p_pd0_ol0_hdf5')
