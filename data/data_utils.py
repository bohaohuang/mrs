"""

"""


# Built-in
import os

# Libs
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Own modules
from mrs_utils import misc_utils


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


if __name__ == '__main__':
    create_toy_set(r'/hdd/mrs/inria/ps512_pd0_ol0', move_dir='/hdd/mrs/inria/toyset')
