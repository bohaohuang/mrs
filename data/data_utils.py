"""

"""


# Built-in

# Libs
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Own modules


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
        return np.rollaxis(data, 0, 3)
    else:
        return np.rollaxis(data, 2, 0)


def visualize(rgb, gt, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.255)):
    """
    Visualize a given pair of image and mask normalized tensors
    :param rgb: the image tensor with shape [c, h, w]
    :param gt: the mask tensor with shape [1, h, w]
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
    gt = change_channel_order(gt, True)
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.imshow((rgb*255).astype(np.uint8))
    plt.subplot(122)
    plt.imshow(gt[:, :, 0].astype(np.uint8))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
