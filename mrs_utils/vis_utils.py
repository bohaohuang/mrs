# Built-in


# Libs
import numpy as np
import matplotlib.pyplot as plt


# Own modules
from data import data_utils


def get_default_colors():
    """
    Get plt default colors
    :return: a list of rgb colors
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors


def get_color_list():
    """
    Get default color list in plt, convert hex value to rgb tuple
    :return:
    """
    colors = get_default_colors()
    return [tuple(int(a.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for a in colors]


def decode_label_map(label, label_num=2, label_colors=None):
    """
    #TODO this could be more efficient
    Decode label prediction map into rgb color map
    :param label: label prediction map
    :param label_num: #distinct classes in ground truth
    :param label_colors: list of tuples with RGB value of label colormap
    :return:
    """
    if len(label.shape) == 3:
        label = np.expand_dims(label, -1)
    n, h, w, c = label.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    if not label_colors:
        color_list = get_color_list()
        label_colors = {}
        for i in range(label_num):
            label_colors[i] = color_list[i]
        label_colors[0] = (255, 255, 255)
    for i in range(n):
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                pixels[j, k] = label_colors[np.int(label[i, j, k, 0])]
        outputs[i] = pixels
    return outputs


def inv_normalize(img, mean, std):
    """
    Do inverse normalize for images
    :param img: the image to be normalized
    :param mean: the original mean
    :param std: the original std
    :return:
    """
    inv_mean = [-a / b for a, b in zip(mean, std)]
    inv_std = [1 / a for a in std]
    if len(img.shape) == 3:
        return (img - inv_mean) / inv_std
    elif len(img.shape) == 4:
        for i in range(img.shape[0]):
            img[i, :, :, :] = (img[i, :, :, :] - inv_mean) / inv_std
        return img


def make_tb_image(img, lbl, pred, n_class, mean, std, chanel_first=True):
    """
    Make validation image for tensorboard
    :param img: the image to display, has shape N * 3 * H * W
    :param lbl: the label to display, has shape N * C * H * W
    :param pred: the pred to display has shape N * C * H * W
    :param n_class: the number of classes
    :param mean: mean used in normalization
    :param std: std used in normalization
    :param chanel_first: if True, the inputs are in channel first format
    :return:
    """
    pred = np.argmax(pred, 1)
    label_image = decode_label_map(lbl, n_class)
    pred_image = decode_label_map(pred, n_class)
    img_image = inv_normalize(data_utils.change_channel_order(img), mean, std) * 255
    banner = np.concatenate([img_image, label_image, pred_image], axis=2).astype(np.uint8)
    if chanel_first:
        banner = data_utils.change_channel_order(banner, False)
    return banner


def make_cmp_mask(lbl, pred, tp_mask_color=(0, 255, 0), fp_mask_color=(255, 0, 0), fn_mask_color=(0, 0, 255)):
    """
    Make compare mask for visualization purpose, the label and prediction maps should be binary and the truth value can
    only be 1
    :param lbl: the label map with dimension height * width
    :param pred: the prediction map with dimension height * width
    :param tp_mask_color: the rgb color of TP pixels, green by default
    :param fp_mask_color: the rgb color of FP pixels, red by default
    :param fn_mask_color: the rgb color of FN pixels, blue by default
    :return:
    """
    assert lbl.shape == pred.shape
    if np.max(lbl) != 1:
        lbl = lbl / np.max(lbl)
    if np.max(pred) != 1:
        pred = pred / np.max(pred)
    cmp_mask = 255 * np.ones((*lbl.shape, 3), dtype=np.uint8)
    tp_mask = (lbl == 1) * (lbl == pred)
    fp_mask = (pred - lbl) == 1
    fn_mask = (lbl - pred) == 1
    cmp_mask[tp_mask, :] = tp_mask_color
    cmp_mask[fp_mask, :] = fp_mask_color
    cmp_mask[fn_mask, :] = fn_mask_color
    return cmp_mask


def compare_figures(images, nrows_ncols, show_axis=False, fig_size=(10, 8), show_fig=True,
                    title_list=None):
    """
    Show three figures in a row, link their axes
    :param img_1: image to show on top left
    :param img_2: image to show at top right
    :param img_3: image to show on bottom left
    :param img_4: image to show on bottom right
    :param show_axis: if False, axes will be hide
    :param fig_size: size of the figure
    :param show_fig: show figure or not
    :param color_bar: if True, add color bar to the last plot
    :return:
    """
    from mpl_toolkits.axes_grid1 import Grid
    if title_list:
        assert len(title_list) == len(images)
    fig = plt.figure(figsize=fig_size)
    grid = Grid(fig, rect=111, nrows_ncols=nrows_ncols, axes_pad=0.25, label_mode='L', share_all=True)
    for i, (ax, img) in enumerate(zip(grid, images)):
        ax.imshow(img)
        if not show_axis:
            ax.axis('off')
        if title_list:
            ax.set_title(title_list[i])
    plt.tight_layout()
    if show_fig:
        plt.show()
