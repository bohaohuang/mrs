# Built-in
import os
import time
import json
import pickle
from functools import wraps

# Libs
import torch
import numpy as np
from PIL import Image
from skimage import io
from torchsummary import summary

# Own modules


def set_gpu(gpu):
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:{}'.format(gpu))
    return device


def make_dir_if_not_exist(dir_path):
    """
    Make the directory if it does not exists
    :param dir_path: absolute path to the directory
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def timer_decorator(func):
    """
    This is a decorator to print out running time of executing func
    :param func:
    :return:
    """
    @wraps(func)
    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        duration = time.time() - start_time
        print('duration: {:.3f}s'.format(duration))
    return timer_wrapper


def str2list(s, sep=',', d_type=int):
    """
    Change a {sep} separated string into a list of items with d_type
    :param s: input string
    :param sep: separator for string
    :param d_type: data type of each element
    :return:
    """
    if type(s) is not list:
        s = [d_type(a) for a in s.split(sep)]
    return s


def load_file(file_name, **kwargs):
    """
    Read data file of given path, use numpy.load if it is in .npy format,
    otherwise use pickle or imageio
    :param file_name: absolute path to the file
    :return: file data, or IOError if it cannot be read by either numpy or pickle or imageio
    """
    try:
        if file_name[-3:] == 'npy':
            data = np.load(file_name)
        elif file_name[-3:] == 'pkl' or file_name[-6:] == 'pickle':
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
        elif file_name[-3:] == 'txt':
            with open(file_name, 'r') as f:
                data = f.readlines()
        elif file_name[-3:] == 'csv':
            data = np.genfromtxt(file_name, delimiter=',', dtype=None, encoding='UTF-8')
        elif file_name[-4:] == 'json':
            data = json.load(open(file_name))
        elif 'pil' in kwargs and kwargs['pil']:
            data = Image.open(file_name)
        else:
            data = io.imread(file_name)

        return data
    except Exception:  # so many things could go wrong, can't be more specific.
        raise IOError('Problem loading {}'.format(file_name))


def save_file(file_name, data, **kwargs):
    """
    Save data file of given path, use numpy.load if it is in .npy format,
    otherwise use pickle or imageio
    :param file_name: absolute path to the file
    :param data: data to save
    :return: file data, or IOError if it cannot be saved by either numpy or or pickle imageio
    """
    try:
        if file_name[-3:] == 'npy':
            np.save(file_name, data)
        elif file_name[-3:] == 'pkl':
            with open(file_name, 'wb') as f:
                pickle.dump(data, f)
        elif file_name[-3:] == 'txt':
            with open(file_name, 'w') as f:
                f.writelines(data)
        elif file_name[-3:] == 'csv':
            np.savetxt(file_name, data, delimiter=',', fmt=kwargs['fmt'])
        elif file_name[-4:] == 'json':
            json.dump(data, open(file_name, 'w'))
        else:
            data = Image.fromarray(data.astype(np.uint8))
            data.save(file_name)
    except Exception:  # so many things could go wrong, can't be more specific.
        raise IOError('Problem saving this data')


def get_img_channel_num(file_name):
    """
    Get #channels of the image file
    :param file_name: absolute path to the image file
    :return: #channels or ValueError
    """
    img = load_file(file_name)
    if len(img.shape) == 2:
        channel_num = 1
    elif len(img.shape) == 3:
        channel_num = img.shape[-1]
    else:
        raise ValueError('Image can only have 2 or 3 dimensions')
    return channel_num


def rotate_list(l):
    """
    Rotate a list of lists
    :param l: list of lists to rotate
    :return:
    """
    return list(map(list, zip(*l)))


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


def make_center_string(char, length, center_str=''):
    """
    Make one line decoration string that has center_str at the center and surrounded by char
    The total length of the string equals to length
    :param char: character to be repeated
    :param length: total length of the string
    :param center_str: string that shown at the center
    :return:
    """
    return center_str.center(length, char)


def float2str(f):
    """
    Return a string for float number and change '.' to character 'p'
    :param f: float number
    :return: changed string
    """
    return '{}'.format(f).replace('.', 'p')


def stem_string(s, lower=True):
    """
    Return a string that with spaces at the begining or end removed and all casted to lower cases
    :param s: input string
    :param lower: if True, the string will be casted to lower cases
    :return: stemmed string
    """
    if lower:
        return s.strip().lower()
    else:
        return s.strip()


def remove_digits(s):
    """
    Remove digits in the given string
    :param s: input string
    :return: digits removed string
    """
    return ''.join([c for c in s if not c.isdigit()])


def get_digits(s):
    """
    Get digits in the given string, cast to int
    :param s: input string
    :return: int from string
    """
    return int(''.join([c for c in s if c.isdigit()]))


def get_model_summary(model, shape, device=None):
    """
    Get model summary with torchsummary
    :param model: the model to visualize summary
    :param shape: shape of the input data
    :return:
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary(model.to(device), shape)
