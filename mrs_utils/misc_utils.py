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


def set_gpu(gpu, enable_benchmark=True):
    """
    Set which gpu to use, also return True as indicator for parallel model if multi-gpu selected
    :param gpu: which gpu to use, could a a string with device ids separated by ','
    :param enable_benchmark: if True, will let CUDNN find optimal set of algorithms for input configuration
    :return: device instance
    """
    if len(str(gpu)) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        parallel = True
        device = torch.device("cuda:{}".format(','.join([str(a) for a in range(len(gpu.split(',')))])))
        print("Devices being used:", device)
    else:
        parallel = False
        device = torch.device("cuda:{}".format(gpu))
        print("Device being used:", device)
    torch.backends.cudnn.benchmark = enable_benchmark
    return device, parallel


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


def save_file(file_name, data, fmt='%.8e', sort_keys=True, indent=4):
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
            np.savetxt(file_name, data, delimiter=',', fmt=fmt)
        elif file_name[-4:] == 'json':
            json.dump(data, open(file_name, 'w'), sort_keys=sort_keys, indent=indent)
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


def set_random_seed(seed_):
    """
    Set random seed for torch, cudnn and numpy
    :param seed_: random seed to use, could be your lucky number :)
    :return:
    """
    torch.manual_seed(seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_)


def args_getter(inspect_class):
    """
    Inspect parameters inside a class
    :param inspect_class: the class to be inspected
    :return: a dict of key value pairs of all parameters in this class
    """
    params = {}
    for k, v in inspect_class.__dict__.items():
        if not k.startswith('__'):
            params[k] = v
    return params


def args_writer(file_name, inspect_class):
    """
    Save parameters inside a class into json file
    :param file_name: path to the file to be saved
    :param inspect_class: the class which parameters are going to be saved
    :return:
    """
    params = args_getter(inspect_class)
    save_file(file_name, params, sort_keys=True, indent=4)
