"""

"""


# Built-in
import os

# Libs

# Pytorch
from torch import nn

# Own modules
import config
from network import unet, network_utils
from mrs_utils import misc_utils, metric_utils


def create_model(args):
    """
    Create the model based on configuration file
    :param args: the configuration parameter class defined in config.py
    :return:
    """
    args.encoder_name = misc_utils.stem_string(args.encoder_name)
    args.decoder_name = misc_utils.stem_string(args.decoder_name)
    if args.decoder_name == 'unet':
        if args.encoder_name == 'base':
            model = unet.UNet(sfn=args.sfn, n_class=args.num_classes, encoder_name=args.encoder_name)
            args.margin = model.lbl_margin
        elif args.encoder_name in ['vgg16', 'vgg']:
            model = unet.UNet(n_class=args.num_classes, encoder_name=args.encoder_name)
            args.margin = model.lbl_margin
        elif args.encoder_name in ['res50', 'resnet50']:
            model = unet.UNet(n_class=args.num_classes, encoder_name=args.encoder_name)
            args.margin = model.lbl_margin
        else:
            raise NotImplementedError('Encoder structure {} for {} is not supported'.format(
                args.encoder_name, args.decoder_name))
    else:
        raise NotImplementedError('Decoder structure {} is not supported'.format(args.decoder_name))
    return model


def create_loss(args):
    """
    Create loss based on configuration
    :param args: the configuration parameter class defined in config.py
    :return:
    """
    criterions = []
    for c_name in misc_utils.stem_string(args.criterion_name).split(','):
        if c_name == 'xent':
            criterions.append(metric_utils.CrossEntropyLoss())
        elif c_name == 'iou':
            criterions.append(metric_utils.IoU())
        else:
            raise NotImplementedError('Criterion type {} is not supported'.format(args.criterion_name))
    return criterions


def load_config(model_dir):
    """
    Load definition arguments in the config file, the dictionary will be load into the argument class defined in
    config.py
    :param model_dir: the directory to the model, this directory should be created by train.py and has a config.json
                      file
    :return: the parsed arguments
    """
    config_file = os.path.join(model_dir, 'config.json')
    args = config.Args()
    args.__dict__ = misc_utils.load_file(config_file)
    return args


def easy_load(model_dir, epoch=None):
    """
    Initialize and define model based on their corresponding configuration file
    :param model_dir: directory of the saved model
    :param epoch: number of epoch to load
    :return:
    """
    config = load_config(model_dir)
    model = create_model(config)
    if epoch:
        load_epoch = epoch
    else:
        load_epoch = config.epochs
    pretrained_dir = os.path.join(model_dir, 'epoch-{}.pth.tar'.format(load_epoch-1))
    network_utils.load(model, pretrained_dir)
    print('Loaded model from {} @ epoch {}'.format(model_dir, load_epoch))
    return model
