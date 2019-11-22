"""

"""


# Built-in
import os

# Libs

# Pytorch

# Own modules
from mrs_utils import misc_utils, metric_utils
from network import unet, pspnet, dlinknet, deeplabv3, network_utils


def create_model(args):
    """
    Create the model based on configuration file
    :param args: the configuration parameter class defined in config.py
    :return:
    """
    args['encoder_name'] = misc_utils.stem_string(args['encoder_name'])
    args['decoder_name'] = misc_utils.stem_string(args['decoder_name'])
    if args['decoder_name'] == 'unet':
        if args['encoder_name'] == 'base':
            model = unet.UNet(sfn=args['sfn'], n_class=args['dataset']['class_num'],
                              encoder_name=args['encoder_name'])
        else:
            model = unet.UNet(n_class=args['dataset']['class_num'], encoder_name=args['encoder_name'],
                              pretrained=eval(args['imagenet']))
    elif args['decoder_name'] in ['psp', 'pspnet']:
        model = pspnet.PSPNet(n_class=args['dataset']['class_num'], encoder_name=args['encoder_name'],
                              pretrained=eval(args['imagenet']))
    elif args['decoder_name'] == 'dlinknet':
        model = dlinknet.DLinkNet(n_class=args['dataset']['class_num'], encoder_name=args['encoder_name'],
                                  pretrained=eval(args['imagenet']))
    elif args['decoder_name'] == 'deeplabv3':
        model = deeplabv3.DeepLabV3(n_class=args['dataset']['class_num'], encoder_name=args['encoder_name'],
                                    pretrained=eval(args['imagenet']))
    else:
        raise NotImplementedError('Decoder structure {} is not supported'.format(args['decoder_name']))
    return model


def create_loss(args, **kwargs):
    """
    Create loss based on configuration
    :param args: the configuration parameter class defined in config.py
    :return:
    """
    criterions = []
    for c_name in misc_utils.stem_string(args['trainer']['criterion_name']).split(','):
        if c_name == 'xent':
            if 'class_weight' in kwargs:
                class_weight = eval(kwargs['class_weight'])
            else:
                class_weight = (1, 1)
            criterions.append(metric_utils.CrossEntropyLoss(class_weight))
        elif c_name == 'iou':
            criterions.append(metric_utils.IoU())
        elif c_name == 'softiou':
            criterions.append(metric_utils.SoftIoULoss(**kwargs))
        else:
            raise NotImplementedError('Criterion type {} is not supported'.format(args['trainer']['criterion_name']))
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
    args = misc_utils.load_file(config_file)
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
