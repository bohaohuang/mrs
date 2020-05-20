"""

"""


# Built-in
import os

# Libs
import numpy as np

# Pytorch
import albumentations as A
from torch import optim
from albumentations.pytorch import ToTensorV2

# Own modules
from mrs_utils import misc_utils, metric_utils, process_block
from network import unet, pspnet, dlinknet, deeplabv3, network_utils


def create_model(args):
    """
    Create the model based on configuration file
    :param args: the configuration parameter class defined in config.py
    :return:
    """
    args['encoder_name'] = misc_utils.stem_string(args['encoder_name'])
    args['decoder_name'] = misc_utils.stem_string(args['decoder_name'])
    if args['optimizer']['aux_loss']:
        aux_loss = True
    else:
        aux_loss = False

    # TODO this is for compatible issue only, we might want to get rid of this later
    if 'imagenet' not in args:
        args['imagenet'] = 'True'

    if args['decoder_name'] == 'unet':
        if args['encoder_name'] == 'base':
            model = unet.UNet(sfn=args['sfn'], n_class=args['dataset']['class_num'],
                              encoder_name=args['encoder_name'], aux_loss=aux_loss, use_emau=args['use_emau'],
                              use_ocr=args['use_ocr'])
        else:
            model = unet.UNet(n_class=args['dataset']['class_num'], encoder_name=args['encoder_name'],
                              pretrained=eval(args['imagenet']), aux_loss=aux_loss, use_emau=args['use_emau'],
                              use_ocr=args['use_ocr'])
    elif args['decoder_name'] in ['psp', 'pspnet']:
        model = pspnet.PSPNet(n_class=args['dataset']['class_num'], encoder_name=args['encoder_name'],
                              pretrained=eval(args['imagenet']), aux_loss=aux_loss, use_emau=args['use_emau'],
                              use_ocr=args['use_ocr'])
    elif args['decoder_name'] == 'dlinknet':
        model = dlinknet.DLinkNet(n_class=args['dataset']['class_num'], encoder_name=args['encoder_name'],
                                  pretrained=eval(args['imagenet']), aux_loss=aux_loss, use_emau=args['use_emau'],
                                  use_ocr=args['use_ocr'])
    elif args['decoder_name'] == 'deeplabv3':
        model = deeplabv3.DeepLabV3(n_class=args['dataset']['class_num'], encoder_name=args['encoder_name'],
                                    pretrained=eval(args['imagenet']), aux_loss=aux_loss, use_emau=args['use_emau'],
                                    use_ocr=args['use_ocr'])
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
            criterions.append(metric_utils.CrossEntropyLoss(eval(args['trainer']['class_weight'])))
        elif c_name == 'iou':
            # this metric is non-differentiable
            criterions.append(metric_utils.IoU())
        elif c_name == 'softiou':
            criterions.append(metric_utils.SoftIoULoss(kwargs['device']))
        elif c_name == 'focal':
            criterions.append(metric_utils.FocalLoss(kwargs['device'], gamma=args['trainer']['gamma'],
                                                     alpha=args['trainer']['alpha']))
        elif c_name == 'lovasz':
            criterions.append(metric_utils.LovaszSoftmax())
        else:
            raise NotImplementedError('Criterion type {} is not supported'.format(args['trainer']['criterion_name']))
    return criterions


def create_optimizer(optm_name, train_params, lr):
    """
    Create optimizer based on configuration
    :param optm_name: the optimizer name defined in config.py
    :param train_params: learning rate arrangement for the training parameters
    :param lr: learning rate
    :return: corresponding torch optim class
    """
    o_name = misc_utils.stem_string(optm_name)
    if o_name == 'sgd':
        optm = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif o_name == 'adam':
        optm = optim.Adam(train_params, lr=lr)
    else:
        raise NotImplementedError('Optimizer name {} is not supported'.format(optm_name))
    return optm


def create_tsfm(args, mean, std, normalize=True):
    """
    Create transform based on configuration
    :param args: the argument parameters defined in config.py
    :param mean: mean of the dataset
    :param std: std of the dataset
    :param normalize: if True, will normalize the dataset
    :return: corresponding train and validation transforms
    """
    input_size = eval(args['dataset']['input_size'])
    crop_size = eval(args['dataset']['crop_size'])
    if normalize:
        tsfms = [A.Flip(), A.RandomRotate90(), A.Normalize(mean=mean, std=std), ToTensorV2()]
    else:
        tsfms = [A.Flip(), A.RandomRotate90(), ToTensorV2()]
    if input_size[0] > crop_size[0] and input_size[1] > crop_size[1]:
        tsfm_train = A.Compose([A.RandomCrop(*crop_size)] + tsfms)
        tsfm_valid = A.Compose([A.RandomCrop(*crop_size)] + tsfms[2:])
    elif input_size[0] < crop_size[0] or input_size[1] < crop_size[1]:
        tsfm_train = A.Compose([A.RandomResizedCrop(*crop_size)] + tsfms)
        tsfm_valid = A.Compose([A.RandomResizedCrop(*crop_size)] + tsfms[2:])
    else:
        tsfm_train = A.Compose(tsfms)
        tsfm_valid = A.Compose(tsfms[-2:])
    return tsfm_train, tsfm_valid


def get_dataset_stats(ds_name, img_dir, load_func=None, file_list=None,
                      mean_val=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    if ds_name == 'inria':
        from data.inria import preprocess
        val = preprocess.get_stats_pb(img_dir)[0]
        print('Use {} mean std stats: {}'.format(ds_name, val))
    elif ds_name == 'deepglobe':
        from data.deepglobe import preprocess
        val = preprocess.get_stats_pb(img_dir)[0]
        print('Use {} mean std stats: {}'.format(ds_name, val))
    elif ds_name == 'deepgloberoad':
        from data.deepgloberoad import preprocess
        val = preprocess.get_stats_pb(img_dir)[0]
        print('Use {} mean std stats: {}'.format(ds_name, val))
    elif ds_name == 'deepglobeland':
        from data.deepglobeland import preprocess
        val = preprocess.get_stats_pb(img_dir)[0]
        print('Use {} mean std stats: {}'.format(ds_name, val))
    elif ds_name == 'mnih':
        from data.mnih import preprocess
        val = preprocess.get_stats_pb(img_dir)[0]
        print('Use {} mean std stats: {}'.format(ds_name, val))
    elif ds_name == 'spca':
        from data.spca import preprocess
        val = preprocess.get_stats_pb(img_dir)[0]
        print('Use {} mean std stats: {}'.format(ds_name, val))
    elif load_func:
        try:
            val = process_block.ValueComputeProcess(
                ds_name, os.path.join(os.path.dirname(__file__), '../data/stats/custom'),
                os.path.join(os.path.dirname(__file__), '../data/stats/custom/{}.npy'.format(ds_name)),
                func=load_func). \
                run(img_dir=img_dir, file_list=file_list).val
            print('Use {} mean std stats: {}'.format(ds_name, val))
        except ValueError:
            print('Dataset {} is not supported, use default mean stats instead'.format(ds_name))
            return np.array(mean_val)
    else:
        print('Dataset {} is not supported, use default mean stats instead'.format(ds_name))
        return np.array(mean_val)
    return val[0, :], val[1, :]


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
    return misc_utils.historical_process_flag(args)


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
