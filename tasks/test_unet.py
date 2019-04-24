"""

"""


# Built-in
import os
import argparse
from glob import glob

# Libs
import torch
import torchvision
from natsort import natsorted

# Own modules
from model import unet
from mrs_utils import misc_utils
from mrs_utils import xval_utils
from mrs_utils import metric_utils


DATA_DIR = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
GT_DATA_DIR = r'/media/ei-edl01/data/uab_datasets/inria/data/TilePreproc/MultChanOp_chans3_Divide_dF255p000'
GPU = 0
PRETRAINED_DIR = r'/home/lab/Documents/bohao/code/mrs/model/log_pre/res101_unet_lre1E-05_lrd1E-05_ep40_ms40/model_20.pt'
ENCODER_NAME = 'res101'
DECODER_NAME = 'unet'
N_CLASS = 2
PREDIR = None
INPUT_SIZE = 224
BATCH_SIZE = 10
PAD = 0


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='path to the rgb files')
    parser.add_argument('--gt-data-dir', type=str, default=GT_DATA_DIR, help='path to the gt files')
    parser.add_argument('--gpu', type=int, default=GPU, help='which gpu to use')
    parser.add_argument('--pretrained-dir', type=str, default=PRETRAINED_DIR, help='path to the pretrained model')
    parser.add_argument('--encoder-name', type=str, default=ENCODER_NAME, help='which encoder to use for extractor, '
                                                                               'see model/model.py for more details')
    parser.add_argument('--decoder-name', type=str, default=DECODER_NAME, help='which decoder style to use')
    parser.add_argument('--n-class', type=int, default=N_CLASS, help='#classes in the output')
    parser.add_argument('--predir', type=str, default=PREDIR, help='path to pretrained encoder')
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE, help='input size of the patches')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size in training')
    parser.add_argument('--pad', type=int, default=PAD, help='padding around the tile')

    '''
    parser.add_argument('--init-lr-encoder', type=float, default=INIT_LR_ENCODER, help='initial learning rate for encoder')
    parser.add_argument('--init-lr-decoder', type=float, default=INIT_LR_DECODER, help='initial learning rate for decoder')
    parser.add_argument('--milestones', type=str, default=MILESTONES, help='milestones for multi step lr drop')
    parser.add_argument('--drop-rate', type=float, default=DROP_RATE, help='drop rate at each milestone in scheduler')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='num of epochs to train')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR, help='path to save the model')
    parser.add_argument('--save-epoch', type=int, default=SAVE_EPOCH, help='model will be saved every #epochs')
    '''

    flags = parser.parse_args()
    '''home_dir = os.path.join(flags.save_dir, get_unique_name(flags.encoder_name, flags.decoder_name,
                                                            flags.init_lr_encoder, flags.init_lr_decoder,
                                                            flags.epochs, flags.milestones))
    flags.log_dir = os.path.join(home_dir, 'log')
    flags.save_dir = os.path.join(home_dir, 'model.pt')
    flags.milestones = misc_utils.str2list(flags.milestones, sep='_')'''
    return flags


def get_file_list(data_dir, gt_data_dir):
    file_list = []
    gt_files = natsorted(glob(os.path.join(gt_data_dir, '*GT_Divide.tif')))
    for gt_file in gt_files:
        rgb_file = os.path.join(data_dir, '{}_RGB.tif'.format(os.path.basename(gt_file).split('_')[0]))
        file_list.append([rgb_file, gt_file])
    return file_list


def data_loader(flags):
    return get_file_list(flags.data_dir, flags.gt_data_dir)


def main(flags):
    # get files to be evaluated
    file_list = data_loader(flags)
    city_id_list = xval_utils.get_inria_city_id(file_list)
    file_list, _ = xval_utils.split_by_id(file_list, city_id_list, list(range(6)))

    # get pretrained model
    device = misc_utils.set_gpu(flags.gpu)
    model = unet.Unet(flags.encoder_name, flags.n_class, flags.predir).to(device)
    model.load_state_dict(torch.load(flags.pretrained_dir))

    # evalutate on each tile
    def city_name_id(file_name):
        city_with_id = os.path.basename(file_name).split('_')[0]
        city_name = misc_utils.remove_digits(city_with_id)
        city_id = misc_utils.get_digits(city_with_id)
        return city_name, city_id

    transforms = torchvision.transforms.Compose({
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    })

    save_dir = os.path.join(os.path.dirname(flags.pretrained_dir), 'results')
    results, summary = metric_utils.eval_on_dataset(file_list, (flags.input_size, flags.input_size), flags.batch_size,
                                                    flags.pad, transforms, device, model, city_name_id, save_dir, True)
    print(summary)


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
