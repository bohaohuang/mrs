"""
This file defines commonly used functions for using networks
"""

# Built-in
import os
import copy
import timeit
from collections import OrderedDict


# Libs
import numpy as np

# PyTorch
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torchsummary import summary


# Own modules
from data import patch_extractor, data_utils
from mrs_utils import misc_utils, metric_utils


def write_and_print(writer, phase, current_epoch, total_epoch, loss_dict, s_time):
    """
    Write loss variables to the tensorboard as well as print log message
    :param writer: tensorboardX SummaryWriter
    :param phase: the current phase, will determine where the variables will be written in tensorboard
    :param current_epoch: current epoch number
    :param total_epoch: total number of epochs
    :param loss_dict: a dictionary with loss name and loss value pairs
    :param s_time: the time before this epoch begins, this is used to calculate duration
    :return:
    """
    loss_str = '[{}] Epoch: {}/{} '.format(phase, current_epoch, total_epoch)
    for loss_name, loss_value in loss_dict.items():
        if 'image' in loss_name:
            grid = torchvision.utils.make_grid(loss_value)
            writer.add_image('{}/{}_epoch'.format(loss_name, phase), grid, current_epoch)
        else:
            writer.add_scalar('data/{}_{}_epoch'.format(phase, loss_name), loss_value, current_epoch)
            loss_str += '{}: {:.3f} '.format(loss_name, loss_value)
    print(loss_str)
    stop_time = timeit.default_timer()
    print('Execution time: {}\n'.format(str(stop_time - s_time)))


def iterate_sublayers(network):
    """
    Iterate through all sublayers
    :param network: the network to iterate through
    :return: a list of layers
    """
    all_layers = []
    for layer in network.children():
        if isinstance(layer, nn.Sequential):
            all_layers.extend(iterate_sublayers(layer))
        if not list(layer.children()):
            all_layers.append(layer)
    return all_layers


def network_summary(network, input_size, **kwargs):
    """
    Make a summary of the network, could be used for debugging purpose
    :param network: network to be summarized
    :param input_size: a tuple of the input size
    :param kwargs: other parameters to initialize the model
    :return:
    """
    net = network(**kwargs)
    summary(net, input_size, device='cpu')


def load_epoch(save_dir, resume_epoch, model, optm):
    """
    Load model from a snapshot, this function can be used to resume training
    :param save_dir: directory that saved the model
    :param resume_epoch: the epoch number to continue training
    :param model: the model created by classes defined in network/
    :param optm: a torch optimizer
    :return:
    """
    checkpoint = torch.load(
        os.path.join(save_dir, 'epoch-' + str(resume_epoch - 1) + '.pth.tar'),
        map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'epoch-' + str(resume_epoch - 1) + '.pth.tar')))
    model.load_state_dict(checkpoint['state_dict'])
    optm.load_state_dict(checkpoint['opt_dict'])


def sequential_load(target, source_state):
    base_dict = OrderedDict(
        [layer for layer in list(target.items())
         if 'num_batches_tracked' not in layer[0]]
    )
    new_dict = OrderedDict(zip(base_dict.keys(), source_state.values()))
    return new_dict


def flex_load(model_dict, ckpt_dict, relax_load=False, disable_parallel=False, verb=True):
    # try to load model with relaxed naming restriction
    ckpt_params = [a for a in ckpt_dict.keys()]
    self_params = [a for a in model_dict.keys()]

    # only exists in ckpt
    model_params = [a for a in ckpt_params if a not in self_params]
    if verb:
        print('Warning: The following parameters in the pretrained model does not exist in the current model')
        for mp in model_params:
            print('\t', mp)

    # only exists in self
    model_params = [a for a in self_params if a not in ckpt_params]
    if verb:
        print('Warning: The following parameters in the current model does not exist in the pretrained model')
        for mp in model_params:
            print('\t', mp)

    # size not match
    model_params = [a for a in ckpt_params if a in self_params and model_dict[a].size() !=
                    ckpt_dict[a].size()]
    if verb:
        print('Warning: The size of the following parameters in the current model does not match the ones in the '
              'pretrained model')
        for mp in model_params:
            print('\t', mp)

    if not relax_load and not disable_parallel:
        pretrained_state = {k: v for k, v in ckpt_dict.items() if k in model_dict and
                            v.size() == model_dict[k].size()}
        if len(pretrained_state) == 0:
            raise ValueError('No parameter matches in the current model in pretrained model, please check '
                             'the model definition or enable relax_load')
        if verb:
            print('Try loading without those parameters')
        return pretrained_state
    elif disable_parallel:
        pretrained_state = {k: v for k, v in ckpt_dict.items() if k.replace('module.', '') in model_dict and
                            v.size() == model_dict[k.replace('module.', '')].size()}
        if len(pretrained_state) == 0:
            raise ValueError('No parameter matches in the current model in pretrained model, please check '
                             'the model definition or enable relax_load')
        if verb:
            print('Try loading without those parameters')
            print('{:.2f}% of the model loaded from the pretrained'.format(len(pretrained_state) / len(self_params) * 100))
        return pretrained_state
    else:
        if verb:
            print('Try loading with relaxed naming rule:')
        pretrained_state = {}

        # find one match string
        prefix = ''
        for self_name in self_params:
            if self_name in ckpt_params[0]:
                prefix = copy.deepcopy(ckpt_params[0]).replace(self_name, '')
                if verb:
                    print('Prefix in pretrained model {}'.format(prefix))
                break
            elif ckpt_params[0] in self_name:
                prefix = copy.deepcopy(self_name).replace(ckpt_params[0], '')
                if verb:
                    print('Prefix in current model {}'.format(prefix))
                break

        for self_name in self_params:
            ckpt_name = '{}{}'.format(prefix, self_name)
            if ckpt_name in ckpt_params:
                if verb:
                    print('\tpretrained param: {} -> current param: {}'.format(self_name, ckpt_name))
                if model_dict[self_name].size() == ckpt_dict[ckpt_name].size():
                    pretrained_state[self_name] = ckpt_dict[ckpt_name]
                else:
                    if verb:
                        print('\t\tIgnoring: {}->{} (size mismatch)'.format(ckpt_name, self_name))

        if verb:
            print('{:.2f}% of the model loaded from the pretrained'.format(len(pretrained_state) / len(self_params) * 100))
        return pretrained_state


def load(model, model_path, relax_load=False, disable_parallel=False):
    """
    Load the weights in the pretrained model directory, the order of loading method is as follows:
    1. Try load the exact name of tensors in the pretrained model file, if not all names are the same, try 2;
    2. Try only load the tensors with names and sizes match, if still no tensor pair found and relax_load, try 3;
    3. Assume the name of one model has a prefix compared with others, find the prefix first, then try load the model
    :param model: the current model that want to load the data weight
    :param model_path: the path to the pretrained model, should be a .pth or equivalent file
    :param relax_load: if true, the model will be load by assuming there's a prefix in one model's tensor names if
                       necessary
    :param disable_parallel: if true, the parameter name difference caused by using parallel model (module.) will be
                             ignored
    :return:
    """
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        pretrained_state = flex_load(model.state_dict(), checkpoint['state_dict'], relax_load, disable_parallel)
        model.load_state_dict(pretrained_state, strict=False)


def save(model, epochs, optm, loss_dict, save_name):
    """
    Save the model to given destination
    :param model: the model to be saved
    :param epochs: current epoch number
    :param optm: optimizer
    :param loss_dict: dictionary of current loss status
    :param save_name: absolute path to the file to store the model
    :return:
    """
    torch.save({
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'opt_dict': optm.state_dict(),
        'loss': loss_dict,
    }, save_name)
    print('Saved model at {}'.format(save_name))


def unique_model_name(cfg):
    """
    Make a unique model name based on the config file arguments
    :param cfg: config dictionary
    :return: unique model string
    """
    decay_str = '_'.join(str(ds) for ds in eval(cfg['optimizer']['decay_step']))
    dr_str = str(cfg['optimizer']['decay_rate']).replace('.', 'p')
    return 'ec{}_dc{}_ds{}_lre{:.0e}_lrd{:.0e}_ep{}_bs{}_ds{}_dr{}'.format(
        cfg['encoder_name'], cfg['decoder_name'], cfg['dataset']['ds_name'], cfg['optimizer']['learn_rate_encoder'],
        cfg['optimizer']['learn_rate_decoder'], cfg['trainer']['epochs'], cfg['dataset']['batch_size'],
        decay_str, dr_str)


class Evaluator:
    def __init__(self, ds_name, data_dir, tsfm, device, load_func=None, **kwargs):
        ds_name = misc_utils.stem_string(ds_name)
        self.tsfm = tsfm
        self.device = device
        if ds_name == 'inria':
            from data.inria import preprocess
            self.rgb_files, self.lbl_files = preprocess.get_images(data_dir, **kwargs)
            assert len(self.rgb_files) == len(self.lbl_files)
            self.truth_val = 255
        elif ds_name == 'deepglobe':
            from data.deepglobe import preprocess
            self.rgb_files, self.lbl_files = preprocess.get_images(data_dir)
            assert len(self.rgb_files) == len(self.lbl_files)
            self.truth_val = 1
        elif ds_name == 'deepgloberoad':
            from data.deepgloberoad import preprocess
            self.rgb_files, self.lbl_files = preprocess.get_images(data_dir, **kwargs)
            assert len(self.rgb_files) == len(self.lbl_files)
            self.truth_val = 255
        elif ds_name == 'mnih':
            from data.mnih import preprocess
            self.rgb_files, self.lbl_files = preprocess.get_images(data_dir, **kwargs)
            assert len(self.rgb_files) == len(self.lbl_files)
            self.truth_val = 255
        elif load_func:
            self.rgb_files, self.lbl_files = load_func(data_dir, **kwargs)
            assert len(self.rgb_files) == len(self.lbl_files)
            self.truth_val = 1
        else:
            raise NotImplementedError('Dataset {} is not supported')

    def evaluate(self, model, patch_size, overlap, pred_dir=None, report_dir=None, save_conf=False, delta=1e-6):
        iou_a, iou_b = 0, 0
        report = []
        if pred_dir:
            misc_utils.make_dir_if_not_exist(pred_dir)
        for rgb_file, lbl_file in zip(self.rgb_files, self.lbl_files):
            file_name = os.path.splitext(os.path.basename(lbl_file))[0]

            # read data
            rgb = misc_utils.load_file(rgb_file)[:, :, :3]
            lbl = misc_utils.load_file(lbl_file)
            # if label has multiple channels, only keep the first channel, this is not elegant but it is useful to deal
            # with deepglobe road
            # TODO make this an option when selecting dataset
            if len(lbl.shape) == 3:
                lbl = lbl[:, :, 0]

            # evaluate on tiles
            tile_dim = rgb.shape[:2]
            tile_dim_pad = [tile_dim[0]+2*model.lbl_margin, tile_dim[1]+2*model.lbl_margin]
            grid_list = patch_extractor.make_grid(tile_dim_pad, patch_size, overlap)
            tile_preds = []
            for patch in patch_extractor.patch_block(rgb, model.lbl_margin, grid_list, patch_size, False):
                for tsfm in self.tsfm:
                    tsfm_image = tsfm(image=patch)
                    patch = tsfm_image['image']
                patch = torch.unsqueeze(patch, 0).to(self.device)
                pred = F.softmax(model.forward(patch), 1).detach().cpu().numpy()
                tile_preds.append(data_utils.change_channel_order(pred, True)[0, :, :, :])
            # stitch back to tiles
            tile_preds = patch_extractor.unpatch_block(
                np.array(tile_preds),
                tile_dim_pad,
                patch_size,
                tile_dim,
                [patch_size[0]-2*model.lbl_margin, patch_size[1]-2*model.lbl_margin],
                overlap=2*model.lbl_margin
            )
            if save_conf:
                misc_utils.save_file(os.path.join(pred_dir, '{}.npy'.format(file_name)), tile_preds[:, :, 1])
            tile_preds = np.argmax(tile_preds, -1)
            a, b = metric_utils.iou_metric(lbl/self.truth_val, tile_preds)
            print('{}: IoU={:.2f}'.format(file_name, a/(b+delta)*100))
            report.append('{},{},{},{}\n'.format(file_name, a, b, a/(b+delta)*100))
            iou_a += a
            iou_b += b
            if pred_dir:
                misc_utils.save_file(os.path.join(pred_dir, '{}.png'.format(file_name)), tile_preds*self.truth_val)
        print('Overall: IoU={:.2f}'.format(iou_a/iou_b*100))
        report.append('Overall,{},{},{}\n'.format(iou_a, iou_b, iou_a/iou_b*100))
        if report_dir:
            misc_utils.make_dir_if_not_exist(report_dir)
            misc_utils.save_file(os.path.join(report_dir, 'result.txt'), report)
