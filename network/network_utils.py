"""
This file defines commonly used functions for using networks
"""

# Built-in
import os
import copy
import timeit
from collections import OrderedDict


# Libs

# PyTorch
import torch
import torchvision
from torch import nn
from torchsummary import summary


# Own modules
from mrs_utils import misc_utils


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


def load_epoch(save_dir, resume_epoch, model, optm, device):
    """
    Load model from a snapshot, this function can be used to resume training
    :param save_dir: directory that saved the model
    :param resume_epoch: the epoch number to continue training
    :param model: the model created by classes defined in network/
    :param optm: a torch optimizer
    :return:
    """
    checkpoint = torch.load(
        os.path.join(save_dir, 'epoch-' + str(resume_epoch) + '.pth.tar'),
        map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'epoch-' + str(resume_epoch) + '.pth.tar')))
    model.load_state_dict(checkpoint['state_dict'])
    optm.load_state_dict(checkpoint['opt_dict'])
    # individually transfer the optimizer parts, this part comes from
    # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/4
    for state in optm.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


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
    elif disable_parallel or 'module' in [a for a in ckpt_params if a not in self_params][0]:
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
        try:
            pretrained_state = flex_load(model.state_dict(), checkpoint['state_dict'], relax_load, disable_parallel)
            model.load_state_dict(pretrained_state, strict=False)
        except ValueError:
            model.encoder = DataParallelPassThrough(model.encoder)
            model.decoder = DataParallelPassThrough(model.decoder)
            model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        # FIXME this is a adhoc fix to be compatible with RSMoCo
        pretrained_state = flex_load(model.state_dict(), checkpoint['model'], relax_load, disable_parallel)
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


def make_criterion_str(cfg):
    """
    Make a string for criterion used, the format will be [criterion a][weight]_[criterion b][weight]
    :param cfg: config dictionary
    :return:
    """
    criterion = cfg['trainer']['criterion_name'].split(',')
    if not isinstance(cfg['trainer']['bp_loss_idx'], list):
        bp_idx = [int(a) for a in eval(cfg['trainer']['bp_loss_idx'])]
    else:
        bp_idx = [int(a) for a in cfg['trainer']['bp_loss_idx']]
    bp_criterion = [criterion[a] for a in bp_idx]
    try:
        loss_weights = [misc_utils.float2str(float(a)) for a in eval(cfg['trainer']['loss_weights'])]
        return '_'.join('{}{}'.format(a, b) for (a, b) in zip(bp_criterion, loss_weights))
    except TypeError:
        assert len(bp_criterion) == 1
        return '_'.join('{}'.format(a) for a in bp_criterion)


def unique_model_name(cfg):
    """
    Make a unique model name based on the config file arguments
    :param cfg: config dictionary
    :return: unique model string
    """
    criterion_str = make_criterion_str(cfg)
    decay_str = '_'.join(str(ds) for ds in eval(cfg['optimizer']['decay_step']))
    dr_str = str(cfg['optimizer']['decay_rate']).replace('.', 'p')
    if cfg['optimizer']['aux_loss']:
        aux_str = '_aux{}'.format(misc_utils.float2str(cfg['optimizer']['aux_loss_weight']))
    else:
        aux_str = ''
    return 'ec{}_dc{}_ds{}_lre{:.0e}_lrd{:.0e}_ep{}_bs{}_ds{}_dr{}_cr{}{}'.format(
        cfg['encoder_name'], cfg['decoder_name'], cfg['dataset']['ds_name'], cfg['optimizer']['learn_rate_encoder'],
        cfg['optimizer']['learn_rate_decoder'], cfg['trainer']['epochs'], cfg['dataset']['batch_size'],
        decay_str, dr_str, criterion_str, aux_str)


class DataParallelPassThrough(torch.nn.DataParallel):
    """
    Access model attributes after DataParallel wrapper
    this code comes from: https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
