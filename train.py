"""

"""


# Built-in
import os
import sys
import json
import timeit
import argparse

# Libs
from tensorboardX import SummaryWriter

# Pytorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Own modules
from data import data_loader
from mrs_utils import misc_utils
from network import network_utils, network_io

CONFIG_FILE = 'temp_config.json'


def read_config():
    parser = argparse.ArgumentParser()
    args, extras = parser.parse_known_args(sys.argv[1:])
    cfg_dict = misc_utils.parse_args(extras)
    if 'config' not in cfg_dict:
        cfg_dict['config'] = CONFIG_FILE
    flags = json.load(open(cfg_dict['config']))
    flags = misc_utils.update_flags(flags, cfg_dict)
    flags['save_dir'] = os.path.join(flags['trainer']['save_root'], network_utils.unique_model_name(flags))
    return flags


def train_model(args, device, parallel):
    """
    The function to train the model
    :param args: the class carries configuration parameters defined in config.py
    :param device: the device to run the model
    :return:
    """
    
    model = network_io.create_model(args)
    log_dir = os.path.join(args['save_dir'], 'log')
    writer = SummaryWriter(log_dir=log_dir)
    try:
        writer.add_graph(model, torch.rand(1, 3, *eval(args['dataset']['input_size'])))
    except (RuntimeError, TypeError, AttributeError):
        print('Warning: could not write graph to tensorboard, this might be a bug in tensorboardX')
    if parallel:
        model.encoder = network_utils.DataParallelPassThrough(model.encoder)
        model.decoder = network_utils.DataParallelPassThrough(model.decoder)
        if args['optimizer']['aux_loss']:
            model.cls = network_utils.DataParallelPassThrough(model.cls)
        print('Parallel training mode enabled!')
    train_params = model.set_train_params((args['optimizer']['learn_rate_encoder'],
                                           args['optimizer']['learn_rate_decoder']))

    # make optimizer
    optm = network_io.create_optimizer(args['optimizer']['name'], train_params, args['optimizer']['learn_rate_encoder'])
    criterions = network_io.create_loss(args, device=device)
    if args['optimizer']['aux_loss']:
        from mrs_utils import metric_utils
        cls_criterion = metric_utils.BCEWithLogitLoss(device, eval(args['trainer']['class_weight']))
    scheduler = optim.lr_scheduler.MultiStepLR(optm, milestones=eval(args['optimizer']['decay_step']),
                                               gamma=args['optimizer']['decay_rate'])

    # if not resume, train from scratch
    if args['trainer']['resume_epoch'] == 0 and args['trainer']['finetune_dir'] == 'None':
        print('Training decoder {} with encoder {} from scratch ...'.format(args['decoder_name'], args['encoder_name']))
    elif args['trainer']['resume_epoch'] == 0 and args['trainer']['finetune_dir']:
        print('Finetuning model from {}'.format(args['trainer']['finetune_dir']))
        if args['trainer']['further_train']:
            network_utils.load(model, args['trainer']['finetune_dir'], relax_load=True, optm=optm, device=device)
        else:
            network_utils.load(model, args['trainer']['finetune_dir'], relax_load=True)
    else:
        print('Resume training decoder {} with encoder {} from epoch {} ...'.format(
            args['decoder_name'], args['encoder_name'], args['trainer']['resume_epoch']))
        network_utils.load_epoch(args['save_dir'], args['trainer']['resume_epoch'], model, optm, device)

    # prepare training
    print('Total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    for c in criterions:
        c.to(device)

    # make data loader
    mean, std = network_io.get_dataset_stats(args['dataset']['ds_name'], args['dataset']['data_dir'],
                                             mean_val=(eval(args['dataset']['mean']), eval(args['dataset']['std'])))
    tsfm_train, tsfm_valid = network_io.create_tsfm(args, mean, std)
    train_loader = DataLoader(data_loader.get_loader(
        args['dataset']['data_dir'], args['dataset']['train_file'], transforms=tsfm_train,
        aux_loss=args['optimizer']['aux_loss']),
        batch_size=int(args['dataset']['batch_size']), shuffle=True, num_workers=int(args['dataset']['num_workers']),
        drop_last=True)
    valid_loader = DataLoader(data_loader.get_loader(
        args['dataset']['data_dir'], args['dataset']['valid_file'], transforms=tsfm_valid,
        aux_loss=args['optimizer']['aux_loss']),
        batch_size=int(args['dataset']['batch_size']), shuffle=False, num_workers=int(args['dataset']['num_workers']))
    print('Training model on the {} dataset'.format(args['dataset']['ds_name']))
    train_val_loaders = {'train': train_loader, 'valid': valid_loader}

    # train the model
    loss_dict = {}
    for epoch in range(int(args['trainer']['resume_epoch']), int(args['trainer']['epochs'])):
        # each epoch has a training and validation step
        for phase in ['train', 'valid']:
            start_time = timeit.default_timer()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # TODO align aux loss and normal train
            if args['optimizer']['aux_loss']:
                loss_dict = model.step_aux(train_val_loaders[phase], device, optm, phase, criterions, cls_criterion,
                                           args['optimizer']['aux_loss_weight'], eval(args['trainer']['bp_loss_idx']),
                                           True, mean, std, loss_weights=eval(args['trainer']['loss_weights']),
                                           use_emau=args['use_emau'])
            else:
                loss_dict = model.step(train_val_loaders[phase], device, optm, phase, criterions,
                                       eval(args['trainer']['bp_loss_idx']), True, mean, std,
                                       loss_weights=eval(args['trainer']['loss_weights']), use_emau=args['use_emau'])
            network_utils.write_and_print(writer, phase, epoch, int(args['trainer']['epochs']), loss_dict, start_time)

        scheduler.step()
        # save the model
        if epoch % int(args['trainer']['save_epoch']) == 0 and epoch != 0:
            save_name = os.path.join(args['save_dir'], 'epoch-{}.pth.tar'.format(epoch))
            network_utils.save(model, epoch, optm, loss_dict, save_name)
    # save model one last time
    save_name = os.path.join(args['save_dir'], 'epoch-{}.pth.tar'.format(int(args['trainer']['epochs'])))
    network_utils.save(model, int(args['trainer']['epochs']), optm, loss_dict, save_name)
    writer.close()


def main():
    # settings
    cfg = read_config()
    # set gpu to use
    device, parallel = misc_utils.set_gpu(cfg['gpu'])
    # set random seed
    misc_utils.set_random_seed(cfg['random_seed'])
    # make training directory
    misc_utils.make_dir_if_not_exist(cfg['save_dir'])
    misc_utils.save_file(os.path.join(cfg['save_dir'], 'config.json'), cfg)

    # train the model
    train_model(cfg, device, parallel)


if __name__ == '__main__':
    main()
