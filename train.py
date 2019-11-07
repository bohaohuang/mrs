"""

"""


# Built-in
import os
import json
import shutil
import timeit
import argparse

# Libs
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter

# Pytorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Own modules
from data import data_loader
from mrs_utils import misc_utils
from network import network_utils, network_io

CONFIG_FILE = 'config.json'


def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=CONFIG_FILE, type=str, help='config file location')
    flags = parser.parse_args()
    config_file = flags.config
    flags = json.load(open(flags.config))

    flags['save_dir'] = os.path.join(flags['trainer']['save_root'], network_utils.unique_model_name(flags))
    flags['config'] = config_file

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
        model.encoder = nn.DataParallel(model.encoder)
        model.decoder = nn.DataParallel(model.decoder)
        print('Parallel training mode enabled!')
    train_params = model.set_train_params((args['optimizer']['learn_rate_encoder'],
                                           args['optimizer']['learn_rate_decoder']))

    # make optimizer
    optm = optim.SGD(train_params, lr=args['optimizer']['learn_rate_encoder'], momentum=0.9, weight_decay=5e-4)
    criterions = network_io.create_loss(args, device=device)
    scheduler = optim.lr_scheduler.MultiStepLR(optm, milestones=eval(args['optimizer']['decay_step']),
                                               gamma=args['optimizer']['decay_rate'])

    # if not resume, train from scratch
    if args['trainer']['resume_epoch'] == 0 and args['trainer']['finetune_dir'] == 'None':
        print('Training decoder {} with encoder {} from scratch ...'.format(args['decoder_name'], args['encoder_name']))
    elif args['trainer']['resume_epoch'] == 0 and args['trainer']['finetune_dir']:
        print('Finetuning model from {}'.format(args['trainer']['finetune_dir']))
        network_utils.load(model, args['trainer']['finetune_dir'], relax_load=True)
    else:
        print('Resume training decoder {} with encoder {} from epoch {} ...'.format(
            args['decoder_name'], args['encoder_name'], args['trainer']['resume_epoch']))
        network_utils.load_epoch(args['save_dir'], args['trainer']['resume_epoch'], model, optm)

    # prepare training
    print('Total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    for c in criterions:
        c.to(device)

    # make data loader
    mean = eval(args['dataset']['mean'])
    std = eval(args['dataset']['std'])
    input_size = eval(args['dataset']['input_size'])
    crop_size = eval(args['dataset']['crop_size'])
    tsfms = [A.Flip(), A.RandomRotate90(), A.Normalize(mean=mean, std=std), ToTensorV2()]
    if input_size[0] != crop_size[0] or input_size[1] != crop_size[1]:
        tsfm_train = A.Compose([A.RandomCrop(*crop_size)]+tsfms)
        tsfm_valid = A.Compose([A.RandomCrop(*crop_size)]+tsfms[2:])
    else:
        tsfm_train = A.Compose(tsfms)
        tsfm_valid = A.Compose(tsfms[-2:])
    train_loader = DataLoader(data_loader.get_loader(
        args['dataset']['data_dir'], args['dataset']['train_file'], transforms=tsfm_train),
        batch_size=args['dataset']['batch_size'], shuffle=True, num_workers=args['dataset']['num_workers'],
        pin_memory=True)
    valid_loader = DataLoader(data_loader.get_loader(
        args['dataset']['data_dir'], args['dataset']['valid_file'], transforms=tsfm_valid),
        batch_size=args['dataset']['batch_size'], shuffle=False, num_workers=args['dataset']['num_workers'],
        pin_memory=True)
    print('Training model on the {} dataset'.format(args['dataset']['ds_name']))
    train_val_loaders = {'train': train_loader, 'valid': valid_loader}

    # train the model
    loss_dict = {}
    for epoch in range(args['trainer']['resume_epoch'], args['trainer']['epochs']):
        # each epoch has a training and validation step
        for phase in ['train', 'valid']:
            start_time = timeit.default_timer()
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()

            loss_dict = model.step(train_val_loaders[phase], device, optm, phase, criterions,
                                   args['trainer']['bp_loss_idx'], True, mean, std)
            network_utils.write_and_print(writer, phase, epoch, args['trainer']['epochs'], loss_dict, start_time)

        # save the model
        if epoch % args['trainer']['save_epoch'] == 0 and epoch != 0:
            save_name = os.path.join(args['save_dir'], 'epoch-{}.pth.tar'.format(epoch))
            network_utils.save(model, epoch, optm, loss_dict, save_name)
    # save model one last time
    save_name = os.path.join(args['save_dir'], 'epoch-{}.pth.tar'.format(args['trainer']['epochs']))
    network_utils.save(model, args['trainer']['epochs'], optm, loss_dict, save_name)
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
    shutil.copyfile(cfg['config'], os.path.join(cfg['save_dir'], 'config.json'))

    # train the model
    train_model(cfg, device, parallel)


if __name__ == '__main__':
    main()
