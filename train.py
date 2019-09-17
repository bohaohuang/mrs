"""

"""


# Built-in
import os
import timeit

# Libs
import albumentations as A
from albumentations.pytorch import ToTensor
from tensorboardX import SummaryWriter

# Pytorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Own modules
import config
from data import data_loader
from mrs_utils import misc_utils
from network import network_utils, network_io


def train_model(args, device, parallel):
    """
    The function to train the model
    :param args: the class carries configuration parameters defined in config.py
    :param device: the device to run the model
    :return:
    """
    model = network_io.create_model(args)
    log_dir = os.path.join(args.save_dir, 'log')
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model, torch.rand(1, 3, *args.input_size))
    if parallel:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.encoder = nn.DataParallel(model.encoder)
        model.decoder = nn.DataParallel(model.decoder)
    train_params = model.set_train_params((args.learn_rate_encoder, args.learn_rate_decoder))

    # make optimizer
    optm = optim.SGD(train_params, lr=args.learn_rate_encoder, momentum=0.9, weight_decay=5e-4)
    criterions = network_io.create_loss(args)
    scheduler = optim.lr_scheduler.MultiStepLR(optm, milestones=args.decay_step, gamma=args.decay_rate)

    # if not resume, train from scratch
    if args.resume_epoch == 0 and not args.finetune_dir:
        print('Training decoder {} with encoder {} from scratch ...'.format(args.decoder_name, args.encoder_name))
    elif args.resume_epoch == 0 and args.finetune_dir:
        print('Finetuning model from {}'.format(args.finetune_dir))
        network_utils.load(model, args.finetune_dir, relax_load=True)
    else:
        print('Resume training {} from epoch {} ...'.format(args.model_name, args.resume_epoch))
        network_utils.load_epoch(args.save_dir, args.resume_epoch, model, optm)

    # prepare training
    print('Total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    for c in criterions:
        c.to(device)

    # make data loader
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tsfm_train = A.Compose([
        A.Flip(),
        A.RandomRotate90(),
        A.Normalize(mean=mean, std=std),
        ToTensor(sigmoid=False),
    ])
    tsfm_valid = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensor(sigmoid=False),
    ])
    train_loader = DataLoader(data_loader.RSDataLoader(args.data_dir, args.train_file, transforms=tsfm_train),
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(data_loader.RSDataLoader(args.data_dir, args.valid_file, transforms=tsfm_valid),
                              batch_size=args.batch_size, shuffle=False, num_workers=4)
    print('Training model on the {} dataset'.format(args.ds_name))
    train_val_loaders = {'train': train_loader, 'valid': valid_loader}

    # train the model
    for epoch in range(args.resume_epoch, args.epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'valid']:
            start_time = timeit.default_timer()
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            loss_dict = model.step(train_val_loaders[phase], device, optm, phase, criterions,
                                   args.bp_loss_idx, True, mean, std)
            network_utils.write_and_print(writer, phase, epoch, args.epochs, loss_dict, start_time)

        # save the model
        if epoch % args.save_epoch == (args.save_epoch - 1):
            save_name = os.path.join(args.save_dir, 'epoch-{}.pth.tar'.format(epoch))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optm.state_dict(),
                'loss': loss_dict,
            }, save_name)
            print('Saved model at {}'.format(save_name))
    writer.close()


def main():
    # settings
    cfg = config.Args()
    # set gpu to use
    device, parallel = misc_utils.set_gpu(cfg.gpu)
    # set random seed
    misc_utils.set_random_seed(cfg.random_seed)
    # make training directory
    misc_utils.make_dir_if_not_exist(cfg.save_dir)
    misc_utils.args_writer(os.path.join(cfg.save_dir, 'config.json'), cfg)

    # train the model
    train_model(cfg, device, parallel)


if __name__ == '__main__':
    main()
