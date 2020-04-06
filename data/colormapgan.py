"""
This implementation comes from the paper: https://arxiv.org/abs/1907.12859
"""


# Built-in
import os

# Libs
import skimage.transform
import numpy as np
from tqdm import tqdm
import albumentations as A
from tensorboardX import SummaryWriter
from albumentations.pytorch import ToTensorV2

# PyTorh
import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Own modules
from data import data_loader, data_utils
from mrs_utils import metric_utils, misc_utils, process_block


class ColorMapGenerator(nn.Module):
    def __init__(self):
        super(ColorMapGenerator, self).__init__()
        self.w = nn.Parameter(torch.ones((256 * 256 * 256, 3)), requires_grad=True)
        self.k = nn.Parameter(torch.zeros((256 * 256 * 256, 3)), requires_grad=True)

    def forward(self, x):
        _, _, h, w = x.shape
        x = x.view((-1, 3))
        inds = (x[:, 0] * 256 * 256 + x[:, 1] * 256 + x[:, 2]).long()
        return torch.tanh(x * self.w[inds, :] + self.k[inds, :]).view((-1, 3, h, w))
        # return torch.clamp(x * self.w[inds, :] + self.k[inds, :], -1, 1).view((-1, 3, h, w))


class ColorMapDiscriminator(nn.Module):
    def __init__(self):
        super(ColorMapDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.view((-1, 1))


class ColorMatcher(object):
    def __init__(self, source_ds_dir, source_ds_list, lr_g=5e-4, lr_d=1e-4):
        self.tsfms = [A.Flip(), A.RandomRotate90(), ToTensorV2()]
        self.source_loader = data_loader.RSDataLoader(source_ds_dir, source_ds_list, transforms=self.tsfms,
                                                      with_label=True)
        self.generator = ColorMapGenerator()
        self.discriminator = ColorMapDiscriminator()
        self.optm_g = optim.Adam(self.generator.parameters(), lr=lr_g)
        self.optm_d = optim.Adam(self.generator.parameters(), lr=lr_d)
        self.criterion = nn.MSELoss()

    def fit_helper(self, target_ds_dir, target_ds_list, device, save_dir, batch_size=1, num_workers=4, total_epoch=20):
        # create writer directory
        writer = SummaryWriter(log_dir=save_dir)

        self.generator.to(device)
        self.discriminator.to(device)
        self.criterion.to(device)

        d_meter, g_meter, all_meter = metric_utils.LossMeter(), metric_utils.LossMeter(), metric_utils.LossMeter()

        source_loader = DataLoader(
            self.source_loader,
            batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
        target_loader = DataLoader(
            data_loader.RSDataLoader(target_ds_dir, target_ds_list, transforms=self.tsfms, with_label=False),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        source_loader = data_loader.infi_loop_loader(source_loader)

        for epoch in range(total_epoch):
            for img_cnt, image_target in enumerate(tqdm(target_loader, desc='Epoch: {}/{}'.format(epoch, total_epoch))):
                image_source, _ = next(source_loader)

                image_source = (image_source / 127.5) - 1
                image_target = (image_target / 127.5) - 1
                image_source = Variable(image_source, requires_grad=True).to(device)
                image_target = Variable(image_target, requires_grad=True).to(device)

                valid = Variable(torch.FloatTensor(image_source.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(torch.FloatTensor(image_source.shape[0], 1).fill_(0.0), requires_grad=False).to(device)

                # generator
                self.optm_g.zero_grad()
                fake_imgs = self.generator(image_source)
                g_loss = self.criterion(self.discriminator(fake_imgs), valid)
                g_loss.backward()
                self.optm_g.step()

                # discriminator
                self.optm_d.zero_grad()
                real_loss = self.criterion(self.discriminator(image_target), valid)
                fake_loss = self.criterion(self.discriminator(fake_imgs.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                self.optm_d.step()

                g_meter.update(g_loss, image_source.size(0))
                d_meter.update(d_loss, image_source.size(0))
                all_meter.update(g_loss+d_loss, image_source.size(0))

            print('loss_d: {:.3f}\tloss_g: {:.3f}\tloss_total: {:.3f} '.format(d_meter.get_loss(), g_meter.get_loss(),
                                                                               all_meter.get_loss()))

            banner_orig = np.floor((image_source.detach().cpu().numpy() + 1) * 127.5)
            banner_fake = np.floor((fake_imgs.detach().cpu().numpy() + 1) * 127.5)
            banner_real = np.floor((image_target.detach().cpu().numpy() + 1) * 127.5)

            grid_orig = torchvision.utils.make_grid(torch.from_numpy(banner_orig)).cpu().numpy().astype(np.uint8)
            grid_fake = torchvision.utils.make_grid(torch.from_numpy(banner_fake)).cpu().numpy().astype(np.uint8)
            grid_real = torchvision.utils.make_grid(torch.from_numpy(banner_real)).cpu().numpy().astype(np.uint8)
            if grid_real.shape[0] != grid_orig.shape[0] or grid_real.shape[1] != grid_orig.shape[1]:
                grid_real = data_utils.change_channel_order(
                    skimage.transform.resize(data_utils.change_channel_order(grid_real),
                                             (grid_orig.shape[1], grid_orig.shape[2]),
                                             preserve_range=True).astype(np.uint8),
                    False)
            grid_img = np.concatenate([grid_orig, grid_fake, grid_real], axis=2)

            writer.add_image('img', grid_img, epoch)
            writer.add_scalar('loss_d', d_meter.get_loss(), epoch)
            writer.add_scalar('loss_g', g_meter.get_loss(), epoch)
            writer.add_scalar('loss_total', all_meter.get_loss(), epoch)
            g_meter.reset()
            d_meter.reset()
            all_meter.reset()

        save_name = os.path.join(save_dir, 'model.pth.tar')
        torch.save({
            'state_dict_d': self.discriminator.state_dict(),
            'state_dict_g': self.generator.state_dict(),
        }, save_name)
        print('Saved model at {}'.format(save_name))

    def fit(self, mapper_name, target_ds_dir, target_ds_list, device, save_dir, batch_size=1, num_workers=4,
            total_epoch=20, force_run=False):
        pb = process_block.BasicProcess(mapper_name, save_dir, func=self.fit_helper)
        pb.run(force_run, target_ds_dir=target_ds_dir, target_ds_list=target_ds_list, device=device, save_dir=save_dir,
               batch_size=batch_size, num_workers=num_workers, total_epoch=total_epoch)


if __name__ == '__main__':
    device, _ = misc_utils.set_gpu('0')

    color_matcher = ColorMatcher(r'/hdd/mrs/inria/ps512_pd0_ol0/patches', r'/hdd/mrs/inria/ps512_pd0_ol0/file_list_train.txt')
    color_matcher.fit_helper(r'/hdd/mrs/deepglobe/14p_pd0_ol0/patches', r'/hdd/mrs/deepglobe/14p_pd0_ol0/file_list_train.txt', device,
                             r'/home/lab/Documents/bohao/tasks/2017.12.11.inria_unet_test/mapper')

    '''color_matcher = ColorMatcher(r'/hdd/mrs/spca/patches',
                                 r'/hdd/mrs/spca/file_list_train.txt')
    color_matcher.fit_helper(r'/hdd/mrs/lbnlb/patches',
                             r'/hdd/mrs/lbnlb/file_list_train.txt', device,
                             r'/home/lab/Documents/bohao/tasks/2017.12.11.inria_unet_test/mapper')'''
