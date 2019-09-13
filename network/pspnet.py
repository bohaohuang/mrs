"""

"""


# Built-in

# Libs
from tqdm import tqdm

# Pytorch
import torch
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

# Own modules
from network import base_model
from mrs_utils import misc_utils, vis_utils


class PSPVGG16Encoder(nn.Module):
    """
    This module is a VGG16 network as the encoder of the PSPNet
    """
    def __init__(self, pretrained=True):
        super(PSPVGG16Encoder, self).__init__()
        vgg16 = list(models.vgg16(pretrained).features.children())
        maxpool_rm_idx = [23, 30]
        self.vgg16 = [vgg16[l] for l in range(len(vgg16)) if l not in maxpool_rm_idx]
        self.vgg16 = nn.Sequential(*self.vgg16)
        self.out_chan = 512

    def forward(self, x):
        return self.vgg16(x)


class PSPDecoder(nn.Module):
    """
    This module defines the decoder of the PSPNet
    The main body of the code comes from https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py
    """
    def __init__(self, n_class, in_chan, out_chan=1024, bin_sizes=(1, 2, 3, 6),
                 drop_rate=0.3):
        super(PSPDecoder, self).__init__()
        self.stages = nn.ModuleList([self.make_stage(in_chan, size) for size in bin_sizes])
        self.bottleneck = nn.Conv2d(in_chan * (len(bin_sizes) + 1), out_chan, kernel_size=1)
        self.relu = nn.ReLU()
        self.drop_1 = nn.Dropout2d(p=drop_rate)
        self.up_1 = PSPUpsample(out_chan, out_chan//4)
        self.up_2 = PSPUpsample(out_chan//4, out_chan//16)
        self.up_3 = PSPUpsample(out_chan//16, out_chan//16)
        self.drop_2 = nn.Dropout2d(p=drop_rate/2)
        self.final = nn.Conv2d(out_chan//16, n_class, kernel_size=1)

    @staticmethod
    def make_stage(features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [F.interpolate(input=stage(x), size=(h, w), mode='bilinear') for stage in self.stages] + [x]
        bottle = self.bottleneck(torch.cat(tuple(priors), 1))
        bottle = self.drop_1(bottle)
        up = self.up_1(bottle)
        up = self.drop_2(up)
        up = self.up_2(up)
        up = self.drop_2(up)
        up = self.up_3(up)
        up = self.drop_2(up)
        return self.final(up)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(base_model.Base):
    """
    This module is the original Unet defined in paper
    """
    def __init__(self, n_class, out_chan=1024, bin_sizes=(1, 2, 3, 6), drop_rate=0.3,
                 encoder_name='vgg16', pretrained=True):
        """
        Initialize the Unet model
        :param n_class: the number of class
        :param encoder_name: name of the encoder, could be 'base', 'vgg16'
        :param pretrained: if True, load the weights from pretrained model
        """
        super(PSPNet, self).__init__()
        self.n_class = n_class
        self.encoder_name = misc_utils.stem_string(encoder_name)
        if self.encoder_name in ['vgg16', 'vgg']:
            self.encoder = PSPVGG16Encoder(pretrained)
        else:
            raise NotImplementedError('Encoder architecture not supported')
        self.decoder = PSPDecoder(n_class, self.encoder.out_chan, out_chan, bin_sizes, drop_rate)

    def forward(self, x):
        ftr = self.encoder(x)
        pred = self.decoder(ftr)
        return pred

    def step(self, data_loader, device, optm, phase, criterions, bp_loss_idx=0, save_image=True,
             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        loss_dict = {}
        for img_cnt, (image, label) in enumerate(tqdm(data_loader, desc='{}'.format(phase))):
            image = Variable(image, requires_grad=True).to(device)
            label = Variable(label).long().to(device)
            optm.zero_grad()

            # forward step
            if phase == 'train':
                pred = self.forward(image)
            else:
                with torch.autograd.no_grad():
                    pred = self.forward(image)

            # loss
            for c_cnt, c in enumerate(criterions):
                loss = c(pred, label)
                if phase == 'train' and c_cnt == bp_loss_idx:
                    loss.backward()
                    optm.step()
                c.update(loss, image.size(0))

            if save_image and img_cnt == 0:
                img_image = image.detach().cpu().numpy()
                lbl_image = label.cpu().numpy()
                pred_image = pred.detach().cpu().numpy()
                banner = vis_utils.make_tb_image(img_image, lbl_image, pred_image, self.n_class, mean, std)
                loss_dict['image'] = torch.from_numpy(banner)
        for c in criterions:
            loss_dict[c.name] = c.get_loss()
            c.reset()
        return loss_dict

    def set_train_params(self, learn_rate, **kwargs):
        return [
            {'params': self.encoder.parameters(), 'lr': learn_rate[0]},
            {'params': self.decoder.parameters(), 'lr': learn_rate[1]}
        ]


if __name__ == '__main__':
    vgg16 = PSPNet(2)
    from torchsummary import summary
    summary(vgg16, (3, 512, 512), device='cpu')
