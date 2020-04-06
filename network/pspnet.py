"""

"""


# Built-in

# Libs
from tqdm import tqdm

# Pytorch
import torch
from torch import nn
from torch.nn import functional as F

# Own modules
from network import base_model, emau
from network.backbones import encoders
from mrs_utils import misc_utils


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
        self.relu = nn.ReLU(inplace=True)
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
                 encoder_name='vgg16', pretrained=True, aux_loss=False, use_emau=False):
        """
        Initialize the Unet model
        :param n_class: the number of class
        :param encoder_name: name of the encoder, could be 'base', 'vgg16'
        :param pretrained: if True, load the weights from pretrained model
        :param aux_loss: if True, will create a classification branch for extracted features
        :param use_emau: if True or int, the an EMAU will be appended at the end of the encoder
        """
        super(PSPNet, self).__init__()
        self.n_class = n_class
        self.aux_loss = aux_loss
        self.use_emau = use_emau
        self.encoder_name = misc_utils.stem_string(encoder_name)
        strides = (2, 2, 2, 1, 1)
        self.encoder = encoders.models(self.encoder_name, pretrained, strides, False)
        if self.use_emau:
            if isinstance(self.use_emau, int):
                c = self.use_emau
            else:
                c = 64
            self.encoder.emau = emau.EMAU(self.encoder.chans[0], c)
        self.decoder = PSPDecoder(n_class, self.encoder.chans[0], out_chan, bin_sizes, drop_rate)
        if self.aux_loss:
            self.cls = nn.Sequential(
                nn.Linear(self.encoder.chans[0], 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_class)
            )
        else:
            self.cls = None

    def forward(self, x):
        ftr = self.encoder(x)
        if self.use_emau:
            ftr, mu = self.encoder.emau(ftr)
            pred = self.decoder(ftr)
            return pred, mu
        if self.aux_loss:
            if self.use_emau:
                ftr, mu = self.encoder.emau(ftr)
            pred = self.decoder(ftr)
            aux = F.adaptive_max_pool2d(input=ftr, output_size=(1, 1)).view(-1, ftr.size(1))
            if self.use_emau:
                return pred, mu, self.cls(aux)
            else:
                return pred, self.cls(aux)
        else:
            pred = self.decoder(ftr)
            return pred


if __name__ == '__main__':
    vgg16 = PSPNet(2, encoder_name='vgg16_bn', aux_loss=True, use_emau=True)
    x = torch.randn((5, 3, 512, 512))
    y, cls = vgg16(x)
    print(y.shape, cls.shape)
