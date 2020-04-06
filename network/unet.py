"""

"""


# Built-in

# Libs

# Pytorch
import torch
from torch import nn
from torch.nn import functional as F

# Own modules
from network import base_model, emau
from network.backbones import encoders
from mrs_utils import misc_utils


class ConvDownSample(nn.Module):
    """
    This module defines conv-downsample block in the Unet
    conv->act->bn -> conv->act->bn -> (pool)
    """
    def __init__(self, in_chan, out_chan, pool=True):
        super(ConvDownSample, self).__init__()
        self.pool = pool
        self.conv_1 = nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=(0, 0))
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=(0, 0))
        self.bn_1 = nn.BatchNorm2d(out_chan)
        self.bn_2 = nn.BatchNorm2d(out_chan)
        self.act = nn.PReLU()
        self.pool2d = nn.MaxPool2d((2, 2), (2, 2))

    def forward(self, x):
        x = self.bn_1(self.act(self.conv_1(x)))
        x = self.bn_2(self.act(self.conv_2(x)))
        if self.pool:
            return x, self.pool2d(x)
        else:
            return x


class UpSampleConv(nn.Module):
    """
    This module defines upsample-concat-conv block in the Unet
    interp->conv
             |
    crop->concat -> conv->act->bn -> conv->act->bn
    """
    def __init__(self, in_chan, out_chan, margin, conv_chan=None, pad=0):
        super(UpSampleConv, self).__init__()
        self.margin = margin
        if not conv_chan:
            conv_chan = out_chan
        self.up_conv = nn.Conv2d(in_chan, in_chan//2, kernel_size=(3, 3), padding=(1, 1))
        self.conv_1 = nn.Conv2d(conv_chan, out_chan, kernel_size=(3, 3), padding=pad)
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=pad)
        self.bn_1 = nn.BatchNorm2d(out_chan)
        self.bn_2 = nn.BatchNorm2d(out_chan)
        self.act = nn.PReLU()

    def forward(self, x_1, x_2):
        x = F.interpolate(x_2, scale_factor=2)
        x = self.up_conv(x)
        if self.margin != 0:
            x_1 = x_1[:, :, self.margin:-self.margin, self.margin:-self.margin]
        x = torch.cat((x_1, x), 1)
        x = self.bn_1(self.act(self.conv_1(x)))
        x = self.bn_2(self.act(self.conv_2(x)))
        return x


class UnetBaseEncoder(nn.Module):
    """
    This module is the original encoder of the Unet
    """
    def __init__(self, sfn):
        super(UnetBaseEncoder, self).__init__()
        self.sfn = sfn
        self.cd_1 = ConvDownSample(3, self.sfn)
        self.cd_2 = ConvDownSample(self.sfn, self.sfn * 2)
        self.cd_3 = ConvDownSample(self.sfn * 2, self.sfn * 4)
        self.cd_4 = ConvDownSample(self.sfn * 4, self.sfn * 8)
        self.cd_5 = ConvDownSample(self.sfn * 8, self.sfn * 16, pool=False)

    def forward(self, x):
        layer0, x = self.cd_1(x)
        layer1, x = self.cd_2(x)
        layer2, x = self.cd_3(x)
        layer3, x = self.cd_4(x)
        layer4 = self.cd_5(x)
        return layer4, layer3, layer2, layer1, layer0


class UnetDecoder(nn.Module):
    """
    This module is the original decoder in the Unet
    """
    def __init__(self, in_chans, out_chans, margins, n_class, conv_chan=None, pad=0, up_sample=0):
        super(UnetDecoder, self).__init__()
        assert len(in_chans) == len(out_chans) == len(margins)
        self.uc = []
        if not conv_chan:
            conv_chan = in_chans
        for i, o, c, m in zip(in_chans, out_chans, conv_chan, margins):
            self.uc.append(UpSampleConv(i, o, m, c, pad))
        self.uc = nn.ModuleList(self.uc)
        self.classify = nn.Conv2d(out_chans[-1], n_class, kernel_size=(3, 3), padding=(1, 1))
        self.up_sample = up_sample

    def forward(self, ftr, layers):
        for l, uc in zip(layers, self.uc):
            ftr = uc(l, ftr)
        if self.up_sample > 0:
            ftr = F.interpolate(ftr, scale_factor=self.up_sample, mode='bilinear')
        return self.classify(ftr)


class UNet(base_model.Base):
    """
    This module is the original Unet defined in paper
    """
    def __init__(self, n_class, sfn=32, encoder_name='base', pretrained=True, aux_loss=False, use_emau=False):
        """
        Initialize the Unet model
        :param sfn: the start filter number, following blocks have n*sfn number of filters
        :param n_class: the number of class
        :param encoder_name: name of the encoder, could be 'base', 'vgg16'
        :param pretrained: if True, load the weights from pretrained model
        :param aux_loss: if True, will create a classification branch for extracted features
        :param use_emau: if True or int, the an EMAU will be appended at the end of the encoder
        """
        super(UNet, self).__init__()
        self.n_class = n_class
        self.encoder_name = misc_utils.stem_string(encoder_name)
        self.aux_loss = aux_loss
        self.use_emau = use_emau
        if self.encoder_name == 'base':
            self.sfn = sfn
            self.encoder = UnetBaseEncoder(self.sfn)
            self.margins = [4, 16, 40, 88]
            filter_nums = [self.sfn*(2 ** a) for a in range(4, -1, -1)]
            self.decode_in_chans = filter_nums[:-1]
            self.decode_out_chans = filter_nums[1:]
            self.lbl_margin = 92
            conv_chan = None
            pad = 0
            up_sample = 0
        else:
            self.encoder = encoders.models(self.encoder_name, pretrained, (2, 2, 2, 2, 2), True)
            self.margins = [0, 0, 0, 0]
            filter_nums = self.encoder.chans
            self.decode_in_chans = filter_nums[:-1]
            self.decode_out_chans = filter_nums[1:]
            self.lbl_margin = 0
            conv_chan = [d_in//2+d_out for (d_in, d_out) in zip(self.decode_in_chans, self.decode_out_chans)]
            pad = 1
            up_sample = 0 if 'vgg' in self.encoder_name else 2
        if self.aux_loss:
            self.cls = nn.Sequential(
                nn.Linear(self.decode_in_chans[0], 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_class)
            )
        else:
            self.cls = None
        if self.use_emau:
            if isinstance(self.use_emau, int):
                c = self.use_emau
            else:
                c = 64
            self.encoder.emau = emau.EMAU(self.decode_in_chans[0], c)
        self.decoder = UnetDecoder(self.decode_in_chans, self.decode_out_chans, self.margins, self.n_class,
                                   conv_chan, pad, up_sample)

    def forward(self, x):
        x = self.encoder(x)
        ftr, layers = x[0], x[1:]
        if self.use_emau:
            ftr, mu = self.encoder.emau(ftr)
            pred = self.decoder(ftr, layers)
            return pred, mu
        if self.aux_loss:
            if self.use_emau:
                ftr, mu = self.encoder.emau(ftr)
            pred = self.decoder(ftr, layers)
            aux = F.adaptive_max_pool2d(input=ftr, output_size=(1, 1)).view(-1, ftr.size(1))
            if self.use_emau:
                return pred, mu, self.cls(aux)
            else:
                return pred, self.cls(aux)
        else:
            pred = self.decoder(ftr, layers)
            return pred


if __name__ == '__main__':
    from network import network_utils
    network_utils.network_summary(UNet, (3, 512, 512), sfn=32, n_class=2, encoder_name='resnet152')
