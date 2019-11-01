"""

"""


# Built-in

# Libs

# Pytorch
import torch
import torch.nn.functional as F
from torch import nn

# Own modules
from network import base_model
from network.backbones import encoders


class _ASPPModule(nn.Module):
    """
    The atrous conv block defined in the paper, This code comes from:
    https://github.com/jfzhang95/pytorch-deeplab-xception/blob/c8ff02c6eeb2b774ad5a25557a60ee48e04635c6/modeling/aspp.py#L7
    """
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    """
    The ASPP module defined in the paper, this code modifies from:
    https://github.com/jfzhang95/pytorch-deeplab-xception/blob/c8ff02c6eeb2b774ad5a25557a60ee48e04635c6/modeling/aspp.py#L34
    """
    def __init__(self, inchan, outchan=256, dropout_rate=0.5):
        super(ASPP, self).__init__()
        self.inchan = inchan
        self.dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inchan, outchan, 1, padding=0, dilation=self.dilations[0], BatchNorm=nn.BatchNorm2d)
        self.aspp2 = _ASPPModule(inchan, outchan, 3, padding=self.dilations[1], dilation=self.dilations[1],
                                 BatchNorm=nn.BatchNorm2d)
        self.aspp3 = _ASPPModule(inchan, outchan, 3, padding=self.dilations[2], dilation=self.dilations[2],
                                 BatchNorm=nn.BatchNorm2d)
        self.aspp4 = _ASPPModule(inchan, outchan, 3, padding=self.dilations[3], dilation=self.dilations[3],
                                 BatchNorm=nn.BatchNorm2d)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inchan, outchan, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(outchan),
                                             nn.ReLU())
        self.final_conv = nn.Sequential(nn.Conv2d(outchan*5, outchan, 1, bias=False),
                                        nn.BatchNorm2d(outchan),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate))
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.final_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabV3Decoder(base_model.Base):
    """
    This module defines part (b) and (c) in the D-LinkNet paper
    Grouping them together to match the MRS naming convention
    """
    def __init__(self, class_num, inchan, layerchan, outchan=256, dropout_rate=0.5):
        super(DeepLabV3Decoder, self).__init__()
        self.aspp = ASPP(inchan, outchan=outchan, dropout_rate=dropout_rate)
        self.conv1 = nn.Sequential(nn.Conv2d(layerchan, 48, 1, bias=False),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(304, outchan, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(outchan),
                                   nn.ReLU(),
                                   nn.Dropout(dropout_rate),
                                   nn.Conv2d(outchan, outchan, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(outchan),
                                   nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(outchan, class_num, kernel_size=1, stride=1))

    def forward(self, ftr, layer, input_size):
        # conv intermediate layer
        layer = self.conv1(layer)

        # upsample encoder feature
        ftr = self.aspp(ftr)
        upsample = F.interpolate(ftr, size=layer.size()[2:], mode='bilinear', align_corners=True)
        ftr = torch.cat([upsample, layer], dim=1)

        # make predictions
        pred = self.conv3(ftr)
        x = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)

        return x


class DeepLabV3(base_model.Base):
    """
    This module implements the DeepLabV3 in paper https://arxiv.org/pdf/1802.02611.pdf
    """
    def __init__(self, n_class, encoder_name='resnet34', pretrained=True, outchan=256, dropout_rate=0.5):
        super(DeepLabV3, self).__init__()
        self.n_class = n_class
        self.encoder_name = encoder_name
        self.encoder = encoders.models(self.encoder_name, pretrained, (2, 2, 2, 2, 1), True)
        self.decoder = DeepLabV3Decoder(n_class, self.encoder.chans[0], self.encoder.chans[3], outchan, dropout_rate)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.encoder(x)
        ftr, layer = x[0], x[3]
        pred = self.decoder(ftr, layer, input_size)
        return pred
