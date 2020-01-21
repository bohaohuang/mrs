"""

"""


# Built-in

# Libs

# Pytorch
from torch import nn
from torch.nn import functional as F

# Own modules
from network import base_model
from network.backbones import encoders


class UpSample(nn.Module):
    """
    This module defines the upsample operation in the D-LinkNet
    """
    def __init__(self, in_chan, out_chan):
        super(UpSample, self).__init__()
        self.chan = in_chan
        self.conv1 = nn.Conv2d(in_chan, in_chan//4, 1, 1, 0)
        self.tconv = nn.ConvTranspose2d(in_chan//4, in_chan//4, (3, 3), 2, 1, 1)
        self.conv2 = nn.Conv2d(in_chan//4, out_chan, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tconv(x)
        x = self.conv2(x)
        return x


class CenterDilation(nn.Module):
    """
    This module defines the center dilation part of the D-Link Net
    """
    def __init__(self, chan):
        super(CenterDilation, self).__init__()
        self.chan = chan
        self.dconv_1 = nn.Conv2d(chan, chan, 3, 1, 1, 1)
        self.dconv_2 = nn.Conv2d(chan, chan, 3, 1, 2, 2)
        self.dconv_3 = nn.Conv2d(chan, chan, 3, 1, 4, 4)
        self.dconv_4 = nn.Conv2d(chan, chan, 3, 1, 8, 8)

    def forward(self, x):
        x_1 = self.dconv_1(x)
        x_2 = self.dconv_2(x_1)
        x_3 = self.dconv_3(x_2)
        x_4 = self.dconv_4(x_3)
        x = x + x_1 + x_2 + x_3 + x_4
        return x


class DLinkNetDecoder(base_model.Base):
    """
    This module defines part (b) and (c) in the D-LinkNet paper
    Grouping them together to match the MRS naming convention
    """
    def __init__(self, chans, n_class, final_upsample=True):
        super(DLinkNetDecoder, self).__init__()
        self.chans = chans
        self.center_dilation = CenterDilation(self.chans[0])
        self.upsample_1 = UpSample(self.chans[0], self.chans[1])
        self.upsample_2 = UpSample(self.chans[1], self.chans[2])
        self.upsample_3 = UpSample(self.chans[2], self.chans[3])
        self.upsample_4 = UpSample(self.chans[3], self.chans[4])
        if final_upsample:
            self.tconv = nn.ConvTranspose2d(self.chans[4], self.chans[4]//2, 4, 2, 1)
        else:
            self.tconv = nn.Conv2d(self.chans[4], self.chans[4]//2, 3, 1, 1)
        self.classify = nn.Conv2d(self.chans[4]//2, n_class, 3, 1, 1)

    def forward(self, ftr, layers, input_size):
        ftr = self.center_dilation(ftr)
        ftr = self.upsample_1(ftr)
        ftr = ftr + layers[0]
        ftr = self.upsample_2(ftr)
        ftr = ftr + layers[1]
        ftr = self.upsample_3(ftr)
        ftr = ftr + layers[2]
        ftr = self.upsample_4(ftr)
        ftr = self.tconv(ftr)
        return self.classify(ftr)


class DLinkNet(base_model.Base):
    """
    This module is the original DLinknet defined in paper
    http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf
    """
    def __init__(self, n_class, encoder_name='resnet34', pretrained=True, aux_loss=False):
        super(DLinkNet, self).__init__()
        self.n_class = n_class
        self.aux_loss = aux_loss
        self.encoder_name = encoder_name
        self.encoder = encoders.models(self.encoder_name, pretrained, (2, 2, 2, 2, 2), True)
        if 'vgg' in self.encoder_name:
            self.decoder = DLinkNetDecoder(self.encoder.chans, n_class, final_upsample=False)
        else:
            self.decoder = DLinkNetDecoder(self.encoder.chans, n_class)
        if self.aux_loss:
            self.cls = nn.Sequential(
                nn.Linear(self.encoder.chans[0], 256),
                nn.ReLU(),
                nn.Linear(256, self.n_class)
            )
        else:
            self.cls = None

    def forward(self, x):
        # part a: encoder
        input_size = x.size()[2]
        x = self.encoder(x)
        ftr, layers = x[0], x[1:-1]
        # part b and c: center dilation + decoder
        pred = self.decoder(ftr, layers, input_size)
        if self.aux_loss:
            aux = F.adaptive_max_pool2d(input=ftr, output_size=(1, 1)).view(-1, ftr.size(1))
            return pred, self.cls(aux)
        else:
            return pred


if __name__ == '__main__':
    import torch

    net = DLinkNet(2, encoder_name='resnet101', aux_loss=True)
    x = torch.randn((5, 3, 512, 512))
    y, cls = net(x)
    print(y.shape, cls.shape)
