"""
This implementation is based on:
https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
"""

# Built-in
import math
from collections import OrderedDict
import sys
sys.path.append(r'C:\Users\wh145\Documents\mrs')

# Libs

# Pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from torch.utils import model_zoo

# Own modules
# from network import network_utils


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


fire_cfg = {
    '1_0': [
        [(96, 16, 64, 64), (128, 16, 64, 64), (128, 32, 128, 128)],
        [(256, 32, 128, 128), (256, 48, 192, 192),
         (384, 48, 192, 192), (384, 64, 256, 256)],
        [(512, 64, 256, 256)]
    ],
    '1_1': [
        [(64, 16, 64, 64), (128, 16, 64, 64)],
        [(128, 32, 128, 128), (256, 32, 128, 128)],
        [(256, 48, 192, 192), (384, 48, 192, 192),
         (384, 64, 256, 256), (512, 64, 256, 256)]
    ]
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', strides=(2, 2, 2, 2), inter_features=False, fire_cfg=fire_cfg):
        super(SqueezeNet, self).__init__()
        self.inter_features = inter_features
        self.chans = [a[-1][0] for a in fire_cfg[version][::-1]]
        if version == '1_0':
            self.layer_0 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=strides[0]),
                nn.ReLU(inplace=True)
            )
            self.layer_1 = self._make_layer(
                fire_cfg[version][0], stride=strides[1])
            self.layer_2 = self._make_layer(
                fire_cfg[version][1], stride=strides[2])
            self.layer_3 = self._make_layer(
                fire_cfg[version][2], stride=strides[3])

        elif version == '1_1':
            self.layer_0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True)
            )
            self.layer_1 = self._make_layer(
                fire_cfg[version][0], stride=strides[1])
            self.layer_2 = self._make_layer(
                fire_cfg[version][1], stride=strides[2])
            self.layer_3 = self._make_layer(
                fire_cfg[version][2], stride=strides[3])
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))
        self.chans.extend([self.layer_0[0].out_channels])
        # Final convolutional block for classification is removed

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, fire_cfg, kernel_size=3, stride=2, ceil_mode=True):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        for cfg in fire_cfg:
            layers.append(Fire(*cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.inter_features:
            layer_0 = self.layer_0(x)
            layer_1 = self.layer_1(layer_0)
            layer_2 = self.layer_2(layer_1)
            layer_3 = self.layer_3(layer_2)

            return layer_0, layer_1, layer_2, layer_3
        else:
            x = self.layer_0(x)
            x = self.layer_1(x)
            x = self.layer_2(x)
            x = self.layer_3(x)

            return x


def _squeezenet(version, pretrained, strides, inter_features, progress, **kwargs):
    model = SqueezeNet(version, strides, inter_features, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # The final classification block of squeezenet is removed, corresponding weight and bias should also be removed
        cls_keys = [key for key in state_dict if 'classifier' in key]
        for key in cls_keys: del state_dict[key]
        # Loaded pretrained state_dict has different prefixes for state_dict which need to be renamed
        state_dict = OrderedDict(zip(model.state_dict().keys(), state_dict.values()))
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, strides=(2, 2, 2, 2), inter_features=True,  progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, strides, inter_features,  progress, **kwargs)


def squeezenet1_1(pretrained=False, strides=(2, 2, 2, 2), inter_features=True,  progress=True, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, strides, inter_features,  progress, **kwargs)


if __name__ == '__main__':
    model = squeezenet1_0(True, (2, 2, 2, 2), True)
    from torchsummary import summary
    summary(model, (3, 512, 512), device='cpu')
    