"""

"""


# Built-in

# Libs
import torch
from torch import nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Own modules
from network import network_utils


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, inter_feature, init_weights=True):
        super(VGG, self).__init__()
        self.inter_feature = inter_feature
        self.conv_0 = features[0][0]
        self.pool_0 = features[0][1]
        self.conv_1 = features[1][0]
        self.pool_1 = features[1][1]
        self.conv_2 = features[2][0]
        self.pool_2 = features[2][1]
        self.conv_3 = features[3][0]
        self.pool_3 = features[3][1]
        self.conv_4 = features[4][0]
        self.pool_4 = features[4][1]
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if not self.inter_feature:
            x = self.conv_0(x)
            x = self.pool_0(x)
            x = self.conv_1(x)
            x = self.pool_1(x)
            x = self.conv_2(x)
            x = self.pool_2(x)
            x = self.conv_3(x)
            x = self.pool_3(x)
            x = self.conv_4(x)
            return x
        else:
            layer0 = self.conv_0(x)
            x = self.pool_0(layer0)
            layer1 = self.conv_1(x)
            x = self.pool_1(layer1)
            layer2 = self.conv_2(x)
            x = self.pool_2(layer2)
            layer3 = self.conv_3(x)
            x = self.pool_3(layer3)
            layer4 = self.conv_4(x)
            return layer4, layer3, layer2, layer1, layer0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, strides, batch_norm=False):
    in_channels = 3
    all_layer = []
    dilations = [1 for _ in range(len(strides))]
    dilation_cnt = 0
    for cnt, s in enumerate(strides):
        if s == 1:
            dilation_cnt += 1
            dilations[cnt] = 2 ** dilation_cnt
    for cnt, (block, stride) in enumerate(zip(cfg, strides)):
        layers = []
        for v in block:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=dilations[cnt], dilation=dilations[cnt])
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        if stride >= 2:
            all_layer.append([nn.Sequential(*layers), nn.MaxPool2d(kernel_size=2, stride=stride)])
        else:
            all_layer.append([nn.Sequential(*layers), nn.Sequential()])
    return all_layer


cfgs = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}


def _vgg(arch, strides, inter_feature, cfg, batch_norm, pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], strides, batch_norm=batch_norm), inter_feature, **kwargs)
    model.chans = [a[-1] for a in cfgs[cfg]][::-1]
    if pretrained:
        pretrained_state = network_utils.sequential_load(model.state_dict(), load_state_dict_from_url(model_urls[arch]))
        model.load_state_dict(pretrained_state, strict=False)
    return model


def vgg11(pretrained=False, strides=(2, 2, 2, 2, 2), inter_feature=False, **kwargs):
    return _vgg('vgg11', strides, inter_feature, 'A', False, pretrained, **kwargs)


def vgg11_bn(pretrained=False, strides=(2, 2, 2, 2, 2), inter_feature=False, **kwargs):
    return _vgg('vgg11_bn', strides, inter_feature, 'A', True, pretrained, **kwargs)


def vgg13(pretrained=False, strides=(2, 2, 2, 2, 2), inter_feature=False, **kwargs):
    return _vgg('vgg13', strides, inter_feature, 'B', False, pretrained, **kwargs)


def vgg13_bn(pretrained=False, strides=(2, 2, 2, 2, 2), inter_feature=False, **kwargs):
    return _vgg('vgg13_bn', strides, inter_feature, 'B', True, pretrained, **kwargs)


def vgg16(pretrained=False, strides=(2, 2, 2, 2, 2), inter_feature=False, **kwargs):
    return _vgg('vgg16', strides, inter_feature, 'D', False, pretrained, **kwargs)


def vgg16_bn(pretrained=False, strides=(2, 2, 2, 2, 2), inter_feature=False, **kwargs):
    return _vgg('vgg16_bn', strides, inter_feature, 'D', True, pretrained, **kwargs)


def vgg19(pretrained=False, strides=(2, 2, 2, 2, 2), inter_feature=False, **kwargs):
    return _vgg('vgg19', strides, inter_feature, 'E', False, pretrained, **kwargs)


def vgg19_bn(pretrained=False, strides=(2, 2, 2, 2, 2), inter_feature=False, **kwargs):
    return _vgg('vgg19_bn', strides, inter_feature, 'E', True, pretrained, **kwargs)


if __name__ == '__main__':
    model = vgg16(False, (2, 2, 2, 1, 1), True)
    from torchsummary import summary
    summary(model, (3, 512, 512), device='cpu')
