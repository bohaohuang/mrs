"""

"""


# Built-in
import math

# Libs

# Pytorch
import torch
from torch import nn
from torch.utils import model_zoo

# Own modules
from network import network_utils


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal', dilation=1,
                 norm_layer=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                                   bias=False))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), strides=(2, 2, 2, 1, 1), inter_features=False, groups=1,
                 width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64
        self.inter_features = inter_features
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation=2//strides[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation=4**(2-strides[4]))
        self.chans = [64, 64*block.expansion, 128*block.expansion, 256*block.expansion, 512*block.expansion][::-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        base_width=self.base_width, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, dilation=dilation,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.inter_features:
            x = self.conv1(x)
            x = self.bn1(x)
            layer0 = self.relu(x)
            x = self.maxpool(layer0)

            layer1 = self.layer1(x)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            return layer4, layer3, layer2, layer1, layer0
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            return x


class Res2Net(nn.Module):

    def __init__(self, block, layers=(3, 4, 23, 3), strides=(2, 2, 2, 1, 1), inter_features=False, baseWidth=26,
                 scale=4, norm_layer=None):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inter_features = inter_features

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation=2//strides[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation=4**(2-strides[4]))
        self.chans = [64, 64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
                      512 * block.expansion][::-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        norm_layer = self._norm_layer
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.baseWidth, scale=self.scale, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale,
                                dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.inter_features:
            x = self.conv1(x)
            x = self.bn1(x)
            layer0 = self.relu(x)
            x = self.maxpool(layer0)

            layer1 = self.layer1(x)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            return layer4, layer3, layer2, layer1, layer0
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            return x


def resnet18(pretrained=True, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2], strides=strides, inter_features=inter_features)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet18']),
                                                   verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnet34(pretrained=True, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3], strides=strides, inter_features=inter_features)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet34']),
                                                   verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnet50(pretrained=True, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], strides=strides, inter_features=inter_features)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet50']),
                                                   verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnet101(pretrained=True, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3], strides=strides, inter_features=inter_features)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet101']),
                                                   verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnet152(pretrained=True, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 8, 36, 3], strides=strides, inter_features=inter_features)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet152']),
                                                   verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnext50_32x4d(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], strides=strides, inter_features=inter_features, groups=32,
                   width_per_group=4)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['resnext50_32x4d']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnext101_32x8d(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3], strides=strides, inter_features=inter_features, groups=32,
                   width_per_group=8)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['resnext101_32x8d']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def wide_resnet50_2(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], strides=strides, inter_features=inter_features, width_per_group=64*2)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['wide_resnet50_2']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def wide_resnet101_2(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3], strides=strides, inter_features=inter_features, width_per_group=64*2)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['wide_resnet101_2']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def res2net50_26w_4s(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], strides=strides, inter_features=inter_features, baseWidth=26, scale=4)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['res2net50_26w_4s']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def res2net50_26w_6s(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], strides=strides, inter_features=inter_features, baseWidth=26, scale=6)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['res2net50_26w_4s']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def res2net50_26w_8s(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], strides=strides, inter_features=inter_features, baseWidth=26, scale=8)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['res2net50_26w_8s']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def res2net50_48w_2s(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], strides=strides, inter_features=inter_features, baseWidth=48, scale=2)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['res2net50_48w_2s']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def res2net50_14w_8s(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], strides=strides, inter_features=inter_features, baseWidth=14, scale=8)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['res2net50_48w_2s']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


def res2net101_26w_4s(pretrained=False, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], strides=strides, inter_features=inter_features, baseWidth=26, scale=4)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(),
                                                   model_zoo.load_url(model_urls['res2net101_26w_4s']), verb=False)
        model.load_state_dict(pretrained_state, strict=False)
    return model


if __name__ == '__main__':
    import torch

    model = res2net101_26w_4s(True, strides=(2, 2, 2, 1, 1), inter_features=False)
    x = torch.randn((5, 3, 512, 512))
    y = model(x)
    print(y.shape)
