"""

"""


# Built-in
import math

# Libs

# Pytorch
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
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), strides=(2, 2, 2, 1, 1), inter_features=False):
        self.inplanes = 64
        self.inter_features = inter_features
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
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
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

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
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet18']))
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnet34(pretrained=True, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3], strides=strides, inter_features=inter_features)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet34']))
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnet50(pretrained=True, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], strides=strides, inter_features=inter_features)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet50']))
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnet101(pretrained=True, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3], strides=strides, inter_features=inter_features)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet101']))
        model.load_state_dict(pretrained_state, strict=False)
    return model


def resnet152(pretrained=True, strides=(2, 2, 2, 1, 1), inter_features=False):
    model = ResNet(Bottleneck, [3, 8, 36, 3], strides=strides, inter_features=inter_features)
    if pretrained:
        pretrained_state = network_utils.flex_load(model.state_dict(), model_zoo.load_url(model_urls['resnet152']))
        model.load_state_dict(pretrained_state, strict=False)
    return model


if __name__ == '__main__':
    from torchsummary import summary
    model = resnet18(True, strides=(2, 2, 2, 2, 2), inter_features=True)
    summary(model, (3, 512, 512), device='cpu')
