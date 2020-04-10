"""

"""


# Built-in

# Libs
import pytest

# PyTorch
import torch
from torch import nn

# Own modules
from mrs_utils import misc_utils
from network import unet, deeplabv3, pspnet, dlinknet


@pytest.mark.parametrize('decoder_func', [
    unet.UNet, deeplabv3.DeepLabV3, pspnet.PSPNet, dlinknet.DLinkNet
])
@pytest.mark.parametrize('encoder_name', [
    'vgg16_bn', 'vgg19_bn',
    'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2',
    'res2net50_14w_8s', 'res2net50_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s', 'res2net50_48w_2s', 'res2net101_26w_4s',
    'inception_v3',
    'squeezenet1_0', 'squeezenet1_1',
])
def test_model(decoder_func, encoder_name):
    net = decoder_func(2, encoder_name=encoder_name, aux_loss=False, use_emau=False)
    assert isinstance(net, nn.Module)


@pytest.mark.parametrize('decoder_func', [
    unet.UNet, deeplabv3.DeepLabV3, pspnet.PSPNet, dlinknet.DLinkNet
])
@pytest.mark.parametrize('encoder_name', [
    'vgg16_bn', 'vgg19_bn',
    'resnet18', 'resnet101',
    'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2',
    'res2net50_26w_4s', 'res2net101_26w_4s',
    'inception_v3',
    'squeezenet1_0',
])
@pytest.mark.parametrize('input_shape', [
    (5, 3, 512, 512), (5, 3, 1024, 1024),
])
@pytest.mark.parametrize('class_number', [
    2, 7
])
@pytest.mark.parametrize('emau', [
    False, 64
])
@pytest.mark.parametrize('aux', [
    False, True
])
def test_decoder_output_shape(decoder_func, encoder_name, input_shape, class_number, emau, aux):
    device, parallel = misc_utils.set_gpu('1')
    net = decoder_func(class_number, encoder_name=encoder_name, aux_loss=aux, use_emau=emau).to(device)
    x = torch.randn(input_shape).to(device)
    output_dict = net(x)
    for key, val in output_dict.items():
        val_shape = list(val.shape)
        if key == 'pred':
            assert len(val_shape) == 4
            assert val_shape[0] == input_shape[0]
            assert val_shape[1] == class_number
            assert val_shape[2] == input_shape[2]
            assert val_shape[3] == input_shape[3]
        elif key == 'aux':
            assert len(val_shape) == 2
            assert val_shape[0] == input_shape[0]
            assert val_shape[1] == class_number
        elif key == 'mu':
            assert len(val_shape) == 3
            assert val_shape[0] == input_shape[0]
            assert val_shape[2] == emau
