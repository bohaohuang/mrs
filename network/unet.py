"""

"""


# Built-in

# Libs
from tqdm import tqdm

# Pytorch
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# Own modules
from network import base_model
from mrs_utils import misc_utils, vis_utils


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
    def __init__(self, in_chan, out_chan, margin):
        super(UpSampleConv, self).__init__()
        self.margin = margin
        self.up_conv = nn.Conv2d(in_chan, in_chan//2, kernel_size=(3, 3), padding=(1, 1))
        self.conv_1 = nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=(0, 0))
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=(0, 0))
        self.bn_1 = nn.BatchNorm2d(out_chan)
        self.bn_2 = nn.BatchNorm2d(out_chan)
        self.act = nn.PReLU()

    def forward(self, x_1, x_2):
        x = F.interpolate(x_2, scale_factor=2)
        x = self.up_conv(x)
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
    def __init__(self, in_chans, out_chans, margins, n_class):
        super(UnetDecoder, self).__init__()
        assert len(in_chans) == len(out_chans) == len(margins)
        self.uc = []
        for i, o, m in zip(in_chans, out_chans, margins):
            self.uc.append(UpSampleConv(i, o, m))
        self.uc = nn.ModuleList(self.uc)
        self.classify = nn.Conv2d(out_chans[-1], n_class, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, ftr, layers):
        for l, uc in zip(layers, self.uc):
            ftr = uc(l, ftr)
        return self.classify(ftr)


class UNet(base_model.Base):
    """
    This module is the original Unet defined in paper
    """
    def __init__(self, sfn, n_class, encoder_name='base'):
        """
        Initialize the Unet model
        :param sfn: the start filter number, following blocks have n*sfn number of filters
        :param n_class: the number of class
        :param encoder_name: name of the encoder, could be 'base',
        """
        super(UNet, self).__init__()
        self.sfn = sfn
        self.n_class = n_class
        self.encoder_name = misc_utils.stem_string(encoder_name)
        if self.encoder_name == 'base':
            self.encoder = UnetBaseEncoder(self.sfn)
            self.margins = [4, 16, 40, 88]
            self.decode_in_chans = [self.sfn*16, self.sfn*8, self.sfn*4, self.sfn*2]
            self.decode_out_chans = [self.sfn*8, self.sfn*4, self.sfn*2, self.sfn]
            self.lbl_margin = 92
        else:
            raise NotImplementedError('Encoder architecture not supported')
        self.decoder = UnetDecoder(self.decode_in_chans, self.decode_out_chans, self.margins, self.n_class)

    def forward(self, x):
        x = self.encoder(x)
        ftr, layers = x[0], x[1:]
        pred = self.decoder(ftr, layers)
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
            # crop margin if necessary & reduce channel dimension
            if self.lbl_margin > 0:
                label = label[:, :, self.lbl_margin:-self.lbl_margin, self.lbl_margin:-self.lbl_margin]
            for c_cnt, c in enumerate(criterions):
                loss = c(pred, label)
                if phase == 'train' and c_cnt == bp_loss_idx:
                    loss.backward()
                    optm.step()
                c.update(loss, image.size(0))

            if save_image and img_cnt == 0:
                img_image = image.detach().cpu().numpy()
                if self.lbl_margin > 0:
                    img_image = img_image[:,:, self.lbl_margin: -self.lbl_margin, self.lbl_margin: -self.lbl_margin]
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
    from network import network_utils
    network_utils.network_summary(UNet, (3, 572, 572), sfn=32, n_class=2)
