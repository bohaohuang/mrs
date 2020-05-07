"""

"""


# Built-in

# Libs
from tqdm import tqdm

# PyTorch
import torch
from torch import nn
from torch.autograd import Variable

# Own modules
from mrs_utils import misc_utils, vis_utils


class TransformLayer(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    def step(self, model, data_loaders, device, optm, phase, criterions, bp_loss_idx=0, loss_weights=None,
             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), class_num=2, normalize=False):
        loss_dict = {}
        for img_cnt, data_dict in enumerate(tqdm(data_loaders[0], desc='{}'.format(phase))):
            if not normalize:
                data_dict['image'] = (data_dict['image'] / 127.5) - 1
            image = Variable(data_dict['image'], requires_grad=True).to(device)
            label = Variable(data_dict['mask']).long().to(device)

            # forward step
            if phase == 'train':
                image_adjust = self.forward(image)
                output_dict = model.forward(image_adjust)
            else:
                with torch.autograd.no_grad():
                    image_adjust = self.forward(image)
                    output_dict = model.forward(image_adjust)

            # loss
            # crop margin if necessary & reduce channel dimension
            if model.lbl_margin > 0:
                label = label[:, model.lbl_margin:-model.lbl_margin, model.lbl_margin:-model.lbl_margin]
            loss_all = 0
            for c_cnt, c in enumerate(criterions):
                loss = c(output_dict['pred'], label)
                # FIXME adhoc solution for OCRNet's region supervision
                if phase == 'train' and c_cnt in bp_loss_idx:
                    loss_all += loss_weights[c_cnt] * loss
                c.update(loss, image.size(0))
            if phase == 'train':
                loss_all.backward()
                optm.step()

            # make image for tensorboard
            if img_cnt == 0:
                img_image = image.detach().cpu().numpy()
                image_adjust = image_adjust.detach().cpu().numpy()

                if model.lbl_margin > 0:
                    img_image = img_image[:, :, model.lbl_margin: -model.lbl_margin,
                                model.lbl_margin: -model.lbl_margin]
                lbl_image = label.cpu().numpy()
                pred_image = output_dict['pred'].detach().cpu().numpy()
                banner_cmp = vis_utils.make_image_banner([img_image, image_adjust, lbl_image, pred_image],
                                                         class_num, mean, std, max_ind=(3,), decode_ind=(2, 3))
                loss_dict['image'] = torch.from_numpy(banner_cmp)

        for c in criterions:
            loss_dict[c.name] = c.get_loss()
            c.reset()

        return loss_dict


class GammaAdjustTransform(TransformLayer):
    def __init__(self):
        super(GammaAdjustTransform, self).__init__()
        self.gamma = nn.Parameter(torch.ones((1,)), requires_grad=True)

    def forward(self, x):
        return torch.pow(x, self.gamma)


class AffineTransform(TransformLayer):
    def __init__(self):
        super(AffineTransform, self).__init__()
        self.a = nn.Parameter(torch.ones((1,)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros((1,)), requires_grad=True)

    def forward(self, x):
        return self.a * x + self.b


class ColorMap(TransformLayer):
    def __init__(self):
        super(ColorMap, self).__init__()
        self.w = nn.Parameter(torch.ones((256 * 256 * 256, 3)), requires_grad=True)
        self.k = nn.Parameter(torch.zeros((256 * 256 * 256, 3)), requires_grad=True)

    def forward(self, x):
        _, _, h, w = x.shape
        x = x.view((-1, 3))
        inds = (x[:, 0] * 256 * 256 + x[:, 1] * 256 + x[:, 2]).long()
        return torch.clamp(x * self.w[inds, :] + self.k[inds, :], -1, 1).view((-1, 3, h, w))


def create_preproc_layer(preproc_name):
    preproc_name = misc_utils.stem_string(preproc_name)
    if preproc_name == 'gamma':
        preproc_layer = GammaAdjustTransform()
    elif preproc_name == 'affine':
        preproc_layer = AffineTransform()
    elif preproc_name == 'colormap':
        preproc_layer = ColorMap()
    else:
        raise NotImplementedError('Preproc layer {} not supported'.format(preproc_name))
    return preproc_layer
