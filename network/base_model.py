"""
This file defines the abstract model for all the models in this repo
This is suppose to be the parent class for all the other models
"""


# Built-in

# Libs
from tqdm import tqdm

# PyTorch
import torch
from torch import nn
from torch.autograd import Variable

# Own modules
from data import data_loader
from mrs_utils import vis_utils
from network import network_utils


class Base(nn.Module):
    def __init__(self):
        self.lbl_margin = 0
        super(Base, self).__init__()

    def forward(self, *inputs_):
        """
        Forward operation in network training
        This does not necessarily equals to the inference, i.e., less output in inference
        :param inputs_:
        :return:
        """
        raise NotImplementedError

    def inference(self, *inputs_):
        outputs = self.forward(*inputs_)['pred']
        if isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs

    def init_weight(self):
        """
        Initialize weights of the model
        :return:
        """
        for m in network_utils.iterate_sublayers(self):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)
                torch.nn.init.xavier_uniform(m.bias)

    def set_train_params(self, learn_rate, **kwargs):
        """
        Set training parameters with proper weights
        :param learn_rate:
        :param kwargs:
        :return:
        """
        if 'emau' in kwargs and kwargs['emau']:
            return [
                {'params': self.encoder.parameters(), 'lr': learn_rate[0]},
                {'params': self.decoder.parameters(), 'lr': learn_rate[1]},
                {'params': self.encoder.emau.parameters(), 'lr': learn_rate[1]}
            ]
        else:
            return [
                {'params': self.encoder.parameters(), 'lr': learn_rate[0]},
                {'params': self.decoder.parameters(), 'lr': learn_rate[1]}
            ]

    def step(self, data_loaders, device, optm, phase, criterions, bp_loss_idx=0, save_image=True,
             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), loss_weights=None, use_emau=False,
             cls_criterion=None, cls_weight=0.1):
        """
        This function does one forward and backward path in the training
        Print necessary message
        :param kwargs:
        :return:
        """
        # settings
        if isinstance(bp_loss_idx, int):
            bp_loss_idx = (bp_loss_idx,)
        if loss_weights is None:
            loss_weights = {a: 1.0 for a in bp_loss_idx}
        else:
            assert len(loss_weights) == len(bp_loss_idx)
            loss_weights = [a/sum(loss_weights) for a in loss_weights]
            loss_weights = {a: b for (a, b) in zip(bp_loss_idx, loss_weights)}
        aux_train = False
        if cls_criterion is not None:
            aux_train = True

        # make infi-loop data loader
        mix_batch = False
        data_loader_others = []
        if len(data_loaders) > 1:
            mix_batch = True
            for dlo in data_loaders[1:]:
                data_loader_others.append(data_loader.infi_loop_loader(dlo))

        loss_dict = {}
        for img_cnt, data_dict in enumerate(tqdm(data_loaders[0], desc='{}'.format(phase))):
            if mix_batch and phase == 'train':
                for dlo in data_loader_others:
                    data_dict_other = next(dlo)
                    for key, val in data_dict.items():
                        data_dict[key] = torch.cat([val, data_dict_other[key]], dim=0)

            image = Variable(data_dict['image'], requires_grad=True).to(device)
            label = Variable(data_dict['mask']).long().to(device)
            if aux_train:
                cls = Variable(data_dict['cls']).to(device)
            optm.zero_grad()

            # forward step
            if phase == 'train':
                output_dict = self.forward(image)
            else:
                with torch.autograd.no_grad():
                    output_dict = self.forward(image)

            # loss
            # crop margin if necessary & reduce channel dimension
            if self.lbl_margin > 0:
                label = label[:, self.lbl_margin:-self.lbl_margin, self.lbl_margin:-self.lbl_margin]
            loss_all = 0
            for c_cnt, c in enumerate(criterions):
                loss = c(output_dict['pred'], label)
                if phase == 'train' and c_cnt in bp_loss_idx:
                    loss_all += loss_weights[c_cnt] * loss
                c.update(loss, image.size(0))
            if aux_train:
                aux_loss = cls_criterion(output_dict['aux'], cls)
                loss_all += cls_weight * aux_loss
                cls_criterion.update(aux_loss, image.size(0))
            if phase == 'train':
                if use_emau:
                    with torch.no_grad():
                        mu = output_dict['mu'].mean(dim=0, keepdim=True)
                        momentum = 0.9
                        self.encoder.emau.mu *= momentum
                        self.encoder.emau.mu += mu * (1 - momentum)
                loss_all.backward()
                optm.step()

            if save_image and img_cnt == 0:
                img_image = image.detach().cpu().numpy()
                if self.lbl_margin > 0:
                    img_image = img_image[:, :, self.lbl_margin: -self.lbl_margin, self.lbl_margin: -self.lbl_margin]
                lbl_image = label.cpu().numpy()
                pred_image = output_dict['pred'].detach().cpu().numpy()
                banner = vis_utils.make_tb_image(img_image, lbl_image, pred_image, self.n_class, mean, std)
                loss_dict['image'] = torch.from_numpy(banner)
        for c in criterions:
            loss_dict[c.name] = c.get_loss()
            c.reset()
        if aux_train:
            loss_dict[cls_criterion.name] = cls_criterion.get_loss()
            cls_criterion.reset()
        return loss_dict


if __name__ == '__main__':
    pass
