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
        return self.forward(*inputs_)

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
        return [{'params': self.parameters(), 'lr': learn_rate}]

    def step(self, data_loader, device, optm, phase, criterions, bp_loss_idx=0, save_image=True,
             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        This function does one forward and backward path in the training
        Print necessary message
        :param kwargs:
        :return:
        """
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
            for c_cnt, c in enumerate(criterions):
                loss = c(pred, label)
                if phase == 'train' and c_cnt == bp_loss_idx:
                    loss.backward()
                    optm.step()
                c.update(loss, image.size(0))

            if save_image and img_cnt == 0:
                img_image = image.detach().cpu().numpy()
                lbl_image = label.cpu().numpy()
                pred_image = pred.detach().cpu().numpy()
                banner = vis_utils.make_tb_image(img_image, lbl_image, pred_image, self.n_class, mean, std)
                loss_dict['image'] = torch.from_numpy(banner)
        for c in criterions:
            loss_dict[c.name] = c.get_loss()
            c.reset()
        return loss_dict

    def step_mixed_batch(self, data_loader_ref, data_loader_others, device, optm, phase, criterions, bp_loss_idx=0,
                         save_image=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        This function does one forward and backward path in the training
        Print necessary message
        :param kwargs:
        :return:
        """
        loss_dict = {}
        for img_cnt, (image, label) in enumerate(tqdm(data_loader_ref, desc='{}'.format(phase))):
            # load data
            for dlo in data_loader_others:
                try:
                    image_other, label_other = next(dlo)
                except StopIteration:
                    dlo = iter(dlo)
                    image_other, label_other = next(dlo)
                image = torch.cat([image, image_other], dim=0)
                label = torch.cat([label, label_other], dim=0)
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
            for c_cnt, c in enumerate(criterions):
                loss = c(pred, label)
                if phase == 'train' and c_cnt == bp_loss_idx:
                    loss.backward()
                    optm.step()
                c.update(loss, image.size(0))

            if save_image and img_cnt == 0:
                img_image = image.detach().cpu().numpy()
                lbl_image = label.cpu().numpy()
                pred_image = pred.detach().cpu().numpy()
                banner = vis_utils.make_tb_image(img_image, lbl_image, pred_image, self.n_class, mean, std)
                loss_dict['image'] = torch.from_numpy(banner)
        for c in criterions:
            loss_dict[c.name] = c.get_loss()
            c.reset()
        return loss_dict


if __name__ == '__main__':
    pass
