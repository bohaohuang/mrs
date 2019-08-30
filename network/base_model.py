"""
This file defines the abstract model for all the models in this repo
This is suppose to be the parent class for all the other models
"""


# Built-in

# Libs
import torch
from torch import nn

# Own modules
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

    def step(self, **kwargs):
        """
        This function does one forward and backward path in the training
        Print necessary message
        :param kwargs:
        :return:
        """
        raise NotImplementedError


if __name__ == '__main__':
    pass
