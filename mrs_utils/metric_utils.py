"""

"""


# Built-in
import os

# Libs
import numpy as np

# Pytorch
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


# Own modules


class FocalLoss(nn.Module):
    """
    Focal loss: this code comes from
    https://github.com/mbsariyildiz/focal-loss.pytorch/blob/6551bd3e433ce41020b6bc8d99221eb6cd10ae17/focalloss.py#L33
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class LossClass(nn.Module):
    """
    The base class of loss metrics, all loss metrics should inherit from this class
    This class contains a function that defines how loss is computed (def forward) and a loss tracker that keeps
    updating the loss within an epoch
    """
    def __init__(self):
        super(LossClass, self).__init__()
        self.loss = 0
        self.cnt = 0

    def forward(self, pred, lbl):
        raise NotImplementedError

    def update(self, loss, size):
        """
        Update the current loss tracker
        :param loss: the computed loss
        :param size: #elements in the batch
        :return:
        """
        self.loss += loss.item() * size
        self.cnt += 1

    def reset(self):
        """
        Reset the loss tracker
        :return:
        """
        self.loss = 0
        self.cnt = 0

    def get_loss(self):
        """
        Get mean loss within this epoch
        :return:
        """
        return self.loss / self.cnt


class CrossEntropyLoss(LossClass):
    """
    Cross entropy loss function used in training
    """
    def __init__(self, class_weights=(1, 1)):
        super(CrossEntropyLoss, self).__init__()
        self.name = 'xent'
        class_weights = torch.tensor(class_weights)
        self.criterion = nn.CrossEntropyLoss(class_weights)

    def forward(self, pred, lbl):
        if len(lbl.shape) == 4 and lbl.shape[1] == 1:
            lbl = lbl[:, 0, :, :]
        return self.criterion(pred, lbl)


class SoftIoULoss(LossClass):
    """
    Soft IoU loss that is differentiable
    This code comes from https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5
    Paper: http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
    """
    def __init__(self, device, delta=1e-12):
        super(SoftIoULoss, self).__init__()
        self.name = 'softIoU'
        self.device = device
        self.delta = delta

    def forward(self, pred, lbl):
        num_classes = pred.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[lbl.squeeze(1)].to(self.device)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(pred)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[lbl.squeeze(1)].to(self.device)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(pred, dim=1)
        true_1_hot = true_1_hot.type(pred.type())
        dims = (0,) + tuple(range(2, lbl.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.delta)).mean()
        return (1 - dice_loss)


class IoU(LossClass):
    """
    IoU metric that is not differentiable in training
    """
    def __init__(self, delta=1e-7):
        super(IoU, self).__init__()
        self.name = 'IoU'
        self.numerator = 0
        self.denominator = 0
        self.delta = delta

    def forward(self, pred, lbl):
        truth = lbl.flatten().float()
        _, pred = torch.max(pred[:, :, :, :], 1)
        pred = pred.flatten().float()
        intersect = truth * pred
        return torch.sum(intersect == 1), torch.sum(truth + pred >= 1)

    def update(self, loss, size):
        self.numerator += loss[0].item() * size
        self.denominator += loss[1].item() * size

    def reset(self):
        self.numerator = 0
        self.denominator = 0

    def get_loss(self):
        return self.numerator / (self.denominator + self.delta)


def iou_metric(truth, pred, divide=False):
    """
    Compute IoU, i.e., jaccard index
    :param truth: truth data matrix, should be H*W
    :param pred: prediction data matrix, should be the same dimension as the truth data matrix
    :param divide: if True, will return the IoU, otherwise return the numerator and denominator
    :return:
    """
    truth = truth.flatten()
    pred = pred.flatten()
    intersect = truth*pred
    if not divide:
        return float(np.sum(intersect == 1)), float(np.sum(truth+pred >= 1))
    else:
        return float(np.sum(intersect == 1) / np.sum(truth+pred >= 1))


if __name__ == '__main__':
    pass
