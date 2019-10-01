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


def make_one_hot(labels, device, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_().to(device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


def weighted_jaccard_loss(outputs, labels, criterion, alpha, delta=1e-12):
    """
    Weighted jaccard loss and a criterion function of choice
    :param outputs: predictions of shape C*H*W
    :param labels: ground truth data of shape H*W
    :param criterion: criterion function, could be cross entropy
    :param alpha: weight on jaccard index function
    :param delta: small value that avoid zero value in denominator
    :return:
    """
    #TODO this does not support multi-categorical loss yet
    orig_loss = criterion(outputs, labels)
    labels = make_one_hot(torch.unsqueeze(labels, dim=1))
    inter_ = torch.sum(outputs * labels)
    union_ = torch.sum(outputs + labels) - inter_
    jaccard_loss = torch.mean((inter_ + delta) / (union_ + delta))
    return alpha * (1 - jaccard_loss) + (1 - alpha) * orig_loss


class WeightedJaccardCriterion(object):
    """
    Weighted Jaccard criterion function used in training
    """
    def __init__(self, alpha, criterion, delta=1e-12):
        """
        :param alpha: weight on jaccard index function
        :param criterion: criterion function, could be cross entropy
        :param delta: small value that avoid zero value in denominator
        """
        self.alpha = alpha
        self.criterion = criterion
        self.delta = delta

    def __call__(self, pred, lbl):
        return weighted_jaccard_loss(pred, lbl, self.criterion, self.alpha, self.delta)


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
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.name = 'xent'
        self.criterion = nn.CrossEntropyLoss()

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
    def __init__(self):
        super(IoU, self).__init__()
        self.name = 'IoU'
        self.numerator = 0
        self.denominator = 0

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
        return self.numerator / self.denominator


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
