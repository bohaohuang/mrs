"""

"""


# Built-in
try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

# Libs
import numpy as np

# Pytorch
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


# Own modules


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


class LossMeter(LossClass):
    """
    A meter for calculated loss
    """
    def __init__(self, name_ext=''):
        super(LossMeter, self).__init__()
        self.name = 'meter_' + name_ext

    def forward(self, pred, lbl):
        pass


class CrossEntropyLoss(LossClass):
    """
    Cross entropy loss function used in training
    """
    def __init__(self, class_weights=(1., 1.)):
        super(CrossEntropyLoss, self).__init__()
        self.name = 'xent'
        class_weights = torch.tensor([float(a) for a in class_weights])
        self.criterion = nn.CrossEntropyLoss(class_weights)

    def forward(self, pred, lbl):
        if len(lbl.shape) == 4 and lbl.shape[1] == 1:
            lbl = lbl[:, 0, :, :]
        return self.criterion(pred, lbl)


class BCEWithLogitLoss(LossClass):
    """
    Cross entropy loss function used in training
    """
    def __init__(self, device, class_weights=(1., 1.)):
        super(BCEWithLogitLoss, self).__init__()
        self.name = 'bcelogits'
        class_weights = torch.tensor([float(a) for a in class_weights]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def forward(self, pred, lbl):
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
        class_n = list(pred.shape)[1]
        if class_n == 2:
            truth = lbl.flatten().float()
            _, pred = torch.max(pred[:, :, :, :], 1)
            pred = pred.flatten().float()
            intersect = truth * pred
            return torch.sum(intersect == 1), torch.sum(truth + pred >= 1)
        else:
            # multi-label class, code comes from https://github.com/keras-team/keras/issues/11350
            truth = lbl.flatten().float()
            _, pred = torch.max(pred[:, :, :, :], 1)
            pred = pred.flatten().float()
            intersect, union = 0, 0
            for i in range(class_n):
                true_labels = torch.eq(truth, i)
                pred_labels = torch.eq(pred, i)
                intersect += torch.sum(true_labels * pred_labels == 1)
                union += torch.sum(true_labels + pred_labels >= 1)
            return intersect, union

    def update(self, loss, size):
        self.numerator += loss[0].item() * size
        self.denominator += loss[1].item() * size

    def reset(self):
        self.numerator = 0
        self.denominator = 0

    def get_loss(self):
        return self.numerator / (self.denominator + self.delta)


class FocalLoss(LossClass):
    """
    Focal loss: this code comes from
    https://github.com/mbsariyildiz/focal-loss.pytorch/blob/6551bd3e433ce41020b6bc8d99221eb6cd10ae17/focalloss.py#L33
    """
    def __init__(self, device, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.name = 'focal'
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha]).to(device)
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha).to(device)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
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


class LovaszSoftmax(LossClass):
    """
    LovaszSoftmax loss, this code comes from:
    https://github.com/bermanmaxim/LovaszSoftmax/blob/7d48792d35a04d3167de488dd00daabbccd8334b/pytorch/lovasz_losses.py
    """
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.name = 'lovasz'
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, pred, lbl):
        if self.per_image:
            loss = self.mean(
                self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore))
                for prob, lab in zip(pred, lbl))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(pred, lbl, self.ignore))
        return loss

    def lovasz_softmax_flat(self, probas, labels):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if self.classes in ['all', 'present'] else self.classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (self.classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(self.classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted))))
        return self.mean(losses)

    @staticmethod
    def mean(l, ignore_nan=False, empty=0):
        """
        nan mean compatible with generators.
        """
        def isnan(x):
            return x != x
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    @staticmethod
    def flatten_probas(probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

    @staticmethod
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


def iou_metric(truth, pred, divide=False, eval_class=(1,)):
    """
    Compute IoU, i.e., jaccard index
    :param truth: truth data matrix, should be H*W
    :param pred: prediction data matrix, should be the same dimension as the truth data matrix
    :param divide: if True, will return the IoU, otherwise return the numerator and denominator
    :param eval_class: the label class to be evaluated
    :return:
    """
    truth = truth.flatten()
    pred = pred.flatten()
    iou_score = np.zeros((2, len(eval_class)), dtype=float)
    for c_cnt, curr_class in enumerate(eval_class):
        iou_score[0, c_cnt] += np.sum(((truth == curr_class) * (pred == curr_class)) == 1)
        iou_score[1, c_cnt] += np.sum(((truth == curr_class) + (pred == curr_class)) >= 1)
    if not divide:
        return iou_score
    else:
        return np.mean(iou_score[0, :] / iou_score[1, :])


if __name__ == '__main__':
    pass
