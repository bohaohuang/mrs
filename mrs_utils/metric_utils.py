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


# Own modules
from mrs_utils import misc_utils


def make_one_hot(labels, C=2):
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
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
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


def parse_dataset_iou(results):
    """
    Parse the results in a more readable summarized format
    :param results: evaluated results, generated by eval_on_dataset()
    :return: parsed results, a dictionary with city-wise and overall IoU
    """
    result_sum = {}
    i_all, u_all = 0, 0
    for city_name, sub_result in results.items():
        i_city, u_city = 0, 0
        for city_id, (i, u) in sub_result.items():
            i_city += i
            u_city += u
        result_sum[city_name] = i_city/u_city * 100
        i_all += i_city
        u_all += u_city
    result_sum['overall'] = i_all/u_all * 100
    return result_sum


def eval_on_dataset(file_list, input_size, batch_size, pad, transforms, device, model, city_id_func,
                    save_dir, force_run):
    """
    Eval the performance on a dataset
    :param file_list: list of image names to be evaluated with the model
    :param input_size: input size of the CNN model
    :param batch_size: #samples per batch
    :param pad: #pixels to be padded around the tile
    :param transforms: torchvision transforms for the input image, should be the same as training
    :param device: which GPU to run
    :param model: the CNN model, could be the one defined in model/
    :param city_id_func: function to get city name and city id from a file name
    :param save_dir: path to save the result, results will be saved in json format
    :param force_run: if True, will run the evaluation whether the results file exists or not
    :return:
    """
    misc_utils.make_dir_if_not_exist(save_dir)
    save_file_name = os.path.join(save_dir, 'result.json')
    if not os.path.exists(save_file_name) or force_run:
        results = {}
        for rgb_file, gt_file in file_list:
            city_name, city_id = city_id_func(rgb_file)

            rgb = misc_utils.load_file(rgb_file)
            gt = misc_utils.load_file(gt_file)
            pred = model.eval_tile(rgb, input_size, batch_size, pad, device, transforms)
            i, u = iou_metric(gt, pred)
            if city_name in results:
                results[city_name][city_id] = (i, u)
            else:
                results[city_name] = {}
                results[city_name][city_id] = (i, u)
            print('{}_{}: IoU={:.2f}'.format(city_name, city_id, i/u*100))
        misc_utils.save_file(save_file_name, results)
    else:
        results = misc_utils.load_file(save_file_name)
    result_sum = parse_dataset_iou(results)
    return results, result_sum


if __name__ == '__main__':
    pass
