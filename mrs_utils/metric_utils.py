"""

"""


# Built-in
import os

# Libs
import torch
import numpy as np
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
    #TODO this does not support multi-categorical loss yet
    orig_loss = criterion(outputs, labels)
    labels = make_one_hot(torch.unsqueeze(labels, dim=1))
    inter_ = torch.sum(outputs * labels)
    union_ = torch.sum(outputs + labels) - inter_
    jaccard_loss = torch.mean((inter_ + delta) / (union_ + delta))
    return alpha * (1 - jaccard_loss) + (1 - alpha) * orig_loss


def iou_metric(truth, pred, divide=False):
    #TODO this has not been tested yet
    truth = truth.flatten()
    pred = pred.flatten()
    intersect = truth*pred
    if not divide:
        return float(np.sum(intersect == 1)), float(np.sum(truth+pred >= 1))
    else:
        return float(np.sum(intersect == 1) / np.sum(truth+pred >= 1))


def parse_dataset_iou(results):
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
