'''Functions for measuring the performance of a classifier.'''

import scipy.ndimage as ndi
from functools import partial
import torch.nn.functional as F
import torch

def accuracy(input, target, ignore_index=None):
    '''Calculate intersection over union for predicted probabilities. Assumes background has
    value 0 and the segmentation has value 1

    Parameters
    ----------
    input : torch.Tensor
        Output class probabilities of the network. Must have shape (batch size, num classes, height, width)
    target : torch.Tensor
        Target tensor. Must have shape (batch size, height, width)
    ignore_index : int
        Not implemented!

    Returns
    -------
    iou : float
        The calculated intersection over union
    '''

    res_labels = torch.argmax(input, dim=1)
    sum_labels = 2*target + res_labels

    tp = (sum_labels==3).sum()
    iou = tp.float()/((sum_labels>0).sum().float())

    return iou

def get_prfa(input, target, mea='iou', mask_value=None, ignore_index=None):
    '''Calculate some accuracy measuremets for segmentation results. Assumes background has value 0 and
    the segmentation has value 1. If more than one image in batch, returns a single value representing the
    average performance for all images in the batch.

    Possible measurements are:
        iou : intersection over union
        f1 : f1 score
        prec : precision
        rec : recall

    TODO: Add option to return one value per image in the batch.

    Parameters
    ----------
    input : torch.Tensor
        Output class probabilities of the network. Must have shape (batch size, num classes, height, width)
    target : torch.Tensor
        Target tensor. Must have shape (batch size, height, width)
    mea : string
        Name of the desired measurement from the set {'iou', 'f1', 'prec', 'rec'}
    mask_value : int
        Not implemented!
    ignore_index : int
        Not implemented!

    Returns
    -------
    out_mea : float
        The calculated accuracy
    '''

    beta = torch.tensor(1.)
    target = target.squeeze(1)

    res_labels = torch.argmax(input, dim=1)
    #res_labels = res_labels.view(-1)
    #yb = yb.view(-1)

    # Assumes only values in res_labels are 0 and 1 (two classes)
    y_cases = 2*target + res_labels
    tp = torch.sum(y_cases == 3).item()
    fp = torch.sum(y_cases == 1).item()
    tn = torch.sum(y_cases == 0).item()
    fn = torch.sum(y_cases == 2).item()

    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = torch.tensor(0.)
    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = torch.tensor(0.)
    try:
        f = (1 + beta ** 2) * p * r / (((beta ** 2) * p) + r)
    except ZeroDivisionError:
        f = torch.tensor(0.)
    try:
        iou = tp / (tp + fp + fn)
    except ZeroDivisionError:
        iou = torch.tensor(0.)

    if mea=='iou':
        out_mea = iou
    elif mea=='f1':
        out_mea = f
    elif mea=='prec':
        out_mea = p
    elif mea=='rec':
        out_mea = r

    if not isinstance(out_mea, torch.Tensor):
        out_mea = torch.tensor(out_mea)

    return out_mea

def build_acc_dict(get_prfa):
    '''Build dictionary containing accuracy functions from `get_prfa`

    Returns
    -------
    acc_dict : dict
        Keys indicate accuracy name and values are respecive functions
    '''

    acc_dict = {}
    for mea in ['iou', 'f1', 'prec', 'rec']:
        acc_dict[mea] = partial(get_prfa, mea=mea)
    return acc_dict

def weighted_cross_entropy(input, target, weight=None, epoch=None):
    '''Weighted cross entropy. The probabilities for each pixel are weighted according to
    `weight`.

    Parameters
    ----------
    input : torch.Tensor
        Output from the model
    target : torch.Tensor
        Target segmentation
    weight : torch.Tensor
        Weight assigned to each pixel
    epoch : int
        Current training epoch
    Returns
    -------
    loss : float
        The calculated loss
    '''

    loss_per_pix = F.nll_loss(input, target, reduction='none')
    loss = (weight*loss_per_pix).mean()

    '''bs = input.shape[0]
    num_comps = torch.zeros(bs)
    for idx in range(bs):
        img_lab, num_comps[idx] = ndi.label(np.array(target.to('cpu')))
    avg_num_comp = num_comps.mean()

    if epoch<10:
        loss = loss_class
    else:
        loss = loss_class + 0.0005*avg_num_comp'''

    return loss

def label_weighted_loss(input, target, *args, loss_func=F.cross_entropy):
    '''Return loss weighted by inverse label frequency. loss_func must have a weight argument.'''

    num_pix_in_class = torch.bincount(target.view(-1)).float()
    weight = 1./num_pix_in_class
    weight = weight/weight.sum()
    return loss_func(input, target, weight=weight)


class SmoothenValue:
    '''Create weighted moving average.'''
    def __init__(self, beta):

        self.beta = beta
        self.n = 0
        self.mov_avg = 0

    def add_value(self, val):

        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)