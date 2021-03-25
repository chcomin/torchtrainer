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

def get_prfa(input, target, meas='iou', reduce_batch=True, mask=None):
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
    meas : str or list of str
        Name of the desired measurements from the set {'iou', 'f1', 'prec', 'rec'}
    reduce_batch : bool
        If True, a single value is returned for the batch for each measurement. If False, returns one
        value for each item in the batch for each measurement.
    mask : int
        Values where mask is 0 will be ignored.

    Returns
    -------
    out_mea : torch.Tensor or dict
        The calculated values. If `meas` contains a single measurement, the returned value is a tensor
        with a single value if `reduce_batch` is True or a tensor array of size target.shape[0] if
        `reduce_batch` is False. If `meas` is a list, the function returns a dictionary keyed by
        the metrics names. The values for each item depend on `reduce_batch` as above.
    '''

    input = input.detach()
    if isinstance(meas, str):
        meas = [meas]

    beta = torch.tensor(1.)
    target = target.squeeze(1)

    res_labels = torch.argmax(input, dim=1)

    # Assumes only values in res_labels are 0 and 1 (two classes)
    y_cases = 2*target + res_labels
    if mask is not None:
        y_cases[mask==0] = 4

    axes = list(range(1, target.ndim))   # Exclude batch dimension in sum
    tps = torch.sum(y_cases == 3, dim=axes)
    fps = torch.sum(y_cases == 1, dim=axes)
    tns = torch.sum(y_cases == 0, dim=axes)
    fns = torch.sum(y_cases == 2, dim=axes)

    if reduce_batch:
        tps = tps.sum(dim=0, keepdim=True)
        fps = fps.sum(dim=0, keepdim=True)
        tns = tns.sum(dim=0, keepdim=True)
        fns = fns.sum(dim=0, keepdim=True)

    bs = target.shape[0]
    precisions = torch.zeros(bs)
    recalls = torch.zeros(bs)
    f1s = torch.zeros(bs)
    ious = torch.zeros(bs)
    for idx, (tp, fp, tn, fn) in enumerate(zip(tps, fps, tns, fns)):
        if tp!=0 or fp!=0:
            precision = tp / (tp + fp)
        else:
            precision = 0.
        if tp!=0 or fn!=0:
            recall = tp / (tp + fn)
        else:
            recall = 0.
        if precision!=0 or recall!=0:
            f1 = (1 + beta ** 2) * precision * recall / (((beta ** 2) * precision) + recall)
        else:
            f1 = 0.
        if tp!=0 or fp!=0 or fn!=0:
            iou = tp / (tp + fp + fn)
        else:
            iou = 0.

        precisions[idx] = precision
        recalls[idx] = recall
        f1s[idx] = f1
        ious[idx] = iou

    out_meas = {}
    if 'iou' in meas:
        out_meas['iou'] = ious
    if 'f1' in meas:
        out_meas['f1'] = f1s
    if 'prec' in meas:
        out_meas['prec'] = precisions
    if 'rec' in meas:
        out_meas['rec'] = recalls

    if reduce_batch:
        for k, v in out_meas.items():
            out_meas[k] = v[0]

    if len(out_meas)==1:
        out_meas = list(out_meas.values())[0]

    return out_meas

def build_acc_dict(get_prfa):
    '''Build dictionary containing accuracy functions from `get_prfa`

    Returns
    -------
    acc_dict : dict
        Keys indicate accuracy name and values are respecive functions
    '''

    acc_dict = {}
    for mea in ['iou', 'f1', 'prec', 'rec']:
        acc_dict[mea] = partial(get_prfa, meas=mea)
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

def apply_on_cropped_data(func, has_weight=False, **kwargs):

    if has_weight:
        def func_cropped(input, target, weight, **kwargs):
            if target.ndim>1:
                if input.shape[2:]!=target.shape[1:]:
                    target = center_crop_tensor(target.squeeze(1), (input.shape[0],)+input.shape[2:])
                weight = center_crop_tensor(weight.squeeze(1), (input.shape[0],)+input.shape[2:])
            return func(input, target, weight, **kwargs)
    else:
        def func_cropped(input, target, **kwargs):
            if target.ndim>1:
                if input.shape[2:]!=target.shape[1:]:
                    target = center_crop_tensor(target.squeeze(1), (input.shape[0],)+input.shape[2:])
            return func(input, target, **kwargs)

    return func_cropped

def center_crop_tensor(tensor, out_shape):
    '''Center crop a tensor without copying its contents.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be cropped
    out_shape : tuple
        Desired shape

    Returns
    -------
    tensor : torch.Tensor
        A new view of the tensor with shape out_shape
    '''

    out_shape = torch.tensor(out_shape)
    tensor_shape = torch.tensor(tensor.shape)
    shape_diff = (tensor_shape - out_shape)//2

    for dim_idx, sd in enumerate(shape_diff):
        tensor = tensor.narrow(dim_idx, sd, out_shape[dim_idx])

    return tensor

def center_expand_tensor(self, tensor, out_shape):
    '''Center expand a tensor. Assumes `tensor` is not larger than `out_shape`

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be expanded
    out_shape : tuple
        Desired shape

    Returns
    -------
    torch.Tensor
        A new tensor with shape out_shape
    '''

    out_shape = torch.tensor(out_shape)
    tensor_shape = torch.tensor(tensor.shape)
    shape_diff = (out_shape - tensor_shape)

    pad = []
    for dim_idx, sd in enumerate(shape_diff.flip(0)):
        if sd%2==0:
            pad += [sd//2, sd//2]
        else:
            pad += [sd//2, sd//2+1]

    return F.pad(tensor, pad)


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