'''Functions for measuring the performance of a classifier.'''
import torch
import torch.nn.functional as F

@torch.no_grad()
def confusion_matrix_metrics(scores, targets, ignore_index=2):
    """Calculate accuracy, precision, recall, IoU and Dice scores for a batch
    of data.

    Parameters
    ----------
    scores
        Output from a network. Dimension 1 is treated as the class dimension.
    targets
        Labels
    ignore_index
        Index on target to ignore when calculating metrics.

    Returns
    -------
        A tuple (acc, iou, prec, rec, dice) containing the accuracy, IoU, 
        precision, recall, and Dice scores.
    """

    pred = scores.argmax(dim=1).reshape(-1)
    targets = targets.reshape(-1)

    if ignore_index is not None:
        pred = pred[targets!=ignore_index]
        targets = targets[targets!=ignore_index]

    pred = pred>0
    targets = targets>0
    tp = (targets & pred).sum()
    tn = (~targets & ~pred).sum()
    fp = (~targets & pred).sum()
    fn = (targets & ~pred).sum()

    eps = 1e-7
    acc = (tp+tn)/(tp+tn+fp+fn+eps)
    iou = tp/(tp+fp+fn+eps)
    prec = tp/(tp+fp+eps)
    rec = tp/(tp+fn+eps)
    dice = 2*tp/(2*tp+fn+eps)

    return acc, iou, prec, rec, dice

class WeightedAverage:
    '''Create exponentially weighted moving average.'''

    def __init__(self, momentum=0.9):

        self.momentum = momentum
        self.n = 0
        self.mov_avg = 0

    def add_value(self, val):

        self.n += 1
        self.mov_avg = self.momentum * self.mov_avg + (1 - self.momentum) * val
        self.smooth = self.mov_avg / (1 - self.momentum ** self.n)

    def get_average(self):

        return self.smooth
   