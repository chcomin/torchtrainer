from sklearn import metrics
import torch

# Alias just to remind that tensors can be on CPU or GPU
type CpuOrCudaTensor = torch.Tensor

class ConfusionMatrixMetrics:
    """Calculate accuracy, precision, recall, IoU and Dice scores for a batch
    of data."""
    def __init__(self, ignore_index: int | None = None):
        """
        Parameters
        ----------
        ignore_index
            Target index to ignore in the calculation
        """
        self.ignore_index = ignore_index

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        scores
            Output from a network. Dimension 1 is treated as the class dimension.
        targets
            Labels
        """
        return confusion_matrix_metrics(scores, targets, self.ignore_index)
        
class AveragePrecisionScore:
    """Calculate the average precision score for a batch of data."""
    def __init__(self, ignore_index: int | None = None):
        """
        Parameters
        ----------
        ignore_index
            Target index to ignore in the calculation
        """
        self.ignore_index = ignore_index

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ) -> torch.Tensor:
        """
        Parameters
        ----------
        scores
            Output from a network. Dimension 1 is treated as the class dimension.
        targets
            Labels
        """
        return average_precision_score(scores, targets, self.ignore_index)

class AUCPrecisionRecall:
    """Calculate the area under the precision-recall curve for a batch of data."""
    def __init__(self, ignore_index: int | None = None):
        """
        Parameters
        ----------
        ignore_index
            Target index to ignore in the calculation
        """
        self.ignore_index = ignore_index

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ) -> torch.Tensor:
        """
        Parameters
        ----------
        scores
            Output from a network. Dimension 1 is treated as the class dimension.
        targets
            Labels
        """
        return auc_precision_recall(scores, targets, self.ignore_index)

class ROCAUCScore:
    """Calculate the area under the ROC curve for a batch of data."""
    def __init__(self, ignore_index: int | None = None):
        """
        Parameters
        ----------
        ignore_index
            Target index to ignore in the calculation
        """
        self.ignore_index = ignore_index

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ) -> torch.Tensor:
        """
        Parameters
        ----------
        scores
            Output from a network. Dimension 1 is treated as the class dimension.
        targets
            Labels
        """
        return roc_auc_score(scores, targets, self.ignore_index)

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

@torch.no_grad()
def confusion_matrix_metrics(
        scores: CpuOrCudaTensor, 
        targets: CpuOrCudaTensor, 
        ignore_index: int | None = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    eps = torch.finfo().eps

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

    acc = (tp+tn)/(tp+tn+fp+fn+eps)
    iou = tp/(tp+fp+fn+eps)
    prec = tp/(tp+fp+eps)
    rec = tp/(tp+fn+eps)
    dice = 2*tp/(2*tp+fp+fn+eps)

    return acc, iou, prec, rec, dice
   
@torch.no_grad()
def confusion_matrix_elements(
        scores: CpuOrCudaTensor, 
        targets: CpuOrCudaTensor, 
        ignore_index: int | None = None
        ) -> dict:
    """Return relevant values from a binary confusion matrix to use on performance metrics.
    See https://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion for more details.

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
        A dictionary containing many metrics (see the source code for a full list).
    """

    eps = torch.finfo().eps

    pred = scores.argmax(dim=1).reshape(-1)
    targets = targets.reshape(-1)

    if ignore_index is not None:
        pred = pred[targets!=ignore_index]
        targets = targets[targets!=ignore_index]

    pred = pred>0
    targets = targets>0

    p = targets.sum()
    n = (~targets).sum()
    pp = pred.sum()
    pn = (~pred).sum()

    tp = (targets & pred).sum()
    tn = (~targets & ~pred).sum()
    fp = (~targets & pred).sum()
    fn = (targets & ~pred).sum()

    tpr = tp/(p+eps)
    fnr = fn/(p+eps)
    fpr = fp/(n+eps)
    tnr = tn/(n+eps)
    ppv = tp/(pp+eps)
    fdr = fp/(pp+eps)
    acc = (tp+tn)/(tp+tn+fp+fn+eps)
    f1 = 2*tp/(2*tp+fp+fn+eps)
    iou = tp/(tp+fp+fn+eps)

    metrics = {
        'tpr': tpr,
        'fnr': fnr,
        'fpr': fpr,
        'tnr': tnr,
        'ppv': ppv,
        'fdr': fdr,
        'acc': acc,
        'f1': f1,
        'iou': iou,
        'recall': tpr,
        'sensitivity': tpr,
        'specificity': tnr,
        'precision': ppv,
        'dice': f1,
        'jaccard': iou,
    }

    return metrics

@torch.no_grad()
def average_precision_score(
    scores: CpuOrCudaTensor,
    targets: CpuOrCudaTensor,
    ignore_index: int | None =None
    ) -> torch.Tensor:
    """Calculate the average precision score for a batch of data."""

    probs, targets = preprocess_sklearn(scores, targets, ignore_index)
    return torch.tensor(metrics.average_precision_score(targets, probs, pos_label=1))

@torch.no_grad()
def auc_precision_recall(
    scores: CpuOrCudaTensor,
    targets: CpuOrCudaTensor,
    ignore_index: int | None =None
    ) -> torch.Tensor:
    """Calculate the area under the precision-recall curve for a batch of data."""

    precisions, recalls, _ = precision_recall_curve(scores, targets, ignore_index)

    return torch.tensor(metrics.auc(recalls, precisions))

@torch.no_grad()
def roc_auc_score(
    scores: CpuOrCudaTensor,
    targets: CpuOrCudaTensor,
    ignore_index: int | None = None
    ) -> torch.Tensor:
    """Calculate the area under the ROC curve for a batch of data."""

    probs, targets = preprocess_sklearn(scores, targets, ignore_index)
    return torch.tensor(metrics.roc_auc_score(targets, probs))

@torch.no_grad()
def precision_recall_curve(
    scores: CpuOrCudaTensor, 
    targets: CpuOrCudaTensor, 
    ignore_index: int | None = None
    ):
    """Calculate the precision-recall curve for a batch of data."""

    probs, targets = preprocess_sklearn(scores, targets, ignore_index)
    precisions, recalls, thresholds = metrics.precision_recall_curve(targets, probs, pos_label=1)
    precisions, recalls, thresholds = torch.from_numpy(precisions), torch.from_numpy(recalls), torch.from_numpy(thresholds.copy())

    return precisions, recalls, thresholds

@torch.no_grad()
def roc_curve(
    scores: CpuOrCudaTensor, 
    targets: CpuOrCudaTensor, 
    ignore_index: int | None = None
    ):
    """Calculate the ROC curve for a batch of data."""

    probs, targets = preprocess_sklearn(scores, targets, ignore_index)
    fpr, tpr, thresholds = metrics.roc_curve(targets, probs, pos_label=1)
    fpr, tpr, thresholds = torch.from_numpy(fpr), torch.from_numpy(tpr), torch.from_numpy(thresholds.copy())

    return fpr, tpr, thresholds

@torch.no_grad()
def preprocess_sklearn(
    scores: CpuOrCudaTensor,
    targets: CpuOrCudaTensor,
    ignore_index: int | None = None
    ):
    """Preprocess scores and targets for sklearn metrics."""

    scores = scores.detach()

    probs = scores.softmax(dim=1)[:,1].reshape(-1)
    targets = targets.reshape(-1)

    if ignore_index is not None:
        probs = probs[targets!=ignore_index]
        targets = targets[targets!=ignore_index]

    return probs.cpu().numpy(), targets.cpu().numpy()