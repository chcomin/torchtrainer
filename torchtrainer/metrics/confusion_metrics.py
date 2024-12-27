import torch
from sklearn import metrics

from ..util.post_processing import logits_to_preds

# Alias just to remind that tensors can be on CPU or GPU
type CpuOrCudaTensor = torch.Tensor

class ConfusionMatrixMetrics:
    """Calculate accuracy, precision, recall, IoU and Dice scores for a batch
    of data."""
    def __init__(
            self, 
            threshold: float = 0.5,
            ignore_index: int | None = None
            ):
        """Calculate accuracy, precision, recall, IoU and Dice scores for a batch
        of data.

        Parameters
        ----------
        threshold
            Threshold to apply to the predictions.
        ignore_index
            Index on target to ignore when calculating the metrics.

        Returns
        -------
            A tuple (acc, iou, prec, rec, dice) containing the accuracy, IoU, 
            precision, recall, and Dice scores.
        """
        self.threshold = threshold
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
            Output of a network.
        targets
            Labels
        """
        return confusion_matrix_metrics(scores, targets, self.threshold, self.ignore_index)
        
class AveragePrecisionScore:
    """Calculate the average precision score for a batch of data.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    """

    def __init__(
            self, 
            task: str,
            ignore_index: int | None = None,
            average: str = "macro", 
            sample_weight = None):
        """
        Parameters
        ----------
        task
            Type of classification. Options are 'binary' or 'multilabel'.
        ignore_index
            Target index to ignore in the calculation
        average
            Type of averaging. Options are 'micro', 'macro', 'samples', 'weighted' and None.
        sample_weight
            Sample weights
        """
        self.task = task
        self.ignore_index = ignore_index
        self.average = average
        self.sample_weight = sample_weight

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ) -> float:

        preds = logits_to_preds(scores, task=self.task)
        y_score, y_true = to_sklearn(preds, targets, self.ignore_index)

        return metrics.average_precision_score(
            y_true, y_score, average=self.average, sample_weight=self.sample_weight)

class BalancedAccuracyScore:
    """Calculate the balanced accuracy score for a batch of data.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    """

    def __init__(
            self, 
            threshold: float = 0.5, 
            ignore_index: int | None = None,
            sample_weight = None):
        """
        Parameters
        ----------
        threshold
            Threshold to apply to the predictions
        ignore_index
            Target index to ignore in the calculation
        sample_weight
            Sample weights
        """
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.sample_weight = sample_weight

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ) -> float:

        preds = logits_to_preds(
            scores, task="binary", return_indices=True, threshold=self.threshold)
        y_pred, y_true = to_sklearn(preds, targets, self.ignore_index)

        return metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=self.sample_weight)

class MathewsCorrcoef:
    """
    Calculate the Matthews correlation coefficient for a batch of data.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """
    def __init__(
            self, 
            threshold: float = 0.5, 
            ignore_index: int | None = None,
            sample_weight = None):
        """
        Parameters
        ----------
        threshold
            Threshold to apply to the predictions
        ignore_index
            Target index to ignore in the calculation
        sample_weight
            Sample weights
        """
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.sample_weight = sample_weight

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ) -> float:

        preds = logits_to_preds(
            scores, task="binary", return_indices=True, threshold=self.threshold)
        y_pred, y_true = to_sklearn(preds, targets, self.ignore_index)

        return metrics.matthews_corrcoef(y_true, y_pred, sample_weight=self.sample_weight)

class PrecisionRecallCurve:
    """Calculate the precision-recall curve for a batch of data.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    """
    def __init__(
            self, 
            ignore_index: int | None = None,
            sample_weight = None,
            drop_intermediate: bool = False):
        """
        Parameters
        ----------
        ignore_index
            Target index to ignore in the calculation
        sample_weight
            Sample weights
        drop_intermediate
            Whether to drop some suboptimal thresholds

        Returns
        ------
        precision
            Precision values
        recall
            Recall values
        thresholds
            Thresholds on the decision function used to compute precision and recall
        """
        self.ignore_index = ignore_index
        self.sample_weight = sample_weight
        self.drop_intermediate = drop_intermediate

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ):

        preds = logits_to_preds(scores, task="binary")
        y_score, y_true = to_sklearn(preds, targets, self.ignore_index)

        return metrics.precision_recall_curve(
            y_true, y_score, sample_weight=self.sample_weight, 
            drop_intermediate=self.drop_intermediate
            )

class ROCAUCScore:
    """
    Calculate the area under the ROC curve for a batch of data.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    """
    def __init__(
            self, 
            task: str,
            ignore_index: int | None = None,
            average: str = "macro",
            sample_weight = None,
            multi_class: str = "raise",
            labels = None
            ):
        """
        Parameters
        ----------
        task
            Type of classification. Options are 'binary', 'multiclass' or 'multilabel'.
        ignore_index
            Target index to ignore in the calculation
        average
            Type of averaging. Options are 'micro', 'macro', 'samples', 'weighted' and None.
        sample_weight
            Sample weights
        multi_class
            How to handle multiclass classification. Options are 'raise', 'ovr' and 'ovo'.
        labels
            List of labels to include in the calculation
        """
        self.task = task
        self.ignore_index = ignore_index
        self.average = average
        self.sample_weight = sample_weight
        self.multi_class = multi_class
        self.labels = labels

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ) -> float:

        preds = logits_to_preds(scores, task=self.task)
        y_score, y_true = to_sklearn(preds, targets, self.ignore_index)

        return metrics.roc_auc_score(
            y_true, y_score, average=self.average, sample_weight=self.sample_weight, 
            multi_class=self.multi_class, labels=self.labels)

class ROCCurve:
    """
    Compute the ROC curve for a batch of data.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    """
    def __init__(
            self, 
            ignore_index: int | None = None,
            sample_weight = None,
            drop_intermediate: bool = True):
        """
        Parameters
        ----------
        ignore_index
            Target index to ignore in the calculation
        sample_weight
            Sample weights
        drop_intermediate
            Whether to drop some suboptimal thresholds

        Returns
        ------
        fpr
            False positive rate
        tpr
            True positive rate
        thresholds
            Thresholds on the decision function used to compute fpr and tpr
        """
        self.ignore_index = ignore_index
        self.sample_weight = sample_weight
        self.drop_intermediate = drop_intermediate

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ):

        preds = logits_to_preds(scores, task="binary")
        y_score, y_true = to_sklearn(preds, targets, self.ignore_index)

        return metrics.roc_curve(
            y_true, y_score, sample_weight=self.sample_weight, 
            drop_intermediate=self.drop_intermediate
            )

class WeightedAverage:
    """Create exponentially weighted moving average."""

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
        threshold: float = 0.5,
        ignore_index: int | None = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate accuracy, precision, recall, IoU and Dice scores for a batch
    of data.

    Parameters
    ----------
    scores
        Output of a network.
    targets
        Labels
    threshold
        Threshold to apply to the predictions.
    ignore_index
        Index on target to ignore when calculating the metrics.

    Returns
    -------
        A tuple (acc, iou, prec, rec, dice) containing the accuracy, IoU, 
        precision, recall, and Dice scores.
    """

    eps = torch.finfo().eps

    preds = logits_to_preds(scores, return_indices=True, threshold=threshold)

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    if ignore_index is not None:
        preds = preds[targets!=ignore_index]
        targets = targets[targets!=ignore_index]

    preds = preds>0
    targets = targets>0
    tp = (targets & preds).sum()
    tn = (~targets & ~preds).sum()
    fp = (~targets & preds).sum()
    fn = (targets & ~preds).sum()

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
        threshold: float = 0.5,
        ignore_index: int | None = None
        ) -> dict:
    """Return relevant values from a binary confusion matrix to use on performance metrics.
    See https://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion for more details.

    Parameters
    ----------
    scores
        Output of a network.
    targets
        Labels
    threshold
        Threshold to apply to the predictions.
    ignore_index
        Index on target to ignore when calculating the metrics.

    Returns
    -------
        A dictionary containing many metrics (see the source code for a full list).
    """

    eps = torch.finfo().eps

    preds = logits_to_preds(scores, task="binary", return_indices=True, threshold=threshold)

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    if ignore_index is not None:
        preds = preds[targets!=ignore_index]
        targets = targets[targets!=ignore_index]

    preds = preds>0
    targets = targets>0

    p = targets.sum()
    n = (~targets).sum()
    pp = preds.sum()

    tp = (targets & preds).sum()
    tn = (~targets & ~preds).sum()
    fp = (~targets & preds).sum()
    fn = (targets & ~preds).sum()

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
        "tpr": tpr,
        "fnr": fnr,
        "fpr": fpr,
        "tnr": tnr,
        "ppv": ppv,
        "fdr": fdr,
        "acc": acc,
        "f1": f1,
        "iou": iou,
        "recall": tpr,
        "sensitivity": tpr,
        "specificity": tnr,
        "precision": ppv,
        "dice": f1,
        "jaccard": iou,
    }

    return metrics

@torch.no_grad()
def to_sklearn(
        preds, 
        targets,
        ignore_index: int | None = None,
        ):
    """
    Convert PyTorch multidimensional tensors to numpy arrays for use with scikit-learn metrics.
    The following conversions are applied depending on the shape of the input data 
    (... means zero or more dimensions and n means the product of all dimension sizes
    except the class dimension):
    
    Binary
    preds: (bs, c=1, ...) -> (n,)
    targets: (bs, ...)    -> (n,)
    
    Multiclass:
    preds: (bs, c>1, ...) -> (n, c)
    targets: (bs, ...)    -> (n,)
    
    Multilabel:
    preds: (bs, c>1, ...) -> (n, c)
    targets: (bs, c>1, ...) -> (n, c)

    Parameters
    ----------
    preds
        Logits, probabilities or class indices predicted by a network.
    targets
        Labels
    ignore_index
        Index on target to ignore when calculating metrics. Only supported for binary tasks.

    Returns
    -------
    y_score
        The same values as in `preds`, but in a shape accepted by scikit-learn.
    y_true
        The same values as in `targets`, but in a shape accepted by scikit-learn.
    """

    if preds.ndim==1:
        raise ValueError("Predictions must have at least 2 dimensions.")
    
    is_multilabel = False
    if preds.shape==targets.shape:
        is_multilabel = True
        if targets.max()>1:
            raise ValueError(
                "Multilabel classification requires targets to be an indicator matrix."
                )

    batch_shape = (preds.shape[0], *preds.shape[2:])
    if not is_multilabel and targets.shape!=batch_shape:
        raise ValueError("Targets must have the same shape as predictions.")
    
    if ignore_index is not None and type in ("multiclass", "multilabel"):
        raise ValueError(f"ignore_index is not supported for {type} classification.")
    
    num_classes = preds.shape[1]
    # Reshape tensor from bs x c x s1 x s2 x ... to n x c, where n = bs*s1*s2*...
    def flattener(x): return x.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).squeeze()

    y_score = flattener(preds)

    if is_multilabel:
        y_true = flattener(targets)
    else:
        y_true = targets.flatten()

    if ignore_index is not None:
        y_score = y_score[y_true!=ignore_index]
        y_true = y_true[y_true!=ignore_index]

    return y_score.cpu().numpy(), y_true.cpu().numpy()