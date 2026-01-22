"""Classes and functions for measuring the loss of a model."""

import torch
import torch.nn.functional as F


class SingleChannelCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Cross entropy loss for single channel input. This function is a replacement for the 
    binary_cross_entropy_with_logits function in PyTorch, that is, it can be used for models 
    that output single channel scores. But contrary to the PyTorch function, this function
    accepts class weights in the same format as in the cross_entropy function of Pytorch. It
    also accepts an ignore_index and label smoothing.

    The parameters are exactly the same as in the cross_entropy function of PyTorch, except
    that the input must be a single channel tensor. The target must be a tensor with the same
    shape as the input except for the channel dimension, and must only have values 0 or 1.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(weight, reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return single_channel_cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
    
class LabelWeightedCrossEntropyLoss(torch.nn.Module):
    """Return loss weighted by inverse label frequency in an image."""
    
    def __init__(self, reduction="mean"):
        super().__init__()
        
        self.reduction = reduction
        
    def forward(self, input, target):
        
        return label_weighted_loss(input, target, F.cross_entropy, self.reduction)

class FocalLoss(torch.nn.Module):
    """"Focal loss from the paper Focal Loss for Dense Object Detection.
    https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
    """
    
    def __init__(self, weight=None, gamma=2., ignore_index=-100, reduction="mean"):
        super().__init__()
        
        if weight is None:
            weight = [1., 1.]
        
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, input, target):
        
        return focal_loss(input, target, self.weight, self.gamma, self.ignore_index, self.reduction)

class BCELossNorm(torch.nn.Module):
    """BCE loss with minium value of 0. 
    
    The usual BCE loss does not go to 0 if target is not 0 or 1. This class 
    defines a normalized BCE loss that goes to 0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy(input, target, reduction="none")
        # clamp values to avoid infinite at 0 and 1
        t_clamp = torch.log(target).clamp(-100)
        ti_clamp = torch.log(1-target).clamp(-100)
        # normalize
        bce_loss_norm = bce_loss + target*t_clamp+(1-target)*ti_clamp
        
        return bce_loss_norm.mean()

class LabelSmoothingLoss(torch.nn.Module):
    """Label smoothing loss. Binary targets are smoothed according to the value of `smoothing`."""

    def __init__(self, num_classes, smoothing=0.0, weight=None, reduction="mean"):
        """Adapted from https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method

        input should be logits
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

        self.confidence = 1.0 - smoothing      
        self.dim = 1      # Channel dimension

    def reduce_loss(self, loss_per_item):

        if self.reduction == "mean":
            loss = loss_per_item.mean() 
        elif self.reduction == "sum":
            loss = loss_per_item.sum() 

        return loss

    def forward(self, pred, target):

        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            view_shape = (1,)*(pred.ndim-2)
            pred = pred * self.weight.view(1, -1, *view_shape)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(self.dim, target.data.unsqueeze(1), self.confidence)
        loss_per_item = -torch.sum(true_dist * pred, dim=self.dim)

        return self.reduce_loss(loss_per_item)

def single_channel_cross_entropy(
        input, 
        target, 
        weight=None, 
        ignore_index=-100, 
        reduction="mean", 
        label_smoothing=0.0
    ):
    """Single channel cross entropy loss. See SingleChannelCrossEntropyLoss for more details."""

    input_c1 = torch.zeros_like(input)
    # for BCE(x), cross_entropy((-x, 0)) must provide the same result
    input = torch.cat((-input, input_c1), dim=1)

    return F.cross_entropy(
        input, 
        target, 
        weight=weight, 
        ignore_index=ignore_index, 
        reduction=reduction, 
        label_smoothing=label_smoothing
        )

def weighted_cross_entropy(input, target, weight=None, epoch=None):
    """Weighted cross entropy. The probabilities for each pixel are weighted according to
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

    Returns:
    -------
    loss : float
        The calculated loss
    """

    loss_per_pix = F.cross_entropy(input, target, reduction="none")
    loss = (weight*loss_per_pix).mean()

    return loss

def label_weighted_loss(input, target, loss_func=F.cross_entropy, reduction="mean"):
    """Return loss weighted by inverse label frequency. loss_func must have a weight argument."""

    num_pix_in_class = torch.bincount(target.view(-1)).float()
    weight = 1./num_pix_in_class
    weight = weight/weight.sum()
    return loss_func(input, target, weight=weight, reduction=reduction)
         
def focal_loss(input, target, weight, gamma, ignore_index=-100, reduction="mean"):
    """Please refer to the FocalLoss class for more details."""
    
    logpt = F.cross_entropy(input, target, ignore_index=ignore_index, reduction="none")
    pt = torch.exp(-logpt)

    focal_term = (1.0 - pt).pow(gamma)
    loss = focal_term * logpt

    loss *= weight[0]*(1-target) + weight[1]*target
    
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
        
    return loss

class DiceLossRaw(torch.nn.Module):
    """input must be logits."""
    
    def __init__(self, squared=False, eps=1e-8):
        super().__init__()
        
        self.squared = squared
        self.eps = eps
        
    def forward(self, input, target):
        
        probs = F.softmax(input, dim=1)
        return dice_loss(probs, target, self.squared, self.eps)
    
class DiceLoss(torch.nn.Module):
    """input must be probabilities."""
    
    def __init__(self, squared=False, eps=1e-8):
        super().__init__()
        
        self.squared = squared
        self.eps = eps
        
    def forward(self, input, target):
        
        return dice_loss(input, target, self.squared, self.eps)
        
def dice_loss(input, target, squared=False, eps=1e-8):
    """Dice loss. input must be probabilities."""
    
    input_1 = input[:, 1]            # Probabilities for class 1

    numerator = 2*torch.sum(input_1*target)
    if squared:
        input_1 = input_1**2
        target = target**2
    denominator = torch.sum(input_1) + torch.sum(target)

    return 1 - (numerator + eps)/(denominator + eps)  
