import numpy as np
import torch
import torch.nn.functional as F

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

    loss_per_pix = F.cross_entropy(input, target, reduction='none')
    loss = (weight*loss_per_pix).mean()

    return loss

class LabelWeightedCrossEntropyLoss(torch.nn.Module):
    '''Return loss weighted by inverse label frequency in an image.'''
    
    def __init__(self, reduction='mean'):
        super().__init__()
        
        self.reduction = reduction
        
    def forward(self, input, target):
        
        return label_weighted_loss(input, target, F.cross_entropy, self.reduction)

def label_weighted_loss(input, target, loss_func=F.cross_entropy, reduction='mean'):
    '''Return loss weighted by inverse label frequency. loss_func must have a weight argument.'''

    num_pix_in_class = torch.bincount(target.view(-1)).float()
    weight = 1./num_pix_in_class
    weight = weight/weight.sum()
    return loss_func(input, target, weight=weight, reduction=reduction)

class FocalLoss(torch.nn.Module):
    
    def __init__(self, weight=None, gamma=2., ignore_index=-100, reduction='mean'):
        super().__init__()
        
        if weight is None:
            weight = [1., 1.]
        
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, input, target):
        
        return focal_loss(input, target, self.weight, self.gamma, self.ignore_index, self.reduction)
          
def focal_loss(input, target, weight, gamma, ignore_index=-100, reduction='mean'):
    
    logpt = F.cross_entropy(input, target, ignore_index=ignore_index, reduction='none')
    pt = torch.exp(-logpt)

    focal_term = (1.0 - pt).pow(gamma)
    loss = focal_term * logpt

    loss *= weight[0]*(1-target) + weight[1]*target
    
    if reduction == 'none':
        pass
    elif reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
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
    
    input_1 = input[:, 1]            # Probabilities for class 1

    numerator = 2*torch.sum(input_1*target)
    if squared:
        input_1 = input_1**2
        target = target**2
    denominator = torch.sum(input_1) + torch.sum(target)

    return 1 - (numerator + eps)/(denominator + eps)  

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return (v*s).sum()/s.sum()

def cl_dice(input, target):
    """[this function computes the cldice metric]
    """

    from skimage.morphology import skeletonize

    if input.ndim!=4:
        raise ValueError(f"Expected input to have dimension 4, but got tensor with sizes {input.shape}")
    if target.ndim!=3:
        raise ValueError(f"Expected target to have dimension 3, but got tensor with sizes {target.shape}")

    bs = input.shape[0]
    res_labels = torch.argmax(input, dim=1)

    res_labels = np.array(res_labels.to('cpu')).astype(np.uint8)
    target = np.array(target.to('cpu')).astype(np.uint8)
    cl_dice_per_img = np.zeros(bs)
    for idx in range(bs):
        tprec = cl_score(res_labels[idx],skeletonize(target[idx]))
        tsens = cl_score(target[idx],skeletonize(res_labels[idx]))
        cl_dice_per_img[idx] = 2*tprec*tsens/(tprec+tsens)  
    cl_dice_batch = cl_dice_per_img.mean()

    return cl_dice_batch

class BCELossNorm(torch.nn.Module):
    """BCE loss with minium value of 0. 
    
    The usual BCE loss does not go to 0 if target is not 0 or 1. This class 
    defines a normalized BCE loss that goes to 0."""

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy(input, target, reduction='none')
        # clamp values to avoid infinite at 0 and 1
        t_clamp = torch.log(target).clamp(-100)
        ti_clamp = torch.log(1-target).clamp(-100)
        # normalize
        bce_loss_norm = bce_loss + target*t_clamp+(1-target)*ti_clamp
        
        return bce_loss_norm.mean()

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.0, weight=None, reduction='mean'):
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

        if self.reduction == 'mean':
            loss = loss_per_item.mean() 
        elif self.reduction == 'sum':
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