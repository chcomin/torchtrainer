import numpy as np
import torch
from skimage.morphology import skeletonize

from ..util.post_processing import logits_to_preds

type CpuOrCudaTensor = torch.Tensor

class ClDice:
    """Calculate the cldice metric for a batch of data."""
    def __init__(self, reduction: str = 'mean'):
        """
        Parameters
        ----------
        reduction
            Reduction method. Can be 'mean' or 'none'.
        """
        self.reduction = reduction

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor
            ) -> np.ndarray | float:
        """
        Parameters
        ----------
        scores
            Output from a network. Dimension 1 is treated as the class dimension.
        targets
            Labels
        """
        return cl_dice(scores, targets, self.reduction)

@torch.no_grad()
def cl_dice(
    scores: CpuOrCudaTensor, 
    targets: CpuOrCudaTensor, 
    reduction: str = 'mean'
    ) -> torch.Tensor:
    """Calculate the clDice metric for a batch of data."""

    preds = logits_to_preds(scores, return_indices=True)
    preds = preds.cpu().numpy()>0
    targets = targets.cpu().numpy()>0

    scores = []
    for pred, target in zip(preds, targets):
        pred_skel = skeletonize(pred)
        target_skel = skeletonize(target)

        tprec = (pred_skel & target).sum()/pred_skel.sum()
        tsens = (target_skel & pred).sum()/target_skel.sum()

        score = 2*tprec*tsens/(tprec+tsens)
        scores.append(score)

    scores = torch.tensor(scores)

    if reduction == 'mean':
        scores = scores.mean()
    elif reduction == 'none':
        pass

    return scores
