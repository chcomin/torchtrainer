"""Utilities for using during the model test phase."""

import torch
from torchvision.transforms.v2.functional import resize
from tqdm.auto import tqdm

from torchtrainer.metrics import ConfusionMatrixMetrics


class TTATransform:
    """Base TTA transform class. Every transform must implement the __call__ method 
    and the inv method that reverts the transformation.
    """

    def __call__(self):
        raise NotImplementedError
    
    def inv(self):
        raise NotImplementedError
    
class Rotation(TTATransform):
    """Rotate image by 90 degrees k times."""

    def __init__(self, k=1):
        """k must be 1, 2 or 3"""
        self.k = k

    def __call__(self, img):
        return torch.rot90(img, self.k, (2, 1))
    
    def inv(self, img):
        return torch.rot90(img, -self.k, (2, 1))
    
class Flip(TTATransform):
    """Flip image along a given dimension."""
    
    def __init__(self, dim):
        """dim must be 1 or 2."""
        self.dim = dim

    def __call__(self, img):
        return torch.flip(img, [self.dim])
    
    def inv(self, img):
        return torch.flip(img, [self.dim])

class ReflectDiag(TTATransform):
    """Reflect image along the main diagonal or the secondary diagonal."""

    def __init__(self, main_diag=True):
        self.main_diag = main_diag

    def __call__(self, img):
        if self.main_diag:
            return torch.transpose(img, 1, 2)
        else:
            return torch.rot90(img, 3, (2, 1)).flip([2])
    
    def inv(self, img):
        if self.main_diag:
            return torch.transpose(img, 1, 2)
        else:
            return torch.flip(img, [2]).rot90(1, (2, 1))

class Scale(TTATransform):
    """Scale image up or down by a small percentage."""
    
    def __init__(self, shape, perc=5, scale_up=True):
        """Shape is the original shape of the image."""
        self.shape = shape

        change = (int(shape[0]*perc/100), int(shape[1]*perc/100))

        if scale_up:
            self.new_shape = shape[0] + change[0], shape[1] + change[1]
        else:
            self.new_shape = shape[0] - change[0], shape[1] - change[1]

    def __call__(self, img):
        return resize(img, self.new_shape)
    
    def inv(self, img):
        return resize(img, self.shape)
    
@torch.no_grad()
def predict_tta(model, img, transforms, type="probs"):
    """Apply the model to an image using test time augmentation.

    Parameters
    ----------
    model
        The model to use for prediction
    img
        The image to predict
    transforms
        List of TTATransform objects
    type
        The values used for TTA averaging. Options are 'probs' or 'logits'

    Returns:
    -------
    avg_scores
        The predicted scores averaged over the transformations
    """

    if type not in ("probs", "logits"):
        raise ValueError('type must be one of "probs" or "logits"')

    avg_scores = 0.
    for t in transforms:
        img_t = t(img)
        scores_t = model(img_t.unsqueeze(0))[0]
        scores_t = t.inv(scores_t)

        if type=="probs":
            scores_t = scores_t.softmax(0)

        avg_scores += scores_t

    avg_scores /= len(transforms)

    if type=="probs":
        # Invert the softmax in order to have scores instead of probabilities
        avg_scores = avg_scores.log() - avg_scores[0].log()

    return avg_scores

@torch.no_grad()
def find_optimal_threshold(model, ds, ignore_index=None, device="cuda") -> float:
    """Find the optimal threshold to apply to model predictions so as to maximize
    the Dice score.

    Parameters
    ----------
    model
        The model to use for prediction
    ds
        The dataset to use
    ignore_index
        The index to ignore on the target when computing the metric
    device
        The device to use for prediction
    """

    model.eval()
    model.to(device)

    all_scores = []
    all_targets = []
    for img, target in ds:
        img = img.to(device)

        scores = model(img.unsqueeze(0))[0].cpu()
        
        all_scores.append(scores)  # (C, N)
        all_targets.append(target)
        
    all_scores = torch.stack(all_scores, dim=0)
    all_targets = torch.stack(all_targets, dim=0)

    perf_values = []
    thresholds = torch.linspace(0, 1, 256)
    pbar = tqdm(
        thresholds,
        desc="Finding threshold",
        unit="thresholds",
        leave=False,
        dynamic_ncols=True,
    ) 
    for threshold in pbar:
        perf = ConfusionMatrixMetrics(threshold, ignore_index=ignore_index)(all_scores, all_targets)
        # perf[4] is the Dice score
        perf_values.append(perf[4])

    perf_values = torch.tensor(perf_values)
    optimal_threshold = thresholds[perf_values.argmax()] 

    return optimal_threshold.item()

