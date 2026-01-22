"""Functions for dealing with the output of a network"""
import torch


def logits_to_preds(
        scores: torch.Tensor,
        task: str = "binary", 
        return_indices: bool = False, 
        threshold: float = 0.5,
        keepdim: bool = True
        ):
    """Transform logits/scores to probabilities or, optionally, to class predictions.
    The function supports binary, multiclass, and multilabel classification.

    Parameters
    ----------
    scores
        The logits/scores output by the network. The tensor must have at least 2 dimensions.
    task
        The type of classification task. Must be one of 'binary', 'multiclass', or 'multilabel'.
    return_indices
        If False, returns class probabilities. If True, returns the predicted class indices.
        For binary and multilabel classification, the indices are calculated by thresholding the 
        probabilities. For multiclass classification, the indices are given by the argmax of the 
        probabilities along the class dimension.
    threshold
        The threshold to apply to the predicted probabilities.
    keepdim
        If True, keeps the class dimension when returning the predicted probabilities/indices. 
        If False, removes the class dimension.

    Returns:
    -------
    preds
        The predicted probabilities or class indices.
    """

    if task not in ("binary", "multiclass", "multilabel"):
        raise ValueError("task must be one of 'binary', 'multiclass', or 'multilabel'.")

    if scores.ndim==1:
        raise ValueError("Scores must have at least the batch and class dimensions.")

    num_classes = scores.shape[1]
    if task=="binary" and num_classes>2:
        raise ValueError("Binary classification is only supported for 2 classes.")
    if task in ("multiclass", "multilabel") and num_classes==1:
        raise ValueError(f"{task} classification is only supported for more than 1 class.")
    if not keepdim and task=="multilabel":
        raise ValueError("keepdim cannot be False for multilabel classification.")

    if task=="binary":
        if num_classes==1:
            probs = scores.sigmoid()
        elif num_classes==2:
            # Remove class 0 index but keep class dimension
            probs = scores.softmax(dim=1)[:, 1:]
    elif task=="multiclass":
        probs = scores.softmax(dim=1)
    elif task=="multilabel":
        probs = scores.sigmoid()

    preds = probs
    if return_indices:
        if task in ("binary", "multilabel"):
            preds = preds > threshold  
        elif task=="multiclass":
            preds = torch.argmax(probs, dim=1, keepdim=True)

    if not keepdim and task in ("binary", "multiclass"):
        preds = preds.squeeze(dim=1)
        
    return preds