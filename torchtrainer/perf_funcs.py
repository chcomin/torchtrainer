'''Functions for measuring the performance of a classifier.'''

from functools import partial
import scipy.ndimage as ndi
import numpy as np
import torch
import torch.nn.functional as F

def iou(input, target, ignore_index=None):
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

def segmentation_accuracy(input, target, meas='iou', reduce_batch=True, mask=None):
    '''Calculate some performance metrics for segmentation results. Assumes background has value 0 and
    the segmentation has value 1. If more than one image in batch, returns a single value representing the
    average performance for all images in the batch.

    Possible measurements are:
        prec : precision
        rec : recall
        f1 : f1 score
        iou : intersection over union

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

def build_segmentation_accuracy_dict():
    '''Build dictionary containing performance metrics from `segmentation_accuracy`

    Returns
    -------
    metrics_dict : dict
        Keys indicate metric name and values are respecive functions.
    '''

    metrics_dict = {}
    for mea in ['iou', 'f1', 'prec', 'rec']:
        metrics_dict[mea] = partial(segmentation_accuracy, meas=mea)
    return metrics_dict

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

class FocalLoss(torch.nn.Module):
    
    def __init__(self, gamma=2., alpha=(1., 1.), reduction='mean'):
        super().__init__()
        
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, input, target):
        
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction)
        
class FocalLossRaw(FocalLoss):
    
    def __init__(self, gamma=2., alpha=(1., 1.), reduction='mean', eps=1e-8):
        """Same as `FocalLoss` but receives raw scores and applies a softmax function
        to obtain probabilities."""

        super().__init__(gamma, alpha, reduction)
        
        self.eps = eps
                
    def forward(self, input, target):
        
        probs = F.softmax(input, dim=1) + self.eps
        return super().forward(probs, target)
    
def focal_loss(probs, target, alpha, gamma, reduction='mean'):
    
    pt = torch.gather(probs, 1, target.unsqueeze(1)).squeeze()
    alpha = alpha[target]        # Broadcast alpha values
    
    loss_all = -alpha*(1-pt)**gamma*pt.log()   
    if reduction == 'none':
        loss = loss_all
    elif reduction == 'mean':
        loss = torch.mean(loss_all)
    elif reduction == 'sum':
        loss = torch.sum(loss_all)
        
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

class CocoPerf:
    """Not the same as COCO, since detection scores are not considered."""
    
    def __init__(self, iou_thrs=None, which='precision'):
        
        if iou_thrs is None:
            iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        elif not isinstance(iou_thrs, (list, tuple)):
            iou_thrs = [iou_thrs]
        if isinstance(which, str):
            which = [which]
            
        self.iou_thrs = iou_thrs
        self.which = which
        
    def __call__(self, img_det, img_gt):
        
        results = self.evaluate(img_det, img_gt)

        return results[self.which[0]][0]

    def compute_ious(self, instances_det, instances_gt):

        ious = np.zeros((len(instances_det), len(instances_gt)))
        for idx_det, instance_det in enumerate(instances_det):
            for idx_gt, instance_gt in enumerate(instances_gt):
                points_det = instance_det.points
                points_gt = instance_gt.points
                iou = len(points_gt & points_det) / len(points_gt | points_det)
                ious[idx_det, idx_gt] = iou

        return ious

    def evaluate(self, img_det, img_gt):
                
        iou_thrs = self.iou_thrs
        num_thrs = len(iou_thrs)
        
        instances_gt = get_instances_coords(img_gt)
        instances_det = get_instances_coords(img_det)
        num_gt = len(instances_gt)
        num_det = len(instances_det)
            
        ious = self.compute_ious(instances_det, instances_gt)
        best_matches_det, best_matches_gt = self.match_instances(instances_det, instances_gt, ious, iou_thrs)
        matches_det, matches_gt = self.threshold_matches(best_matches_det, ious)
        metrics = self.metrics(matches_det, matches_gt, ious)

        
        results = {'instances_det': instances_det, 'instances_gt': instances_gt,
                   'ious': ious, 'matches_gt': matches_gt, 'matches_det': matches_det,
                   **metrics}
         
        return results

    def threshold_matches(self, best_matches_det, ious):
        
        iou_thrs = self.iou_thrs
        num_thrs = len(iou_thrs)
        num_det = ious.shape[0]
        num_gt = ious.shape[1]
        
        matches_gt  = np.full((num_thrs, num_gt), -1)
        matches_det  = np.full((num_thrs, num_det), -1)
        for idx_thrs, thrs in enumerate(iou_thrs):
            for idx_det in range(num_det):      
                idx_gt = best_matches_det[idx_det]
                if ious[idx_det, idx_gt] >= thrs:
                    matches_det[idx_thrs, idx_det]  = idx_gt
                    matches_gt[idx_thrs, idx_gt] = idx_det
                    
        return matches_det, matches_gt
    
    def metrics(self, matches_det, matches_gt, ious):
        
        iou_thrs = self.iou_thrs
        num_thrs = len(iou_thrs)
        
        tp_mask = matches_gt>-1
        fp_mask = matches_det==-1
        fn_mask = matches_gt==-1
        
        tp = np.sum(tp_mask, axis=1)
        fp = np.sum(fp_mask, axis=1)
        fn = np.sum(fn_mask, axis=1)

        precision = tp/(tp + fp)
        recall = tp/(tp + fn)

        sum_ious_tp = np.zeros(num_thrs)
        for idx_thrs in range(num_thrs):
            indices_gt = matches_det[idx_thrs]
            ious_tp = []
            for idx_det, idx_gt in enumerate(indices_gt):
                if idx_gt>-1:
                    ious_tp.append(ious[idx_det, idx_gt])
            sum_ious_tp[idx_thrs] = sum(ious_tp)
        panoptic = sum_ious_tp/(tp + 0.5*fp + 0.5*fn)
            
        return {'precision': precision, 'recall': recall, 'panoptic': panoptic}
        
    def match_instances(self, instances_det, instances_gt, ious, iou_thrs):
        
        num_det = len(instances_det)
        num_gt = len(instances_gt)
        num_thrs = len(iou_thrs)

        # Bipartite graph with weighted edges
        edges = []
        for idx_det in range(num_det): 
            for idx_gt in range(num_gt): 
                edges.append((idx_det, num_det+idx_gt, ious[idx_det, idx_gt]))

        current_det_nodes = set(range(num_det))
        current_gt_nodes = set(range(num_det, num_det+num_gt))
        sorted_edges = sorted(edges, key=lambda edge: edge[2], reverse=True)
        best_matches_gt  = np.full(num_gt, -1)
        best_matches_det  = np.full(num_det, -1)
        # Iterate over edges in decreasing iou order, removing the nodes from the graph
        for idx_det, idx_gt, iou in sorted_edges:
            if iou>0:
                if idx_gt in current_gt_nodes and idx_det in current_det_nodes:
                    best_matches_det[idx_det]  = idx_gt - num_det
                    best_matches_gt[idx_gt - num_det] = idx_det    
                    current_gt_nodes.remove(idx_gt)
                    current_det_nodes.remove(idx_det)
                    
        return best_matches_det, best_matches_gt
    
    def draw_results(self, results, img_shape, thresh_idx,
                     tp_color=(0, 255, 0), fp_color=(0, 0, 255), fn_color=(255, 0, 0)):
        
        instances_det = results['instances_det']
        instances_gt = results['instances_gt']
        matches_det = results['matches_det'][thresh_idx]
        matches_gt = results['matches_gt'][thresh_idx]
        
        img_tp = np.zeros((*img_shape, 3), dtype=np.int)
        img_fp = np.zeros_like(img_tp)
        img_fn = np.zeros_like(img_tp)
        img_gt = np.zeros_like(img_tp)
        for idx_det, idx_gt in enumerate(matches_det):
            if idx_gt==-1:
                self.draw_instance(img_fp, instances_det[idx_det], fp_color)
            else:
                self.draw_instance(img_tp, instances_det[idx_det], tp_color)
        for idx_gt, idx_det in enumerate(matches_gt):
            if idx_det==-1:
                self.draw_instance(img_fn, instances_gt[idx_gt], fn_color)
                
        for instance in instances_gt:
            self.draw_instance(img_gt, instance, (255, 255, 255))
            
        gt_alpha = 0.2
        img_final = (img_tp + img_fp + img_fn + (1-gt_alpha)*img_gt)/(2+gt_alpha)
        img_final = np.round(img_final).astype(np.uint8)
            
        return img_final
      
    def draw_instance(self, img, instance, color):
        
        num_cols = img.shape[1]
        for index in instance.points:
            row = index//num_cols
            col = index - row*num_cols
            img[row, col] = color
    
class Instance:
    
    def __init__(self, id, points):
        
        if not isinstance(points, set):
            points = set(map(tuple, points))
            
        self.id = id
        self.points = points
        
    def center_of_mass(self):
        
        return np.mean(self.points, axis=0)
    
def get_instances_coords(img):

    se = np.ones((3, 3))
    img_label, num_inst = ndi.label(img, se)

    indices = np.arange(1, num_inst + 1)
    inst_points = ndi.labeled_comprehension(img_label, img_label, indices, 
                                             lambda v, p: (v[0], p), list, None, 
                                             pass_positions=True)
    inst_points = [Instance(v[0]-1, set(v[1])) for v in inst_points]
    return inst_points
