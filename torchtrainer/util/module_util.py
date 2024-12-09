'''Utility functions and classes for working with Pytorch modules'''

from torch import nn
from collections import OrderedDict
import torch
import copy

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

class ActivationSampler(nn.Module):
    '''Generates a hook for sampling a layer activation. Can be used as

    sampler = ActivationSampler(layer_in_model)
    output = model(input)
    layer_activation = sampler()

    '''

    def __init__(self, model):
        super().__init__()
        self.model_name = model.__class__.__name__
        self.activation = None
        self.hook_handler = model.register_forward_hook(self.get_hook())

    def forward(self, x=None):
        return self.activation

    def get_hook(self):
        def hook(model, input, output):
            self.activation = output.detach().cpu()
        return hook

    def extra_repr(self):
        return f'{self.model_name}'
    
    def __del__(self): self.hook_handler.remove()

class Lambda(nn.Module):
    '''Transforms function into a module'''

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

class Hook:
    """Hook for capturing module activations.

    Args:
        module (torch.nn.Module): Pytorch module
        detach (bool, optional): If True, detaches the activation from the computation graph. Defaults to True.
    """
    def __init__(self, module, detach=True):
        self.detach = detach
        self.activation = None
        self.handler = module.register_forward_hook(self._create_hook())

    def _create_hook(self):
        def forward_hook(module, inputs, output):
            if self.detach:
                output = output.detach()
            self.activation = output

        return forward_hook

    def remove(self):
        self.handler.remove()

    def __del__(self): self.remove()

class ReceptiveField:
    '''Calculate the receptive field of a pixel in the activation map of a neural network layer. Example
    usage:

    receptive_field = ReceptiveField(model)
    rf = receptive_field.receptive_field(module_name)
    
    *The class creates a copy of the model, which uses additional memory.

    Parameters
    ----------
    model: torch.nn.Module
        The model
    device: str
        The device to use.

    Returns
    -------
    rf: torch.tensor
        An image containing the receptive field.
    '''

    conv_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    conv_transp_layers = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    norm_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    linear_layers = (nn.Linear,)

    def __init__(self, model, device='cuda'):

        self.device = device
        model = copy.deepcopy(model)
        self.model = model
        self.prepare_model(model)

    def prepare_model(self, model):

        model.to(self.device)
        model.eval()

        # Get names of modules containing parameters or buffers
        '''modules_with_params = set()
        for name, _ in model.named_parameters():
            mod_name = '.'.join(name.split('.')[:-1])
            modules_with_params.add(mod_name)

        for name, _ in model.named_buffers():
            mod_name = '.'.join(name.split('.')[:-1])
            modules_with_params.add(mod_name)'''
        
        '''with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, self.conv_layers):
                    # Set filters to 1/num_vals_filter. 
                    n = module.weight[0].numel()
                    module.weight[:] = 1./n
                    if module.bias is not None:
                        module.bias[:] = 0.
                #if isinstance(module, nn.ReLU):
                #    module.inplace = False
                elif isinstance(module, self.conv_transp_layers):
                    shape = module.weight
                    stride = module.stride
                    # Effective number of input features is ks//stride for each spatial dimension, times the number of input channels
                    n = shape[0]*int(torch.prod(torch.tensor(shape[2:])//torch.tensor(stride)))
                    module.weight[:] = 1./n
                    if module.bias is not None:
                        module.bias[:] = 0.'''

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, self.conv_layers):
                    # Set filters to 1/num_vals_filter. Assumes filters have the same
                    # size in all dimensions
                    n = module.weight[0].numel()
                    module.weight[:] = 1./n
                    if module.bias is not None:
                        module.bias[:] = 0.
                #if isinstance(module, nn.ReLU):
                #    module.inplace = False
                elif isinstance(module, self.conv_transp_layers):
                    ks = module.kernel_size[0]
                    stride = module.stride[0]
                    dim = len(module.kernel_size)
                    # Effective number of input features is ks//stride for each dimension
                    n = (ks//stride)**dim
                    module.weight[:] = 1./n
                    if module.bias is not None:
                        module.bias[:] = 0.
                elif isinstance(module, self.norm_layers):
                    # Disable batchnorm
                    module.training = False
                    module.weight[:] = 1.
                    module.bias[:] = 0.
                    module.running_mean[:] = 0.
                    module.running_var[:] = 1.
                elif isinstance(module, self.linear_layers):
                    # Number of input features
                    n = module.weight.shape[1]
                    module.weight[:] = 1./n
                elif len(list(module.parameters(recurse=False)))>0:
                    # Layer has parameter but is not one of the modules above
                    print(f'Warning, module {name} was not recognized.')

    def receptive_field(self, module_name, num_channels=1, img_size=(512, 512), pixel=None):
        '''Calculate the receptive field of a pixel in the activation map of a neural network layer.
        
        Parameters
        ----------
        module_name: torch.nn.Module
            A module in `model`. The activation map of this module will be used.
        num_channels: int
            The number of channels of the input
        img_size: tuple
            Image size to use as input
        pixel: (int, int)
            Which activation pixel to use. If not provided, uses pixel at the center.

        Returns
        -------
        rf: torch.tensor
            An image containing the receptive field.
        '''

        model = self.model
        module = model.get_submodule(module_name)
                
        # Attach hook to get activation
        hook = Hook(module, False)

        x = torch.ones(1, num_channels, img_size[0], img_size[1], requires_grad=True, device=self.device)
        res = model(x)

        act = hook.activation
        if pixel is None:
            # Get pixel at the middle of the activation map
            size = act.shape[-2:]
            pixel = (size[0]//2, size[1]//2)
        pix_val = act[0, 0, pixel[0], pixel[1]]
        # Calculate gradients
        pix_val.backward()

        # Gradient with respect to the input
        rf = x.grad[0,0]

        rf = rf/rf.max()
        rf = rf.to("cpu")

        hook.remove()

        return rf

    def receptive_field_bbox(self, module_name, num_channels=1, img_size=(512, 512), pixel=None, eps=1e-8):
        """Returns the bounding box and center of the receptive field."""

        rf = self.receptive_field(module_name, num_channels, img_size, pixel)
        inds = torch.nonzero(rf>=eps)
        r0, c0 = inds.amin(dim=0)
        r1, c1 = inds.amax(dim=0)
        r0, c0, r1, c1 = r0.item(), c0.item(), r1.item(), c1.item()
        bbox = (r0, c0, r1, c1)

        # Calculate position of the maximum
        inds = torch.nonzero(rf==rf.max())
        r, c = inds.float().mean(axis=0)
        center = (int(r.item()), int(c.item()))

        # Check if bbox and center make sense
        #if bbox[0]>0 and bbox[1]>0 and bbox[2]<img_size[0]-1 and bbox[3]<img_size[1]-1:
        #    bbox_center = bbox[0]+(bbox[2]-bbox[0])//2, bbox[1]+(bbox[3]-bbox[1])//2
        ##    if bbox_center!=center:
        #    if abs(center[0]-bbox_center[0])>1 or abs(center[1]-bbox_center[1])>1:
        #        print(f'Warning, the position of the maximum of the receptive field ({center}) is not equal to its center ({bbox_center}).')
        #        print(bbox, inds)

        return bbox, center
    
    def receptive_field_bf(self, module, num_channels=1, img_size=(512, 512)):
        '''Calculate receptive field using brute force.'''

        model = self.model
        self.prepare_model(model)
                
        # Attach hook to get activation
        hook = Hook(module)

        with torch.no_grad():
            x = torch.zeros(1, num_channels, img_size[0], img_size[0])
            res = model(x)
            act = hook.activation
            size = act.shape[-2:]
            pixel = (size[0]//2, size[1]//2)
            pix_val_0 = act[0, 0, pixel[0], pixel[1]]
            pix_val_diffs = []
            for i in range(0, img_size[1]):
                x[0, 0, img_size[0]//2, i] = 1
                _ = model(x)
                x[0, 0, img_size[0]//2, i] = 0

                act = hook.activation
                pix_val = act[0, 0, pixel[0], pixel[1]]
                pix_val_diffs.append(pix_val-pix_val_0)

        rf = torch.tensor(pix_val_diffs)
        rf = rf/rf.max()
        
        return rf

class FeatureExtractor:
    """Capture activations of intermediate layers of a model.

    Args:
        model (torch.nn.Module): model to analyse.
        module_names (List[str]): list of module layers.
    """

    def __init__(self, model, module_names):

        modules = []
        for name in module_names:
            modules.append(model.get_submodule(name))

        self.model = model
        self.module_names = module_names
        self.modules = modules

    def __call__(self, x):

        hooks = self.attach_hooks(self.modules)
        out = self.model(x)
        acts = self.get_activations(self.module_names, hooks)
        self.remove_hooks(hooks)

        acts['out'] = out

        return acts

    def get_activations(self, module_names, hooks):

        acts = {}
        for name, hook in zip(module_names, hooks):
            acts[name] = hook.activation

        return acts

    def attach_hooks(self, modules):
        '''Attach forward hooks to a list of modules to get activations.'''

        hooks = []
        for module in modules:
            hooks.append(Hook(module))

        return hooks

    def remove_hooks(self, hooks):

        for hook in hooks:
            hook.remove()


def split_modules(model, modules_to_split):
    '''Split `model` layers into different groups. Useful for freezing part of the model
    or using different learning rates.'''

    module_groups = [[]]
    for module in model.modules():
        if module in modules_to_split:
            module_groups.append([])
        module_groups[-1].append(module)
    return module_groups

def define_opt_params(module_groups, lr=None, wd=None, debug=False):
    '''Define distinct learning rate and weight decay for parameters belonging
    to groupd modules in `module_groups`. '''

    num_groups = len(module_groups)
    if isinstance(lr, int): lr = [lr]*num_groups
    if isinstance(wd, int): wd = [wd]*num_groups

    opt_params = []
    for idx, group in enumerate(module_groups):
        group_params = {'params':[]}
        if lr is not None: group_params['lr'] = lr[idx]
        if wd is not None: group_params['wd'] = wd[idx]
        for module in group:
            pars = module.parameters(recurse=False)
            if debug: print(module.__class__)
            pars = list(filter(lambda p: p.requires_grad, pars))
            if len(pars)>0:
                group_params['params'] += pars
                if debug:
                    for p in pars:
                        print(p.shape)
        opt_params.append(group_params)
    return opt_params

def groups_requires_grad(module_groups, req_grad=True, keep_bn=False):
    '''Set requires_grad to `req_grad` for all parameters in `module_groups`.
    If `keep_bn` is True, batchnorm layers are not changed.'''

    for idx, group in enumerate(module_groups):
        for module in group:
            for p in module.parameters(recurse=False):
                if not keep_bn or not isinstance(module, bn_types): p.requires_grad=req_grad

def freeze_to(module_groups, group_idx=-1, keep_bn=False):
    '''Freeze model groups up to the group with index `group_idx`. If `group_idx` is None,
    freezes the entire model. If `keep_bn` is True, batchnorm layers are not changed.'''

    num_groups = len(module_groups)
    slice_freeze = slice(0, group_idx)
    if group_idx is not None:
        slice_unfreeze = slice(group_idx, None)

    groups_requires_grad(module_groups[slice_freeze], False, keep_bn)

    if group_idx is not None:
        groups_requires_grad(module_groups[slice_unfreeze], True)

def unfreeze(module_groups):
    '''Unfreezes the entire model.'''

    groups_requires_grad(module_groups, True)

def get_submodule(model, module):
    """Return a module inside `model`. Module should be a string of the form
    'layer_name.sublayer_name'
    """

    modules_names = module.split('.')
    curr_module = model
    for name in modules_names:
        curr_module = curr_module._modules[name]
    requested_module = curr_module

    return requested_module
    
def get_submodule_str(model, module):
    """Return a string representation of `module` in the form 'layer_name.sublayer_name...'
    """

    for name, curr_module in model.named_modules():
        if curr_module is module:
            module_name = name
            break

    return module_name

def _iterate_modules(father_name, module, module_name, adj_list, modules_dict):
    
    modules_dict[module_name] = module
    for child_module_name, child_module in module.named_children():
        full_child_name = f'{module_name}.{child_module_name}'
        if module_name in adj_list:
            adj_list[module_name].append(full_child_name)
        else:
            adj_list[module_name] = [full_child_name]        
        _iterate_modules(module_name, child_module, full_child_name, adj_list, modules_dict)

def _modules_graph(model):
    """Get hiearchy of modules inside model as an adjacency list"""
    
    adj_list = {}
    modules_dict = {}
    _iterate_modules(None, model, model.__class__.__name__, adj_list, modules_dict)
    
    return adj_list, modules_dict

def model_up_to(model, module):
    """Return a new model with all layers in model up to layer `module`."""
    
    split_module_str = get_submodule_str(model, module)
    split_modules_names = split_module_str.split('.')
    module = model
    splitted_model = []
    name_prefix = ''
    for idx, split_module_name in enumerate(split_modules_names):
        for child_module_name, child_module in module.named_children():
            if child_module_name==split_module_name:
                if idx==len(split_modules_names)-1:
                    # If at last module
                    full_name = f'{name_prefix}{child_module_name}'
                    splitted_model.append((full_name, child_module))
                module = child_module
                name_prefix += split_module_name + '_'
                break
            else:
                full_name = f'{name_prefix}{child_module_name}'
                splitted_model.append((full_name, child_module))

    new_model = torch.nn.Sequential(OrderedDict(splitted_model))
    
    return new_model

