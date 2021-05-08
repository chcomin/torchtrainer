'''Utility functions and classes for working with Pytorch modules'''

from functools import partial
from torch import nn
from collections import OrderedDict

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

class ActivationSampler(nn.Module):
    '''Generates a hook for sampling a layer activation. Can be used as

    sampler = ActivationSampler(layer_in_model)
    output = model(input)
    layer_activation = sampler()

    '''

    def __init__(self, model):
        super(ActivationSampler, self).__init__()
        self.model_name = model.__class__.__name__
        self.activation = None
        model.register_forward_hook(self.get_hook())

    def forward(self, x=None):
        return self.activation

    def get_hook(self):
        def hook(model, input, output):
            self.activation = output.detach()
        return hook

    def extra_repr(self):
        return f'{self.model_name}'

class Lambda(nn.Module):
    '''Transforms function into a module'''

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

class Hooks:
    '''Hooks for storing information about layers.

    The attribute `storage` will contain the layers information. It is a dict
    having layer names as keys and respective values generated by `func`.

    Parameters
    ----------
    module : torch.nn
        The module containing the layers. Only used for getting layer names
    layers : list
        List of torch.nn modules for storing the activations
    func : function
        Function to be registered as a hook. Must have signature func(storage, module, input, output) for
        forward hooks and ?? for backward hooks. `storage` is a dictionary used for storing layer information.
    '''

    def __init__(self, module, layers, func, is_forward=True):

        self.hooks = []

        storage = {}         # For storing information, each layer will be a key here
        layers_dict = {}     # Dict of layer names and actual layers
        # Obtain layer names for hashing. Is there a better way?
        for layer_name, layer in module.named_modules():
            if True in [True for l in layers if layer is l]:
                layers_dict[layer_name] = layer
                storage[layer_name] = {}

        self.layers_dict = layers_dict
        self.storage = storage

        if is_forward:
            self._register_forward_hooks(func)
        else:
            self._register_backward_hooks(func)

    def __del__(self): self.remove_hooks()

    def _register_forward_hooks(self, func):
        '''Register one hook for each layer.'''

        for layer_name, layer in self.layers_dict.items():
            hook_func = self._generate_hook(func, self.storage[layer_name])
            self.hooks.append(layer.register_forward_hook(hook_func))

    def _register_backward_hooks(self, func):
        '''Register one hook for each layer.'''

        for layer_name, layer in self.layers_dict.items():
            hook_func = self._generate_hook(func, self.storage[layer_name])
            self.hooks.append(layer.register_backward_hook(hook_func))

    def _generate_hook(self, func, storage):
        '''Generate function to be used in module.register_forward_hook and module.register_backward_hook, fixing
        as a first argument to the function an empty dictionary.'''

        return partial(func, storage)

    def to_cpu(self):
        pass

    def remove_hooks(self):
        '''Remove hooks from the network.'''

        for hook in self.hooks:
            hook.remove()

def _calculate_stats(storage, model, input, output, store_act=True, store_weights=False):

    if store_act:
        if 'activation' not in storage:
            storage['activation'] = {}
        if 'mean' not in storage['activation']:
            storage['activation']['mean'] = []
        if 'std' not in storage['activation']:
            storage['activation']['std'] = []
        if 'hist' not in storage['activation']:
            storage['activation']['hist'] = []

        activation = output.detach()
        storage['activation']['mean'].append(activation.mean().item())
        storage['activation']['std'].append(activation.std().item())
        storage['activation']['hist'].append(activation.cpu().histc(100,-10,10)) #histc isn't implemented on the GPU

    if store_weights:
        if 'weights' not in storage:
            storage['weights'] = {}
        if 'mean' not in storage['weights']:
            storage['weights']['mean'] = []
        if 'std' not in storage['weights']:
            storage['weights']['std'] = []

        try:
            weight = model.weight
        except Exception:
            raise AttributeError('Model does not have `weight` attribute')
        else:
            weight = weight.detach()
            storage['weights']['mean'].append(weight.mean().item())
            storage['weights']['std'].append(weight.std().item())
            storage['weights']['hist'].append(weight.cpu().histc(100,-10,10)) #histc isn't implemented on the GPU

def calculate_stats(store_act=True, store_weights=False):

    return partial(_calculate_stats, store_act=store_act, store_weights=store_weights)

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

def get_output_shape(self, model, img_shape):

    input_img = torch.zeros(img_shape)[None, None]
    input_img = input_img.to(next(model.parameters()).device)
    output = model(input_img)
    return output[0, 0].shape

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