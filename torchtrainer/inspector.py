import torch
import torch.nn as nn
from collections.abc import Callable
from typing import Union, Optional

class Inspector:
    """Inspector class for capturing modules' parameters, gradients, activations and activation gradients."""
    
    def __init__(self, model: nn.Module, modules_to_track: Optional[list[nn.Module]] = None, agg_func: Optional[Callable] = None) -> None:
        """Inspector class for capturing modules' parameters, gradients, activations and activation gradients.

        If agg_func is provided, this function will be applied to the tracked data on the same devide where the model
        resides. This is useful for aggregating the data before copying from gpu to cpu to avoid an expensive copy. 
        One can, for instance, return only the average value of the data. This function must have signature 
        agg_func(data, module_name, data_type, param_name=None), where `data` is a tensor, `module_name` is a string 
        containing the name of a layer, `data_type` can be 'param' (layer parameter), 'grad' (layer parameter gradient), 
        'act' (layer activation), 'act_grad' (layer activation gradient) and param_name contains the name of the parameter 
        in case of 'param' and 'grad'.

        Args:
            model: the model to inspect.
            modules_to_track: if None, track all layers of the model. If list, track the specified layers. the layers
            are specified as, for instance, [model.layer1, model.layer2.conv1, ...]
            agg_func: function used to process the tracked data. See above for an explanation.

        Example:
            Getting all the data about a model:

            inspector = Inspector(model)
            inspector.start_tracking_activations()
            inspector.start_tracking_act_grads()
            model(input).sum().backward()
            params = inspector.get_params()
            grads = inspector.get_grads()
            activations = inspector.get_activations()
            act_grads = inspector.get_act_grads()

            Getting gradient averages:

            def mean(data, module_name, data_type, param_name):
                return data.mean()

            inspector = Inspector(model)
            model(input).sum().backward()
            grad_avgs = inspector.get_grads()
        """
        
        if modules_to_track is None:
            modules = list(model.modules())
            modules = [module for module in modules if not isinstance(module, nn.ModuleList)]
            modules_to_track = modules

        if agg_func is None:
            agg_func = lambda data, module_name, data_type, param_name=None: data
                       
        self.modules_to_track = modules_to_track
        self.agg_func = agg_func
        
        self.model = model
        self.model_name = model._get_name().lower()
        self.dict_module_to_str = self._create_module_dict()
        self.dict_model_stats = self._create_stats_dict()       
        self.forward_hook = None
        self.backward_hook = None
        self.act_hook_handles = []
        self.act_grad_hook_handles = []
        self.tracking_activations = False
        self.tracking_act_grads = False
        
        self._create_hooks()

    def get_params(self, out: Optional[dict] = None) -> dict:
        """Get model parameters. The returned dictionary has the format 
        {layer_name: {param_name: data,...},...}.
        
        Args:
            out: a dictionary of a previous call to this method can be provided, in which case
            the data is copied to it. This helps avoid an additional copy of the data.
        """
        return self._capture_params_grads(out, 'param')
        
    def get_grads(self, out: Optional[dict] = None) -> dict:
        """Get model parameter gradients. The returned dictionary has the format 
        {layer_name: {param_name: data,...},...}.
        
        Args:
            out: a dictionary of a previous call to this method can be provided, in which case
            the data is copied to it. This helps avoid an additional copy of the data.
        """

        return self._capture_params_grads(out, 'grad')

    def get_activations(self) -> dict:
        """Get model activations. The returned dictionary has the format 
        {layer_name: data,...}.
        """
        
        return self._get_dict_data('act')

    def get_act_grads(self) -> dict:
        """Get the gradient of model activations. The returned dictionary has the format 
        {layer_name: data,...}.
        """

        return self._get_dict_data('act_grad')

    def start_tracking_activations(self) -> None:
        '''Start tracking model activations. Warning, tracking activations leads to slower model execution.`
        When a call to model.forward() is executed, the activations are saved internally and can be retrieved
         using the `get_activations()` method.'''

        if self.tracking_activations:
            print('Already tracking activations')
        else:
            self.tracking_activations = True
            self._add_activation_hooks()

    def start_tracking_act_grads(self) -> None:
        '''Start tracking the gradients of the activations. Warning, tracking activation gradients leads to slower model execution.
        When a call to model.forward() is executed, the gradients are saved internally and can be retrieved
        using the `get_act_grads()` method.

        Note: this method does not work for in-place operations.
        '''

        if self.tracking_act_grads:
            print('Already tracking activation gradients')
        else:
            self.tracking_act_grads = True
            self._add_act_grad_hooks()
          
    def stop_tracking_activations(self) -> None:
        '''Stop tracking model data.'''

        for hook in self.act_hook_handles:
            hook.remove()

    def stop_tracking_act_grads(self) -> None:
        '''Stop tracking model data.'''

        for hook in self.act_grad_hook_handles:
            hook.remove()
        
    def _create_module_dict(self) -> dict:
        '''Create dictionary (module instance):(name str) for tracked modules.'''
        
        model = self.model
        model_name = self.model_name
        modules_to_track = self.modules_to_track
        
        def create_key(module):
            if module in modules_to_track:
                dict_module_to_str[module] = None
        dict_module_to_str = {}
        # Recursively applies function to all modules. 
        # Important! model.apply() traverses the model in the exact same order as 
        # the calls to the forward function of each layer. The same is not true
        # for model.modules()
        model.apply(create_key)
        if model in modules_to_track:
            # Add one key for the whole model
            dict_module_to_str[model] = model_name
        
        for name, module in model.named_modules(prefix=model_name):
            if name=='':
                name = model_name
            if module in dict_module_to_str:
                dict_module_to_str[module] = name
        
        return dict_module_to_str
    
    def _create_stats_dict(self) -> dict:
        """Create dictionary for saving activation data."""
        
        dict_model_stats = {}
        for module, module_name in self.dict_module_to_str.items():
            params = list(module.named_parameters(recurse=False))
            dict_model_stats[module_name] = {
                'act':None, 
                'act_grad':None
            }
            
        return dict_model_stats
    
    def _create_hooks(self) -> None:
        """Create hooks for tracking activations."""

        def forward_hook(module, inputs, output):
            name = self.dict_module_to_str[module]
            #Module activations
            self._add_to_dict(output.detach(), name, 'act')

        def backward_hook(module, grad_input, grad_output):
            name = self.dict_module_to_str[module]
            self._add_to_dict(list(grad_output), name, 'act_grad')
          
        self.forward_hook = forward_hook
        self.backward_hook = backward_hook

    def _add_activation_hooks(self) -> None:
        """Add hooks to modules."""

        act_hook_handles = []
        for module, module_name in self.dict_module_to_str.items():
            # Add one hook for each module
            handler_for = module.register_forward_hook(self.forward_hook)
            act_hook_handles.append(handler_for)

        self.act_hook_handles = act_hook_handles

    def _add_act_grad_hooks(self) -> None:
        '''Add hooks to modules.'''

        act_grad_hook_handles = []
        for module, module_name in self.dict_module_to_str.items():
            handler_back = module.register_full_backward_hook(self.backward_hook)
            act_grad_hook_handles.append(handler_back)
        
        self.act_grad_hook_handles = act_grad_hook_handles

    def _capture_params_grads(self, out: Optional[dict] = None, which: str = 'param') -> dict:
        """Get model parameters or gradients. `which` can be 'param' or 'grad'."""

        if out is None:
            out = {}
            create_new = True
        else:
            create_new = False

        for module, module_name in self.dict_module_to_str.items():
            if create_new and len(list(module.named_parameters(recurse=False)))>0:
                out[module_name] = {}
            for param_name, param in module.named_parameters(recurse=False):
                if which=='param':
                    param = param.detach()
                elif which=='grad':
                    param = param.grad
                res = self.agg_func(param, module_name, which, param_name)
                if create_new:
                    out[module_name][param_name] = res.to('cpu', copy=True)
                else:
                    out[module_name][param_name].copy_(res)

        return out  
                
    def _get_dict_data(self, key: str) -> dict:
        """Get activations or activation gradients. `key` can be 'act' or 'act_grad'."""

        out_dict = {}
        for module_name, module_data in self.dict_model_stats.items():
            out_dict[module_name] = module_data[key]

        return out_dict
           
    def _add_to_dict(self, data: torch.Tensor, module_name: str, data_type: str) -> None:
        '''Receives activation data and save to dictionary in the cpu. If the 
        model is being trained on the GPU, `data` resides in the GPU.
        `data_type can be 'act' or 'act_grad'.'''
        
        dict_model_stats = self.dict_model_stats
        saved_data = dict_model_stats[module_name][data_type]
        create_new = saved_data is None    
        # saved_data can be None, a tensor or a list
        # data can be a tensor or a list     
        if data_type=='act':
            data = [data]
            saved_data = [saved_data]
        if create_new:
            saved_data = [None]*len(data)
        #Now, saved_data and data are lists with the same number of elements
        #saved_data may contain references to tensors, or None
        
        for idx, (tensor, storage) in enumerate(zip(data, saved_data)):
            res = self.agg_func(tensor, module_name, data_type)
            if create_new:
                # Copy if already on cpu
                saved_data[idx] = res.to('cpu', copy=True)
            else:
                storage.copy_(res)
        
        if create_new:
            if data_type=='act':
                # Recover format of Pytorch hooks
                saved_data = saved_data[0]

            dict_model_stats[module_name][data_type] = saved_data

def agg_func_stas(data, module_name, data_type, param_name=None):
    """Example aggregator function for storing some statistics about the model."""
    
    mean = data.mean()
    min = data.min()
    max = data.max()
    if min==max:
        std = 0
    else:
        std = data.std()
    res = torch.tensor([mean, std, min, max])
    
    return res

def flatten_data(data: dict) -> tuple[torch.Tensor, torch.Tensor]:

    num_el = 0
    for module_name, module_data in data.items():
        if isinstance(module_data, torch.Tensor):
            num_el += module_data.numel()
        else:
            for param_name, param_data in module_data.items():
                num_el += param_data.numel()

    flattened_data = torch.zeros(num_el)
    idx = 0
    for module_name, module_data in data.items():
        if isinstance(module_data, torch.Tensor):
            num_el = module_data.numel()
            flattened_data[idx:idx+num_el] = module_data.view(-1)
        else:
            for param_name, param_data in module_data.items():
                num_el = param_data.numel()
                flattened_data[idx:idx+num_el] = param_data.view(-1)  

    return flattened_data

if __name__=='__main__':

    import test_models
    import profiling


    model = test_models.SimpleConv(num_layers=4, input_channels=1, num_channels_first=16, channel_factor=2, kernel_size=3)
    model.to('cuda')

    x = torch.full((1, 1, 256, 256), 5., device='cuda')

    def call_model(model, x):
        model(x).sum().backward()

    inspector = Inspector(model)
    #inspector.start_tracking_activations()
    #inspector.start_tracking_act_grads()
    stats = profiling.benchmark_function(call_model, (model, x), profile=False)
    params = inspector.get_params()
    #grads = inspector.get_grads()
    #activations = inspector.get_activations()
    #act_grads = inspector.get_act_grads()
    print(params)
    #print(grads)
    #print(activations)
    #print(act_grads)
    #print(stats)

    """def mean(data, module_name, data_type, param_name):
        return data.mean()

    inspector = Inspector(model, agg_func=mean)
    model(x).sum().backward()
    grad_avgs = inspector.get_grads()
    print(grad_avgs)"""
  
