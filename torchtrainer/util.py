import inspect
import torch

def save_params(store):
    """Annotator for saving function parameters."""

    def func_caller(wrapped_func):
        signature = inspect.signature(wrapped_func)
        params = signature.parameters
        for param_name in params:
            param = params[param_name]
            store[param_name] = param.default
            
        def func(*args, **kwargs):
            for idx, param_name in enumerate(store):
                if idx<len(args):
                    store[param_name] = args[idx]
                else:
                    if param_name in kwargs:
                        store[param_name] = kwargs[param_name]
                
            wrapped_func(*args, **kwargs)
            
        return func
    return func_caller

class MemoryMonitor:
    
    def __init__(self, device):
        
        self.device = device
        #self.max_memory = None
        #self.curr_memory = None
    
    def start(self):
        torch.cuda.reset_peak_memory_stats(device=self.device)
        
    def max_memory_used(self):
        return torch.cuda.max_memory_allocated(device=self.device)/1024**3
    
    def curr_memory(self):
        return torch.cuda.memory_allocated(device=self.device)/1024**3
    
    def reset(self):
        #self.max_memory = None
        #self.curr_memory = None
        self.start()

def count_parameters(model):
    
    num_parameters = 0
    for pars in model.parameters():
        num_parameters += torch.prod(torch.tensor(pars.shape))
        
    return num_parameters