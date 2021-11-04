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

class WeightedAverage:
    
    def __init__(self, momentum=0.9, debias=True):
        
        self.momentum = momentum
        self.debias = debias
        self.count = 0
        self.weighted_average = 0
        
    def add(self, new_value):
        
        self.weighted_average = self.weighted_average*self.momentum + (1-self.momentum)*new_value
        self.count += 1
        
    def value(self):
        
        value = self.weighted_average
        if self.debias and self.count>0:
            value = value/(1-self.momentum**self.count)
            
        return value

def count_parameters(model):
    
    num_parameters = 0
    for pars in model.parameters():
        num_parameters += torch.prod(torch.tensor(pars.shape))
        
    return num_parameters

def profile(learner, epochs):
    with torch.autograd.profiler.profile() as prof:
        learner.fi(epochs)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))