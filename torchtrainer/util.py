import inspect
from torch.optim import lr_scheduler
import math

class Logger:
    """Simple class for logging data."""

    def __init__(self, columns):
        self.data = {}
        self.columns = columns

    def add_data(self, epoch, new_data):
        self.data[epoch] = new_data

    def state_dict(self):
        return {'columns':self.columns, 'data':self.data}
    
    def load_state_dict(self, state_dict):
        self.columns = state_dict['columns']
        self.data = state_dict['data']

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

class CosineAnnealingWarmRestartsImp(lr_scheduler.CosineAnnealingWarmRestarts):
    """Exactly the same as the class CosineAnnealingWarmRestarts from Pytorch with a fix for avoiding a large
    learning rate at the very last step."""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)

    def step(self):

        if self.last_epoch < 0:
            epoch = 0

        epoch = self.last_epoch + 1
        self.T_cur = self.T_cur + 1
        if self.T_cur > self.T_i:
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]