import inspect

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

