import inspect
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

class Logger:
    """
    Class for logging metrics during training and validation.
    """

    def __init__(self): 
        self.epoch_data = {}
        self.current_epoch = 0
        #self.names = metric_names
            
    def log(self, epoch, name, value):
        """Log a metric value for a given epoch.

        Parameters
        ----------
        epoch : int
            Epoch number
        name : str
            Name of the metric
        value : float | int
            Value to be logged
        """

        if epoch!=self.current_epoch and epoch!=self.current_epoch+1:
            raise ValueError(f'Current epoch is {self.current_epoch} but {epoch} received')

        epoch_data = self.epoch_data
        if epoch not in epoch_data:
            epoch_data[epoch] = {}
            self.current_epoch = epoch

        epoch_data[epoch][name] = value

    def get_data(self):
        """Returns a pandas dataframe with the logged data.

        Returns
        -------
        pd.DataFrame
            The dataframe
        """

        df = pd.DataFrame(self.epoch_data).T
        df.insert(0, 'epoch', df.index)

        return df

class Subset(Dataset):
    """Create a new Dataset containing a subset of images from the input Dataset.
    """

    def __init__(self, ds, indices, transform=None):
        """
        Args:
            ds : input Dataset
            indices: indices to use for the new dataset
            transform: transformations to apply to the data. Defaults to None.
        """

        self.ds = ds
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):

        items = self.ds[self.indices[idx]]
        if self.transform is not None:
            items = self.transform(*items)

        return items

    def __len__(self):
        return len(self.indices)

class SingleMetric:
    """
    Class for storing a function representing a performance metric.
    """

    def __init__(self, metric_name, func):
        """
        Create a SingleMetric object from a performance function.

        Parameters
        ----------
        metric_name : Name of the metric
        func : Function that calculates the metric
        """
        self.metric_name = metric_name
        self.func = func

    def __call__(self, *args):
        return (self.metric_name, self.func(*args))

class MultipleMetrics:
    """
    Class for storing a function that calculates many performance metrics in one call.
    """

    def __init__(self, metric_names, func):
        """
        Create a MultipleMetrics object from a performance function.

        Parameters
        ----------
        metric_name : Name of the metric
        func : Function that calculates the metric
        """
        self.metric_names = metric_names
        self.func = func

    def __call__(self, *args):
        results = self.func(*args)
        return ((name,result) for name, result in zip(self.metric_names, results))

def seed_all(seed, deterministic=True):
    """
    Seed all random number generators for reproducibility. If deterministic is
    True, set cuDNN to deterministic mode.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    """
    Set Python and numpy seeds for dataloader workers. Each worker receives a 
    different seed in initial_seed().
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def show_log(logger):
    """
    Plot the logged data from a Logger object in a Jupyter notebook.
    """

    df = logger.get_data()
    epochs = df['epoch']
    train_loss = df['train/loss']
    valid_loss = df['valid/loss']
    acc_names = df.columns[3:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
    ax1.plot(epochs, train_loss, '-o', ms=2, label='Train loss')
    ax1.plot(epochs, valid_loss, '-o', ms=2, label='Valid loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim((0,1.))
    ax1.legend()

    for name in acc_names:
        ax2.plot(epochs, df[name], '-o', ms=2, label=name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(name)
        ax2.set_ylim((0,1.))
        ax2.legend()
    fig.tight_layout()

    display.clear_output(wait=True)
    plt.show()

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