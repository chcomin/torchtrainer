import inspect
import math
import random
import argparse
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

class Logger:
    """ Class for logging metrics during training and validation. """

    def __init__(self): 
        self.epoch_data = {}
        self.batch_data = {}
        self.current_epoch = 0
        self.current_batch = 0
        
            
    def log(self, epoch, batch_idx, name, value, weight=1):
        """ Log a metric value for a given batch. 
        
        The values are logged into the batch_data attribute of the class. When 
        the method epoch_end() is called, the values are averaged over the batches 
        and stored in the epoch_data attribute. The batch metrics are then erased.

        Parameters
        ----------
        epoch: Epoch number
        batch_idx: Index of the batch in the dataloader
        name: Name of the metric
        value : Value to be logged
        weight: Weight assigned to the value when averaging over the whole epoch.
        Usually, this is the batch size because `value` is an average over a
        batch
        """

        if epoch!=self.current_epoch and epoch!=self.current_epoch+1:
            raise ValueError(f'Current epoch is {self.current_epoch} but {epoch} received')
        if batch_idx>0 and batch_idx!=self.current_batch and batch_idx!=self.current_batch+1:
            raise ValueError(f'Current batch is {self.current_batch} but {batch_idx} received')
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        self.current_epoch = epoch
        self.current_batch = batch_idx

        batch_data = self.batch_data
        if name not in batch_data:
            batch_data[name] = [(value, weight)]
        else:
            batch_data[name].append((value, weight))

    def log_epoch(self, epoch, name, value):
        """ Log a metric value for a given epoch.

        Parameters
        ----------
        epoch: Epoch number
        name: Name of the metric
        value: Value to be logged
        """

        if epoch!=self.current_epoch and epoch!=self.current_epoch+1:
            raise ValueError(f'Current epoch is {self.current_epoch} but {epoch} received')

        self.current_epoch = epoch

        epoch_data = self.epoch_data
        if epoch not in epoch_data:
            epoch_data[epoch] = {}

        epoch_data[epoch][name] = value

    def end_epoch(self):
        """ Calculate the average of the logged values over the batches for the 
        current epoch and store them in the epoch_data attribute. The batch 
        metrics are erased.
        """

        current_epoch = self.current_epoch
        epoch_data = self.epoch_data
        if current_epoch not in epoch_data:
            epoch_data[current_epoch] = {}

        for name, data in self.batch_data.items():
            values, weights = zip(*data)
            values = torch.stack(values).cpu()
            weights = torch.tensor(weights)
            avg = (weights*values).sum()/weights.sum()
            self.epoch_data[current_epoch][name] = avg.item()
        self.batch_data = {}

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

class LoggerPlotter:
    """Plot data logged into a Logger object."""

    def __init__(self, groups):
        """
        In order to organize the plots, it is necessary to group the metrics.
        `groups` is a list of dictionaries. Each dictionary has two keys:
        `names` is a list of the names of the metrics to be plotted together.
        `y_max` is the maximum value for the y-axis.

        For example:
        groups = [
            {'names': ['train/loss', 'valid/loss'], 'y_max': 1.0},
            {'names': ['valid/accuracy'], 'y_max': 1.0}
        ]

        This will create two plots. One for the losses and another for the accuracy.

        Parameters
        ----------
        groups
            List of groups of metrics to be plotted together.
        """
        self.groups = groups

    def get_plot(self, logger):

        df = logger.get_data()
        epochs = df['epoch']

        ncols = 2
        nrows = (len(self.groups)+1)//ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(8, 3*nrows), squeeze=False)
        axes = axes.flatten()

        for group, ax in zip(self.groups, axes):
            names = group['names']
            max_y = group['y_max']
            for name in names:
                values = df[name]
                # Mask nan values
                mask = np.isfinite(values)
                ax.plot(epochs[mask], values[mask], '-o', ms=2, label=name)
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.set_ylim((0, max_y))

        fig.tight_layout()
        # Avoid showing the plot in the notebook
        plt.close()

        return fig
        

class Subset(Dataset):
    """Create a new Dataset containing a subset of images from the input Dataset.
    """

    def __init__(self, ds, indices, transform=None, **attributes):
        """
        Args:
            ds : input Dataset
            indices: indices to use for the new dataset
            transform: transformations to apply to the data. Defaults to None.
            attributes: additional attributes to store in the new dataset.
        """

        self.ds = ds
        self.indices = indices
        self.transform = transform
        for k, v in attributes.items():
            setattr(self, k, v)

    def __getitem__(self, idx):

        items = self.ds[self.indices[idx]]
        if self.transform is not None:
            items = self.transform(*items)

        return items

    def __len__(self):
        return len(self.indices)

class WrapDict:
    """
    Wrapper class to name the return values of a function. Given a function
    that returns a tuple, WrapDict creates a new function that returns a
    dictionary with the names of the tuple elements as keys.
    Parameters
    ----------
    names : Names of the values returned by the function
    func : Function to wrap
    """

    def __init__(self, func, names):
        self.func = func
        self.names = names

    def __call__(self, *args):
        values = self.func(*args)
        return {name: value for name, value in zip(self.names, values)}

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

class ParseKwargs(argparse.Action):
    """Class to use when parsing a dictionary from the command line."""
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)

class ParseText(argparse.Action):
    """Class to use when parsing a text from the command line."""
    def __call__(self, parser, namespace, values, option_string=None):
        text = ' '.join(values)
        setattr(namespace, self.dest, text)

def dict_to_argv(param_dict: dict, positional_args: list | None) -> list:
    """Convert a dictionary to a list mimicking an argv list received by a
    python program from the command line.

    Parameters
    ----------
    param_dict
        The parameter dictionary
    positional_args
        List of positional arguments names. This is necessary because the keys
        of positional arguments are not used in the command line.

    Returns
    -------
        List of strings with the parameters
    """

    if positional_args is None:
        positional_args = []

    sys_argv = []
    # Extract positional arguments and put them first
    for k in positional_args:
        sys_argv.append(param_dict[k])
        del param_dict[k]

    for k, v in param_dict.items():
        sys_argv.append(f"--{k}")
        # If the argument is boolean, the value should be empty
        if v!='' and v is not None:
            # Dictionary values are allowed to be int or float, but command line
            # arguments are always strings
            v = str(v)
            sys_argv.extend(v.split())

    return sys_argv

def seed_all(seed):
    """
    Seed all random number generators for reproducibility. If deterministic is
    True, set cuDNN to deterministic mode.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
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
    acc_names = df.columns[4:]

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

def predict_and_save_val_img(runner, epoch, img_idx, run_path):

    img, _ = runner.ds_valid[img_idx]
    output = runner.predict(img.unsqueeze(0))
    prediction = torch.argmax(output, dim=1)
    prediction = 255*prediction/(runner.num_classes-1)
    pil_img = Image.fromarray(prediction.squeeze().to(torch.uint8).numpy())
    pil_img.save(run_path/'images'/f'idx_{img_idx}_epoch_{epoch}.png')

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