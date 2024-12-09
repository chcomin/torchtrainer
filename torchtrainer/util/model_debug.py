'''Useful functions for debugging modules.'''

import matplotlib.pyplot as plt
import torch

def debug_model_state(model, batch, labels, loss_func, plot=True):
    '''Get minimum and maximum values for all activations, parameters, gradients and buffers of a model.'''

    # eval to avoid changing batchnorm stats
    if model.training:
        training = True
        model.eval()
    else:
        training = False

    # Translate modules to names
    module_2_name = {}
    for name, module in model.named_modules(prefix='model'):
        if not isinstance(module, torch.nn.ReLU):
            module_2_name[module] = name

    # awkward hook for saving activations
    stats_act = []
    def hook(module, args, output):
        name = module_2_name[module]
        output = output.detach()
        stats_act.append((name, output.min(), output.max()))

    handles = []
    for name, module in model.named_modules(prefix='model'):
        # No need to save relu activations, we known what they are
        if not isinstance(module, torch.nn.ReLU):
            handle = module.register_forward_hook(hook)
            handles.append(handle)

    # Calculate activations and gradients
    res = model(batch)
    loss = loss_func(res, labels)
    loss.backward()

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Parameters and gradients
    stats_params = []
    stats_grads = []
    for name, values in model.named_parameters():
        stats_params.append((name, values.detach().min(), values.detach().max()))
        stats_grads.append((name, values.grad.min(), values.grad.max()))

    # Buffers (usually batchnorm statistics)
    stats_buffers = []
    for name, values in model.named_buffers():
        if 'num_batches_tracked' not in name:
            stats_buffers.append((name, values.min(), values.max()))

    # Resume model state
    if training:
        model.train()
    model.zero_grad()

    if plot:
        plot_stats(stats_act, 'Activations', 'o')
        plot_stats(stats_params, 'Parameters', '-o')
        plot_stats(stats_grads, 'Gradients', '-o')
        plot_stats(stats_buffers, 'Buffers', '-o')

    return stats_act, stats_params, stats_grads, stats_buffers

def plot_stats(stats, title, marker):
    '''Plot data.'''

    names, mins, maxs = zip(*stats)

    plt.figure(figsize=(14,5))
    plt.plot(names, mins, marker, ms=3)
    plt.plot(names, maxs, marker, ms=3)
    plt.xticks(rotation=90, size=6)
    plt.title(title)