import torch
from torch.profiler import profile as pt_profile, ProfilerActivity
from contextlib import nullcontext
import copy
import time
import torch.nn as nn
from typing import Callable

try:
    import fvcore.nn
except ModuleNotFoundError:
    HAS_FVCORE = False
else:
    HAS_FVCORE = True

def profile_model(model: nn.Module, input_shape: tuple[int,...], no_grad: bool = True, call_backward: bool = False, 
                  sort_by: str = 'cuda_time_total', device: str = 'cuda', use_float16: bool = False) -> str:
    """
    Profile model execution using the Pytorch profiler and prints the results on the screen. The model will be called 
    as ``res = model(input).sum()``, with an optional call to res.backward().

    Args:
        model: A Pytorch model.
        input_shape: size of the input to the model.
        no_grad: profile with torch.no_grad() context.
        call_backward: profile with a call to .backward() from the result of the model. Must be False
          if `no_grad=True`.
        sort_by: how to order the results. Valid values are ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``, ``self_cpu_memory_usage``, 
                ``self_cuda_memory_usage``, ``count``.
        device: device to profile. Can be ``cpu`` or ``cuda``.
        use_float16: set if half precision should be used for the forward pass.

    Returns:
        The result table as a string
    """
    
    if no_grad and call_backward:
        raise ValueError('Error, can only call backward if no_grad is False')
    
    # Avoid changing the device of original model
    model = copy.deepcopy(model)
    
    input = torch.rand(input_shape, device=device)
    model.to(device)

    if no_grad:
        cm = torch.no_grad()
    else:
        cm = nullcontext()
    
    # Do some warmup
    with cm:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_float16):
            res = model(input).sum()
        if call_backward:
            res.backward()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with cm:
        with pt_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True, record_shapes=True, with_flops=True) as prof:
            initial_memory = torch.cuda.max_memory_allocated()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_float16):
                res = model(input).sum()
            if call_backward:
                res.backward()
    torch.cuda.synchronize()
    final_memory = torch.cuda.max_memory_allocated()
            
    memory_allocated = final_memory - initial_memory

    report_str = prof.key_averages(group_by_input_shape=True).table(sort_by=sort_by, top_level_events_only=True, max_name_column_width=20, max_shapes_column_width=40)
    result = f'{report_str}\rGPU memory allocated: {memory_allocated/2**30} GiB'
    print(result)

    return result

def model_info(model: nn.Module, input_shape: tuple[int,...]) -> str:
    '''Prints number of floating points, activations and parameters shapes'''

    if not HAS_FVCORE:
        print("The fvcore package is not installed. Unable to count flops and activations.")
        return

    # Avoid changing the device of original model
    model = copy.deepcopy(model)
    model.to('cpu')
    
    input = torch.rand(input_shape)
    flops = fvcore.nn.FlopCountAnalysis(model, input)
    acts = fvcore.nn.ActivationCountAnalysis(model, input)

    result = fvcore.nn.flop_count_table(flops, activations=acts, show_param_shapes=True)
    print(result)

    return result
    
def benchmark_model(model: nn.Module, input_shape: tuple[int,...], no_grad: bool = True, call_backward: bool = False, 
                    device: str = 'cuda', use_float16: bool = False, return_model_info: bool = False) -> dict:
    
    """
    Benchmark model execution and returns the results in a dictionary. The model will be called as ``res = model(input).sum()``, 
    with an optional call to res.backward().

    Args:
        model: A Pytorch model.
        input_shape: size of the input to the model.
        no_grad: profile with torch.no_grad() context.
        call_backward: profile with a call to .backward() from the result of the model. Must be False
          if `no_grad=True`.
        device: device to profile. Can be ``cpu`` or ``cuda``.
        use_float16: set if half precision should be used for the forward pass.
        return_model_info: if False, do not measure number of parameters, activations and flops, which takes some extra time.

    Returns:
        A dictionary containing:
          params: number of parameters in the model.
          activations: number of activations.
          flops: estimation of the number of floating points operations. Only considers conv and linear layers
          memory: maximum GPU memory used by the model
          time_cpu: CPU time
          time_gpu: GPU time
          info: list containing the units of the measuremets. 
    """

    _M = 10**6  # Mega
    _G = 10**9  # Giga
    _GiB = 2**30  

    if no_grad and call_backward:
        raise ValueError('Error, can only call backward if no_grad is False')
    if device=='cpu':
        dtype=torch.bfloat16
    else:
        dtype=torch.float16

    if return_model_info and not HAS_FVCORE:
        print("The fvcore package is not installed. Unable to count flops and activations.")

    model = copy.deepcopy(model)    

    input = torch.rand(input_shape, device=device)
    model.to(device)

    if no_grad:
        cm = torch.no_grad()
    else:
        cm = nullcontext()

    stats = {}

    if return_model_info:
        num_params = sum([p.numel() for p in model.parameters()])
        if HAS_FVCORE:
            acts = fvcore.nn.ActivationCountAnalysis(model, input).total()
            flops = fvcore.nn.FlopCountAnalysis(model, input).total()
        stats = {
            'params':num_params/_M,
            'activations':acts/_G,
            'flops':flops/_G
        }

    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True) 
    
    # Do some warmup
    with cm:
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_float16):
            res = model(input).sum()
        if call_backward:
            res.backward()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    with cm:
        gpu_start.record()
        cpu_start = time.time()
        initial_memory = torch.cuda.max_memory_allocated()
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_float16):
            res = model(input).sum()
        if call_backward:
            res.backward()
        gpu_end.record()
        cpu_end = time.time()

    torch.cuda.synchronize()
    final_memory = torch.cuda.max_memory_allocated()

    memory_allocated = final_memory - initial_memory
    time_gpu = gpu_start.elapsed_time(gpu_end)/1000
    time_cpu = (cpu_end - cpu_start)
    #el_time = max([time_gpu, time_cpu])

    stats.update({
        'memory':memory_allocated/_GiB,
        'time_cpu':time_cpu,
        'time_gpu':time_gpu
    })

    stats['info'] = []
    if return_model_info:
        stats['info'].extend(['params: M', 'activations: G', 'flops: G'])
    stats['info'].extend(['memory: GiB', 'time_cpu: s', 'time_gpu: s'])
        

    return stats

def benchmark_function(func: Callable, func_params: tuple = (), profile: bool = False) -> dict:
    
    """
    Benchmark function execution and returns the results as a dictionary or as a string. 

    *Note: the function is executed two times. The first execution is necessary as a warmup. Only the
    second execution is benchmarked.

    Args:
        func: the function to benchmark.
        profile: if True, profile the code using Pytorch's profiler. If False, only measure
          max gpu memory used and cpu and gpu times.
        func_params: tuple containing input parameters for executing the function.

    Returns:
        If `profile=True`, returns a string containing the result of the Pytorch's profiler. If false,
        returns a dictionary containing:
          memory: maximum GPU memory used by the model
          time_cpu: CPU time
          time_gpu: GPU time
          info: list containing the units of the measuremets. 
    """

    _GiB = 2**30  

    if profile:
        cm = pt_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True, record_shapes=True, with_flops=True)
    else:
        cm = nullcontext()

    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True) 
    
    # Do some warmup
    func(*func_params)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    gpu_start.record()
    cpu_start = time.time()
    initial_memory = torch.cuda.max_memory_allocated()
    with cm:
        func(*func_params)
    gpu_end.record()
    cpu_end = time.time()

    torch.cuda.synchronize()
    final_memory = torch.cuda.max_memory_allocated()

    memory_allocated = final_memory - initial_memory
    time_gpu = gpu_start.elapsed_time(gpu_end)/1000
    time_cpu = (cpu_end - cpu_start)

    if profile:
        report_str = cm.key_averages(group_by_input_shape=True).table(sort_by='cuda_time_total', top_level_events_only=True, max_name_column_width=20, max_shapes_column_width=40)
        stats = f'{report_str}\rGPU memory allocated: {memory_allocated/_GiB} GiB'
    else:
        stats = {
            'memory':memory_allocated/_GiB,
            'time_cpu':time_cpu,
            'time_gpu':time_gpu,
            'info':['memory: GiB', 'time_cpu: s', 'time_gpu: s']
        }

    return stats

if __name__=='__main__':

    class Model(nn.Module):
        def __init__(self, num_layers=100, num_channels=16):
            super().__init__()
            self.layers = nn.ModuleList([nn.Conv2d(1, num_channels, 3)] + [nn.Conv2d(num_channels, num_channels, 3) for _ in range(num_layers)] + [nn.Conv2d(num_channels, 2, 3)])

        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return x

    model = Model(200, 4).to('cuda')
    input_shape = (1, 1, 1200, 1200)

    '''profile_model(model, input_shape, no_grad=True, call_backward=False, device='cuda', use_float16=False)
    print('\n\n')
    profile_model(model, input_shape, no_grad=False, call_backward=True, device='cuda', use_float16=False)
    print('\n\n')
    profile_model(model, input_shape, no_grad=False, call_backward=True, device='cuda', use_float16=True)
    print('\n\n')
    model_info(model, input_shape)
    print('\n\n')'''

    '''stats = benchmark_model(model, input_shape, no_grad=True, call_backward=False, device='cuda', use_float16=False)
    print(f'{stats}\n\n')'''
    stats = benchmark_model(model, input_shape, no_grad=False, call_backward=True, device='cuda', use_float16=False)
    print(f'{stats}\n\n')
    '''stats = benchmark_model(model, input_shape, no_grad=False, call_backward=True, device='cuda', use_float16=True)
    print(f'{stats}\n\n')
    stats = benchmark_model(model, input_shape, no_grad=False, call_backward=True, device='cuda', use_float16=False, return_model_info=True)
    print(f'{stats}\n\n')'''


    input = torch.rand(1, 1, 1200, 1200, device='cuda')
    def test_func(model, input):
        model(input).sum().backward()

    stats = benchmark_function(test_func, (model, input), profile=False)
    print(f'{stats}\n\n')