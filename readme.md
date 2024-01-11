# Torchtrainer

Utility modules and functions for training Convolutional Neural Networks using [PyTorch](https://pytorch.org/). 

The main modules are:

* [imagedataset.py](torchtrainer/imagedataset.py): Classes for loading image classification and segmentation datasets;
* [models](torchtrainer/models): A collection of some useful CNN architectures for classification and segmentation;
* [img_util.py](torchtrainer/img_util.py): Utility functions for loading and displaying images; 
* [module_util.py](torchtrainer/module_util.py): Utility functions for working with CNNs architectures, including: splitting the model into layer groups, extracting intermediate activations of a model and measuring the receptive field of CNNs.
* [perf_funcs.py](torchtrainer/perf_funcs.py): Functions and classes for measuring the performance of a CNN. Notable metrics are IoU, f1-score, precision, recall, [soft Dice loss](https://arxiv.org/abs/1606.04797), [focal loss](https://arxiv.org/abs/1708.02002) and [COCO metrics](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) (only for segmentation).
* [inspector.py](torchtrainer/inspector.py): Class for easily inspecting model parameters, activations and gradients;
* [profiling.py](torchtrainer/profiling.py): Utilities for profiling CPU and GPU usage;
* [model_debug.py](torchtrainer/model_debug.py): Utilities for debugging models;

You can install the code as an editable package by running the following command inside the root directory (the directory containing the pyproject.toml file):

```pip install -e .```

or if using conda:

```conda develop .```

If you are using conda and for some reason want to install the package as editable using pip, and want to avoid pip messing up your environment, use

```pip install --no-build-isolation --no-deps -e .```