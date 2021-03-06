# Torchtrainer

Utility modules and functions for training Convolutional Neural Networks using [PyTorch](https://pytorch.org/). 

See the [notebooks](notebooks/examples) for examples of training neural networks for image classification and segmentation.

The main modules are:

* [imagedataset.py](torchtrainer/imagedataset.py): Classes for loading image datasets;
* [transforms.py](torchtrainer/transforms.py): Contains image transformation functions, useful for data augmentation. Also contains transformations for easily converting data between the numpy, pillow, pytorch and imgaug libraries. For instance,  the TransfToTensor(img, label) transform converts a 2D or 3D image and respective 2D or 3D label image to a tensor regardless if *img* and *label* are a numpy array, pillow image or imgaug object);
* [models](torchtrainer/models): A collection of some useful CNN architectures for classification and segmentation;
* [learner.py](torchtrainer/learner.py): Contains the `Learner` class, used for training a network. Allows the definition of custom performance functions that are applied during validation, checkpoint saving when performance has improved, custom learning rate schedulers and the definition of callbacks called at the end of each validation epoch;
* [img_util.py](torchtrainer/img_util.py): Utility functions for loading and displaying images as well as visualizing the performance of a model on a set of images (the latter is a work in progress); 
* [module_util.py](torchtrainer/module_util.py): Utility functions for working with CNNs architectures, such as freezing a model or splitting the model into layer groups and defining distinct learning rates. Also contains the `Hooks` class, which allows the registration of arbitrary hooks on any layer of a model.
* [perf_funcs.py](torchtrainer/perf_funcs.py): Functions and classes for measuring the performance of a CNN. Notable metrics are IoU, f1-score, precision, recall, [soft Dice loss](https://arxiv.org/abs/1606.04797), [focal loss](https://arxiv.org/abs/1708.02002) and [COCO metrics](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) (only for segmentation).