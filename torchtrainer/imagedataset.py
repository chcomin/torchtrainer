'''
Dataset storage class
'''

from pathlib import Path
import random
import bisect
import copy
import numpy as np
from PIL import Image
import torch
from torch.utils.data import dataset as torch_dataset
from torch.utils.data import dataloader as torch_dataloader
from .transforms import TransfToTensor, TransfToPil, TransfToNumpy
from .img_util import get_shape
from .util import save_params

class ImageDataset(torch_dataset.Dataset):
    """Image dataset storage class.

    Receives a directory path and stores all images contained in the directory. Images can be retrieved as follows:

    image_ds = ImageDataset(...)
    img, label = image_ds[0]

    Parameters
    ----------
    img_dir : string or Path
        Directory containing the images to be read
    name_to_label_map : callable
        Function with signature name_to_label_map(img_filename) that translates image filenames into
        labels. Receives an image filename (including the extension of the file) and returns the label
        of the respective image.
    filename_filter : list or callable
        If a list, contains names of the image files that should be kept, other images are ignored
        If a function, has signature filename_filter(img_filename), receives an image filename and returns
        True if the image should be kept. The image is discarded otherwise. 
        Note that in both cases the image filename is passed with the extension of the file.
    img_opener : callable
        Function with signature img_opener(img_path) for opening the images. Receives an image path
        and returns a PIL.Image object. Images should be have uint8 type. 
    transforms : list of callable
        List of functions to be applied for image augmentation. Each function should have the signature
        transform(img) and return the transformed image. 
        The arguments of the first transform should be PIL.Image objects, while the return values of the last 
        transform should be float32 torch.Tensor objects.
    cache_size : int
        Size of the memory cache. Images will be stored in memory until the requested size is reached.
    """

    init_params = {}

    @save_params(init_params)
    def __init__(self, img_dir, name_to_label_map, filename_filter=None, img_opener=None,
                 transforms=None, cache_size=0):

        if isinstance(img_dir, str):
            img_dir = Path(img_dir)
        if isinstance(filename_filter, list):
            filename_filter = set(filename_filter)
        if transforms is None:
            transforms = []
        if img_opener is None:
            img_opener = Image.open
        if cache_size>0:
            cache_manager = CacheManager(lambda item: CacheManager.sizeof_pil(item[0]), cache_size)
        else:
            cache_manager = None

        self.img_dir = img_dir
        self.name_to_label_map = name_to_label_map
        self.img_opener = img_opener
        self.transforms = transforms
        self.apply_transform = True
        self.cache_manager = cache_manager

        img_file_paths = []
        for img_file_path in img_dir.iterdir():
            img_filename = img_file_path.name
            if filename_filter is None:
                img_file_paths.append(img_file_path)
            elif isinstance(filename_filter, (set, list)):
                if img_filename in filename_filter: img_file_paths.append(img_file_path)
            elif filename_filter(img_filename):
                img_file_paths.append(img_file_path)

        self.img_file_paths = img_file_paths

    def __getitem__(self, idx):

        if self.cache_manager is None:
            # Do not use `self` to avoid subclasses calling their own get_item
            img, label = ImageDataset.get_item(self, idx)
        else:
            if idx in self.cache_manager:
                img, label = self.cache_manager[idx]
            else:
                img, label = ImageDataset.get_item(self, idx)
                self.cache_manager[idx] = (img, label)

        if self.apply_transform == True:
            img_transf = self.apply_transforms(self.transforms, img)
        else:
            img_transf = img

        # Hack for fastai
        #if isinstance(img_transf, torch.Tensor):
        #    img_transf.size = TensorShape(img_transf.shape[1:])
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        return img_transf, label

    def get_item(self, idx, transforms=None):
        """Same behavior as self.__getitem__() but does not apply transformation functions. Custom
        transformation functions can be passed as an optional parameter.
        
        Parameters
        ----------
        idx : int
            Index of the image
        transforms : list of callable
            List of functions to be applied for image augmentation. See the class docstring for a description.

        Returns
        -------
        img_transf : ImageLike
        label : ImageLike
        """

        if transforms is None:
            transforms = []

        img_file_path = self.img_file_paths[idx]

        img = self.img_opener(img_file_path)
        label = self.name_to_label_map(img_file_path.name)

        img_transf = self.apply_transforms(transforms, img)

        # Hack for fastai
        #if isinstance(img_transf, torch.Tensor):
        #    img_transf.size = TensorShape(img_transf.shape[1:])
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        return img_transf, label

    def __len__(self):

        return len(self.img_file_paths)

    def copy(self):
        """Copy this dataset."""

        return self.subset(lambda x:True)

    def subset(self, filename_filter):
        """Return a new ImageDataset containing only images for which filename_filter(img_filename) is True.
        
        Parameters
        ----------
        filename_filter : list or callable
            If a list, contains names of the image files that should be kept, other images are ignored
            If a function, has signature filename_filter(img_filename), receives an image filename and returns
            True if the image should be kept. The image is discarded otherwise. 
            Note that in both cases the image filename is passed with the extension of the file.

        Returns
        -------
        ImageDataset
            The new dataset.
        """

        #transforms = copy.copy(self.transforms)
        #cache_size = 0 if self.cache_manager is None else self.cache_manager.max_size
        #return self.__class__(self.img_dir, self.name_to_label_map, filename_filter=filename_filter, img_opener=self.img_opener,
        #                      transforms=transforms, cache_size=cache_size)
        init_params = copy.deepcopy(self.init_params)
        del init_params['self']
        init_params['filename_filter'] = filename_filter
        return self.__class__(**init_params)

    def set_transforms(self, transforms):
        """Set the data augmentation transformation functions. 
        
        Parameters
        ----------
        transforms : list of callable
            List of functions to be applied for image augmentation. Each function should have the signature
            transform(img) and return the transformed image. 
            The arguments of the first transform should be PIL.Image objects, while the return values of the last 
            transform should be float32 torch.Tensor objects.       
        """

        self.transforms = transforms

    def dataloader(self, batch_size, shuffle=True, **kwargs):
        """Return a dataloader for this dataset.
        
        Parameters
        ----------
        batch_size : int
            The batch size to use.
        shuffle : bool
            If True, the data is randomly selected for each batch
        kwargs
            Additional keyword arguments are passed to the torch.utils.data.dataloader function.
        """

        return torch_dataloader.DataLoader(self, batch_size=batch_size, shuffle=True, **kwargs)

    def check_dataset(self):
        """Check if all images in the dataset can be read, and if the transformations
        can be successfully applied. It is usefull to call this function right after
        dataset creation.
        """

        img_file_paths = self.img_file_paths

        for item_idx, item in enumerate(iter(self)):
            # Check if all data can be obtained
            pass
        print('All images read')

    def split_train_val(self, valid_set=0.2, seed=None):
        """Split dataset into train and validation. Return two new datasets.

        Parameters
        ----------
        valid_set : float or list
            If float, a fraction `valid_set` of the dataset will be used for validation,
            the rest will be used for training.
            If list, should contain the names of the files used for validation. The remaining
            images will be used for training. Note that the names must include the file extensions.

        Returns
        -------
        train_dataset : ImageDataset
            Dataset to be used for training
        valid_dataset : ImageDataset
            Dataset to be used for validation
        """

        img_file_paths_train, img_file_paths_valid = self.split_train_val_paths(valid_set, seed=seed)
        '''# Hacky way to get parameters passed to __init__ during class construction
        init_pars_train = {}
        init_code = self.__init__.__code__
        for init_par in init_code.co_varnames[1:init_code.co_argcount]:
            if (init_par!='filename_filter'):
                try:
                    init_pars_train[init_par] = self.__getattribute__(init_par)
                except AttributeError:
                    raise AttributeError('Cannot split dataset, init parameter not registered in class')
        init_pars_valid = init_pars_train.copy()

        init_pars_train['filename_filter'] = img_file_paths_train
        init_pars_valid['filename_filter'] = img_file_paths_valid'''

        train_dataset = self.subset(img_file_paths_train)
        valid_dataset = self.subset(img_file_paths_valid)

        return train_dataset, valid_dataset

    def split_train_val_paths(self, valid_set=0.2, seed=None):
        """Generate image names to be used for spliting the dataset.

        Parameters
        ----------
        valid_set : float or list
            If float, a fraction `valid_set` of the dataset will be used for validation,
            the rest will be used for training.
            If list, should containg the names of the files used for validation. The remaining
            images will be used for training. Note that the names must include the file extensions.

        Returns
        -------
        img_file_paths_train : list
            Images used for training
        img_file_paths_valid : list
            Images used for validation
        """

        img_file_paths = self.img_file_paths
        num_images = len(img_file_paths)

        img_file_paths_train = []
        img_file_paths_valid = []

        if isinstance(valid_set, list):

            valid_set_set = set(valid_set)
            for file_idx, img_file_path in enumerate(img_file_paths):
                if img_file_path.name in valid_set_set:
                    img_file_paths_valid.append(img_file_path)
                else:
                    img_file_paths_train.append(img_file_path)

            if (len(img_file_paths_train)+len(img_file_paths_valid))!=len(img_file_paths):
                print('Warning, some files in validation set not found')

        elif isinstance(valid_set, float):
            if seed is not None:
                random.seed(seed)

            num_images_valid = int(num_images*valid_set)
            num_images_train = num_images - num_images_valid

            ind_all = list(range(num_images))
            random.shuffle(ind_all)
            ind_train = ind_all[0:num_images_train]
            ind_valid = ind_all[num_images_train:]

            img_file_paths_train = [img_file_paths[ind] for ind in ind_train]
            img_file_paths_valid = [img_file_paths[ind] for ind in ind_valid]

        img_file_paths_train = [file.name for file in img_file_paths_train]
        img_file_paths_valid = [file.name for file in img_file_paths_valid]

        return img_file_paths_train, img_file_paths_valid

    def as_tensor(self):
        """Converts all images in the dataset to a single torch tensor.

        Returns
        -------
        tensors : torch.Tensor
            Tensor with dimensions (num images, num channels, height, width)
        """

        item = self.__getitem__(0)                         # Open first image to get shape
        for idx, val in enumerate(item):
            if not isinstance(val, torch.Tensor):           
                item[idx] = TransfToTensor()(val)
        
        num_tensors = len(self)
        tensors = [torch.zeros((num_tensors, *val.shape), dtype=val.dtype) for val in item]

        for item_idx, item in enumerate(iter(self)):
            for idx, val in enumerate(item):
                if not isinstance(val, torch.Tensor):           
                    tensors[idx][item_idx] = TransfToTensor()(val)

        return tensors

    def apply_transforms(self, transforms, img):
        """Apply transformations stored in `transforms`.

        Parameters
        ----------
        transforms : list of callable
            List of functions to be applied for image augmentation. See the class docstring for a description.
        img : ImageLike
            Image to be transformed.

        Returns
        -------
        img : ImageLike
            Resulting image.
        """

        if callable(transforms):
            img = transforms(img)
        else:
            for transform in transforms:
                    img = transform(img)
        return img

    def set_apply_transform(self, apply_transform):
        """Set the self.apply_transform attribute. 
        
        Parameters
        ----------
        apply_transform : bool
            If False, transformations are not applied to images when calling self.__getitem___()
        """

        self.apply_transform = apply_transform

    def load_on_memory(self):
        """Load all images on memory."""

        img_file_paths = self.img_file_paths
        img, label = self.get_item(0)

        img = TransfToTensor()(img)
        imgs = torch.zeros((len(img_file_paths),)+img.shape, dtype=img.dtype)
        labels = torch.zeros(len(img_file_paths), dtype=torch.long)
        imgs[0] = img
        labels[0] = label
        for img_idx, img_file_path in enumerate(img_file_paths[1:], start=1):
            img, label = self.get_item(img_idx)
            imgs[img_idx], labels[img_idx] = TransfToTensor()(img), label

        self.cache = {'imgs':imgs, 'labels':labels}            

class ImageSegmentationDataset(ImageDataset):
    """
        Image dataset storage class for segmentation tasks.

        Receives directories for input images and labels and stores all images contained in the directories. 
        Images can be retrieved as follows:

        image_ds = ImageDataset(...)
        img, label = image_ds[0]

        or, in case `weight_func` is not None:

        img, label, weight = image_ds[0]

        Parameters
        ----------
        img_dir : string or Path
            Directory containing the images to be read
        label_dir : string or Path
            Directory containing the labels (ground truth segmentations) to be read
        name_to_label_map : callable
            Function with signature name_to_label_map(img_filename) that translates image filenames into
            labels filenames. Receives an image filename (including the extension of the file) and returns 
            the filename of an image containing the respective label.
        filename_filter : list or callable
            If a list, contains names of the image files that should be kept, other images are ignored
            If a function, has signature filename_filter(img_filename), receives an image filename and returns
            True if the image should be kept. The image is discarded otherwise.
            Note that in both cases the image filename is passed with the extension of the file.
        img_opener : callable
            Function with signature img_opener(img_path) for opening the images. Receives an image path
            and returns a PIL.Image object. Images should be have uint8 type.
        label_opener: callable
            Function with signature label_opener(label_path) for opening the labels. Receives an label path
            and returns a PIL.Image object. The image should contain class indices and have uint8 type
        transforms : list of callable
            List of functions to be applied for image augmentation. Each function should have the signature
            transform(img=None, label=None, weight=None) and return a tuple containing the transformed data. The
            number of elements in the returned tuple is the same as the number of arguments. For instance,  
            transform(img, label) should return (img_transf, label_transf).
            The arguments of the first transform should be PIL.Image objects, while the return values of the last 
            transform should be float32 torch.Tensor objects.
        weight_func : callable
            Function for generating weights associated to each image. Those can be used for defining masks
            or, for instance, weighting the loss function. Must have signature
            weight_func(pil_img, pil_label, img_path) and return a PIL.Image with F (float32) type.
        on_memory : bool
            If True, all images will be read on memory. If False, images are read from the disk when requested.
        """

    def __init__(self, img_dir, label_dir, name_to_label_map, filename_filter=None, img_opener=None,
                 label_opener=None, transforms=None, weight_func=None, cache_size=0):

        super().__init__(img_dir, name_to_label_map, filename_filter=filename_filter, img_opener=img_opener, 
                         transforms=transforms, cache_size=cache_size)

        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        if label_opener is None:
            label_opener = Image.open
        if cache_size>0:
            self.cache_manager.sizeof_func = lambda item: sum(map(CacheManager.sizeof_pil, item))

        self.label_dir = label_dir
        self.label_opener = label_opener
        self.weight_func = weight_func

    def __getitem__(self, idx):
        '''Returns one item from the dataset. Will return an image and label if weight_func was
        not defined during class instantiation or an aditional weight image otherwise.'''

        if self.cache_manager is None:
            # Do not use `self` to avoid subclasses calling their own get_item
            item = ImageSegmentationDataset.get_item(self, idx)
        else:
            if idx in self.cache_manager:
                item = self.cache_manager[idx]
            else:
                item = ImageSegmentationDataset.get_item(self, idx)
                self.cache_manager[idx] = item
    
        if self.apply_transform:
            item_transf = self.apply_transforms(self.transforms, *item)
        else:
            item_transf = item

        # Hack for fastai
        #for r in item_transf:
        #    if isinstance(r, torch.Tensor):
        #        r.size = TensorShape(r.shape[1:])

        if isinstance(item_transf[1], torch.Tensor):
            item_transf[1] = item_transf[1].long().squeeze()  # Label image
        
        return item_transf

    def get_item(self, idx, transforms=None):
        """Same behavior as self.__getitem__() but does not apply transformation functions. Custom
        transformation functions can be passed as an optional parameter.

        Parameters
        ----------
        idx : int
            Index of the image
        transforms : list of callable
            List of functions to be applied for image augmentation. See the class docstring for a description.

        Returns
        -------
        item_transf : tuple of ImageLike
        """

        if transforms is None:
            transforms = []

        img_file_path = self.img_file_paths[idx]

        img = self.img_opener(img_file_path)
        label_file_path = self.label_path_from_image_path(img_file_path)
        label = self.label_opener(label_file_path)

        item = [img, label]
        if self.weight_func is not None:
            weight = self.weight_func(img, label, img_file_path)
            item.append(weight)
        
        item_transf = self.apply_transforms(transforms, *item)

        # Hack for fastai
        #for r in item_transf:
        #    if isinstance(r, torch.Tensor):
        #        r.size = TensorShape(r.shape[1:])

        if isinstance(item_transf[1], torch.Tensor):
            item_transf[1] = item_transf[1].long().squeeze()

        return item_transf

    def subset(self, filename_filter):
        """Return a new ImageSegmentationDataset containing only images for which filename_filter(img_filename) is True.
        
        Parameters
        ----------
        filename_filter : list or callable
            If a list, contains names of the image files that should be kept, other images are ignored
            If a function, has signature filename_filter(img_filename), receives an image filename and returns
            True if the image should be kept. The image is discarded otherwise. 
            Note that in both cases the image filename is passed with the extension of the file.

        Returns
        -------
        ImageSegmentationDataset
            The new dataset.
        """

        transforms = copy.copy(self.transforms)
        cache_size = 0 if self.cache_manager is None else self.cache_manager.max_size
        return self.__class__(self.img_dir, self.label_dir, self.name_to_label_map, filename_filter=filename_filter, img_opener=self.img_opener,
                              label_opener=self.label_opener, transforms=transforms, weight_func=self.weight_func, cache_size=cache_size)

    def label_path_from_image_path(self, img_file_path):
        """Translate image path to label path."""

        img_filename = img_file_path.name
        return self.label_dir/self.name_to_label_map(img_filename)

    def apply_transforms(self, transforms, img, label, weight=None):
        """Apply transformations stored in `transforms`.

        Parameters
        ----------
        transforms : list of callable
            List of functions to be applied for image augmentation. See the class docstring for a description.
        img : ImageLike
            Image to be processed.
        label : ImageLike
            Label to be processed
        weight : ImageLike
            Weight to be processed

        Returns
        -------
        item : tuple of ImageLike
            Resulting images. Either (img, label) if weight is None or (img, label, weight) otherwise.
        """

        if weight is None:
            item = [img, label]
        else:
            item = [img, label, weight]

        if callable(transforms):
            item = transforms(*item)
        else:
            for transform in transforms:
                    item = transform(*item)
        return item

    def load_on_memory(self):
        """Load all images on memory."""

        img_file_paths = self.img_file_paths
        img, label = self.get_item(0)

        img = TransfToTensor()(img)
        imgs = torch.zeros((len(img_file_paths),)+img.shape, dtype=img.dtype)
        labels = torch.zeros_like(imgs, dtype=torch.long)
        imgs[0] = img
        labels[0] = label
        for img_idx, img_file_path in enumerate(img_file_paths[1:], start=1):
            img, label = self.get_item(img_idx)
            imgs[img_idx], labels[img_idx] = TransfToTensor()(img, label)

        self.cache = {'imgs':imgs, 'labels':labels}    

class TensorShape(tuple):
    """Class for adding a `size` atribute on tensors that works on both PyTorch, Jupyter and Fastai."""

    def __new__ (cls, args):
        return super(TensorShape, cls).__new__(cls, tuple(args))

    def __call__(self, dim=None):

        if dim is None:
            return self
        else:
            return self[dim]

class ImageItem:
    """Initial implementation of an image class. Not used."""

    def __init__(self, img, label=None, weight=None):

        self.img = img
        self.label = label
        self.weight = weight

    def update_img(self, img): self.img = img
    def update_label(self, label): self.label = label
    def update_weight(self, weight): self.weight = weight
    def get_img(self): return self.img
    def get_label(self): return self.label
    def get_weight(self): return self.weight
    def get_items(self): return self.img, self.label, self.weight
    def get_defined_items(self):
        '''Return items that are not None in a list. If only the image has been defined, return
        the image (not a list of one item).'''

        ret = [self.img]
        if self.label is not None: ret.append(self.label)
        if self.weight is not None: ret.append(self.weight)
        if len(ret)==1:
            return ret[0]
        else:
            return ret

    def has_label(self): return self.label is not None
    def has_weight(self): return self.weight is not None

    def apply_function(self, func, return_values=False, **kwargs):

        self.img = func(self.img, **kwargs)
        if self.label is not None: self.label = func(self.label, **kwargs)
        if self.weight is not None: self.weight = func(self.weight, **kwargs)

        if return_values:
            ret = [self.img]
            if self.label is not None: ret.append(self.label)
            if self.weight is not None: ret.append(self.weight)
            return ret

class ImagePatchDataset(ImageDataset):
    """Patchwise image dataset storage.
    
    `transforms` must return a tensor with channels on the first axis (so that it can be correctly sliced)
    
    """

    def __init__(self, patch_shape, img_dir, name_to_label_map, filename_filter=None, img_opener=None,
                 transforms=None, stride=None, img_shape=None, patch_transforms=None, cache_size=0):

        super().__init__(img_dir, name_to_label_map, filename_filter=filename_filter, img_opener=img_opener, 
                         transforms=transforms, cache_size=cache_size)

        if stride is None:
            stride = patch_shape
        if isinstance(stride, int):
            stride = (stride,)*len(patch_shape)
        if patch_transforms is None:
            patch_transforms = []

        if len(patch_shape)!=len(stride):
            raise ValueError('`patch_shape` and `stride` must have same length')

        if len(patch_shape)==3:
            is_3d = True
        else:
            is_3d = False

        self.patch_shape = patch_shape
        self.stride = stride
        self.img_shape = img_shape
        self.is_3d = is_3d
        self.patch_transforms = patch_transforms

        patches = generate_patches_corners_for_dataset(self, patch_shape, stride, is_3d, img_shape=img_shape)
        self.patches_corners, self.patches_location = patches

    def __getitem__(self, idx):
        '''Returns one item from the dataset. Will return an image and label if weight_func was
        not defined during class instantiation or an aditional weight image otherwise.'''

        img_patch, label = self.get_item(idx)

        if self.apply_transform == True:
            img_transf = self.apply_transforms(self.patch_transforms, img_patch)
        else:
            img_transf = img_patch

        return img_transf, label

    def get_item(self, idx, img_idx=None, transforms=None):
        '''Returns one item from the dataset. Will return an image and label if weight_func was
        not defined during class instantiation or an aditional weight image otherwise.'''

        if img_idx is None:
            img_idx, patch_corners = self.patches_corners[idx]
        else:
            location = self.patches_location[img_idx]
            _, patch_corners = self.patches_corners[location][idx]        

        if transforms is None:
            transforms = []    

        img, label = self.get_img_item(img_idx)
        if not isinstance(img, torch.Tensor):           
            img = TransfToTensor()(img)
        img_patch = img[(...,)+patch_corners]

        img_transf = self.apply_transforms(transforms, img_patch)

        return img_transf, label

    def get_img_item(self, idx):

        return super().__getitem__(idx)

    def __len__(self):
        return len(self.patches_corners)

    def subset(self, filename_filter):

        transforms = copy.copy(self.transforms)
        cache_size = 0 if self.cache_manager is None else self.cache_manager.max_size
        patch_transforms = copy.copy(self.patch_transforms)
        return self.__class__(self.patch_shape, self.img_dir, self.name_to_label_map, filename_filter=filename_filter,
                              img_opener=self.img_opener, transforms=transforms, stride=self.stride, img_shape=self.img_shape, 
                              patch_transforms=patch_transforms, cache_size=cache_size)

    def set_patch_transforms(self, patch_transforms):

        self.patch_transforms = patch_transforms

class ImagePatchSegmentationDataset(ImageSegmentationDataset):
    """Patchwise image dataset storage."""

    def __init__(self, patch_shape, img_dir, label_dir, name_to_label_map, filename_filter=None, img_opener=None,
                 label_opener=None, transforms=None, stride=None, img_shape=None, patch_transforms=None,
                 weight_func=None, cache_size=0):

        super().__init__(img_dir, label_dir, name_to_label_map, filename_filter=filename_filter, img_opener=img_opener, 
                         label_opener=label_opener, transforms=transforms, weight_func=weight_func, cache_size=cache_size)

        if stride is None:
            stride = patch_shape
        if isinstance(stride, int):
            stride = (stride,)*len(patch_shape)
        if patch_transforms is None:
            patch_transforms = []

        if len(patch_shape)!=len(stride):
            raise ValueError('`patch_shape` and `stride` must have same length')

        if len(patch_shape)==3:
            is_3d = True
        else:
            is_3d = False

        self.patch_shape = patch_shape
        self.stride = stride
        self.img_shape = img_shape
        self.is_3d = is_3d
        self.patch_transforms = patch_transforms

        patches = generate_patches_corners_for_dataset(self, patch_shape, stride, is_3d, img_shape=img_shape)
        self.patches_corners, self.patches_location = patches

    def __getitem__(self, idx):
        '''Returns one item from the dataset. Will return an image and label if weight_func was
        not defined during class instantiation or an aditional weight image otherwise.'''

        item = self.get_item(idx)

        if self.apply_transform == True:
            item_transf = self.apply_transforms(self.patch_transforms, *item)
        else:
            item_transf = item

        if isinstance(item_transf[1], torch.Tensor):
            item_transf[1] = item_transf[1].long().squeeze()
        
        return item_transf

    def get_item(self, idx, img_idx=None, transforms=None):
        '''Returns one item from the dataset. Will return an image and label if weight_func was
        not defined during class instantiation or an aditional weight image otherwise.'''

        if img_idx is None:
            img_idx, patch_corners = self.patches_corners[idx]
        else:
            location = self.patches_location[img_idx]
            _, patch_corners = self.patches_corners[location][idx]        

        if transforms is None:
            transforms = []    
        
        item = self.get_img_item(img_idx)
        for idx, val in enumerate(item):
            if not isinstance(val, torch.Tensor):           
                item[idx] = TransfToTensor()(val)
        
        item_patch = []
        for value in item:
            item_patch.append(value[(...,)+patch_corners])

        # Apply custom transforms
        item_transf = self.apply_transforms(transforms, *item_patch)

        if isinstance(item_transf[1], torch.Tensor):
            item_transf[1] = item_transf[1].long().squeeze()

        return item_transf

    def get_img_item(self, idx):

        return super().__getitem__(idx) 

    def __len__(self):
        return len(self.patches_corners)

    def subset(self, filename_filter):

        transforms = copy.copy(self.transforms)
        cache_size = 0 if self.cache_manager is None else self.cache_manager.max_size
        patch_transforms = copy.copy(self.patch_transforms)
        return self.__class__(self.patch_shape, self.img_dir, self.label_dir, self.name_to_label_map, filename_filter=filename_filter,
                              img_opener=self.img_opener, label_opener=self.label_opener, transforms=transforms,  
                              stride=self.stride, img_shape=self.img_shape, patch_transforms=patch_transforms,
                              weight_func=self.weight_func, cache_size=cache_size)

    def set_patch_transforms(self, patch_transforms):

        self.patch_transforms = patch_transforms

class PatchedImage:
    """Class used for representing patches in an image. Method `get_image_from_patches` can
    be used for reconstructing an image from patches.

    TODO: Implement mode='pad', where the image is padded so that its shape-patch_shape
        is divisible by stride.
    """

    def __init__(self, img, patch_shape, stride=None, is_3d=False, mode='fit'):

        if not isinstance(img, torch.Tensor):
            img = TransfToTensor(is_3d)(img)
        has_channels = self._has_channels(img.ndim, is_3d)
        img_shape = img.shape
        if has_channels:
            img_shape = img_shape[1:]
        ndim = len(img_shape)
        if stride is None:
            stride = patch_shape
        if isinstance(stride, int):
            stride = (stride,)*ndim
        if isinstance(patch_shape, int):
            patch_shape = (patch_shape,)*ndim
        if len(patch_shape)!=len(stride):
            raise ValueError('`patch_size` and `stride` must have same length')
        if mode=='fit':
            padding = (0,)*ndim

        # Get number of patches along each axis
        shape = []
        for i in range(ndim):
            size = int((img_shape[i]+2*padding[i]-patch_shape[i])/stride[i])+1
            if (size-1)*stride[i] != img_shape[i]+2*padding[i]-patch_shape[i]: 
                # If patches do not fit perfectly
                size += 1
            shape.append(size)
        shape = tuple(shape)

        self.img = img
        self.patch_shape = tuple(patch_shape)
        self.stride = tuple(stride)
        self.is_3d = is_3d
        self.img_shape = img_shape
        self.patches_corners = []
        self.shape = shape
        self.size = shape[0]*shape[1]
        self.patches_corners = self.generate_patches_corners_for_image(img_shape, patch_shape, stride)

    @classmethod
    def generate_patches_corners_for_image(cls, img_shape, patch_shape, stride):
        """Generate patches positions"""

        if len(img_shape)==2:
            patches_corners = cls._generate_patches_corners_for_2d_image(img_shape, patch_shape, stride)
        elif len(img_shape)==3:
            patches_corners = cls._generate_patches_corners_for_3d_image(img_shape, patch_shape, stride)
        else:
            raise Exception('Image must be 2D or 3D')

        return patches_corners

    @classmethod
    def _generate_patches_corners_for_2d_image(cls, img_shape, patch_shape, stride):
        """See `generate_patches_corners_for_image`."""

        patches_corners = []
        for row in range(0, img_shape[0]-patch_shape[0]+stride[0], stride[0]):
            if (row+patch_shape[0])>=img_shape[0]:
                # Do not go over image border
                row = img_shape[0] - patch_shape[0]
            for col in range(0, img_shape[1]-patch_shape[1]+stride[1], stride[1]):
                if (col+patch_shape[1])>=img_shape[1]:
                    # Do not go over image border
                    col = img_shape[1] - patch_shape[1]

                patch_corners = (slice(row, row+patch_shape[0]), slice(col, col+patch_shape[1]))
                patches_corners.append(patch_corners)
        return patches_corners

    @classmethod
    def _generate_patches_corners_for_3d_image(cls, img_shape, patch_shape, stride):
        """See `generate_patches_corners_for_image`."""

        patches_corners = []
        for plane in range(0, img_shape[0]-patch_shape[0]+stride[0], stride[0]):
            if (plane+patch_shape[0])>=img_shape[0]:
                # Do not go over image border
                plane = img_shape[0] - patch_shape[0]
            for row in range(0, img_shape[1]-patch_shape[1]+stride[1], stride[1]):
                if (row+patch_shape[1])>=img_shape[1]:
                    # Do not go over image border
                    row = img_shape[1] - patch_shape[1]
                for col in range(0, img_shape[2]-patch_shape[2]+stride[2], stride[2]):
                    if (col+patch_shape[2])>=img_shape[2]:
                        # Do not go over image border
                        col = img_shape[2] - patch_shape[2]

                    patch_corners = (slice(plane, plane+patch_shape[0]), slice(row, row+patch_shape[1]),
                                     slice(col, col+patch_shape[2]))
                    patches_corners.append(patch_corners)

        return patches_corners

    def __getitem__(self, idx):

        patch_corners = self.patches_corners[idx]
        return self.img[(...,)+patch_corners]

    def __setitem__(self, idx, patch):

        patch_corners = self.patches_corners[idx]
        self.img[(...,)+patch_corners] = patch

    def getitem_rc(self, row, col):
        """Get patch according to its row and column position on the grid."""

        idx = row*self.shape[1] + col
        return self.__getitem__(idx)

    def setitem_rc(self, row, col, patch):
        """Set patch according to its row and column position on the grid."""
            
        idx = row*self.shape[1] + col
        self.__setitem__(idx, patch)

    @staticmethod
    def _has_channels(ndim, is_3d):
        """Identify if image has channels (color, CNN features, etc)."""

        if ndim==2:
            has_channels = False
        elif ndim==3:
            if is_3d:
                # Grayscale 3D image
                has_channels = False
            else:
                # Color 2D image
                has_channels = True
        elif ndim==4:
            has_channels = True
        else:
            raise ValueError("Can't detect if image has channels.")

        return has_channels

    @staticmethod
    def _count_patches(img_shape, patch_shape, stride):
        """Count the number of patches according to the provided parameters."""

        num_p_rows = (img_shape[0]-patch_shape[0])//stride[0] + 1
        if num_p_rows*stride[0]!=img_shape[0]:
            # If patches do not fit perfectly
            num_p_rows += 1
        num_p_cols = (img_shape[1]-patch_shape[1])//stride[1] + 1
        if num_p_cols*stride[1]!=img_shape[1]:
            # If patches do not fit perfectly
            num_p_cols += 1

        return num_p_rows*num_p_cols

    @classmethod
    def get_image_from_patches(cls, patches, img_shape, stride=None, operation='max', is_3d=False):
        """Reconstruct image from a set of patches. Note that no checks are done to verify if 
        the patches fit on the image. It is expected that the patches were generated by instantiating
        this class.

        Parameter `operation` sets the approach for overlapping patches. Possible values are {'max', 
        'min', 'mean'}.

        If patches have channels, first value of `img_shape` is discarded. Thus, you can pass img.shape
        as the shape for the final image or img.shape[1:].
        """

        if not isinstance(patches[0], torch.Tensor):
            patches = [TransfToTensor(is_3d)(patch) for patch in patches]
        has_channels = cls._has_channels(patches[0].ndim, is_3d)
        patch_shape = patches[0].shape
        if has_channels:
            patch_shape = patch_shape[1:]
        ndim = len(patch_shape)
        if ndim!=len(img_shape):
            img_shape = img_shape[1:]
        if stride is None:
            stride = patch_shape
        elif isinstance(stride, int):
            stride = (stride,)*ndim

        patches_corners = cls.generate_patches_corners_for_image(img_shape, patch_shape, stride)

        if has_channels:
            img_full_shape = (patches[0].shape[0],) + img_shape
        else:
            img_full_shape = img_shape
        img = torch.zeros(img_full_shape, dtype=patches[0].dtype)
        if operation=='mean':
            img_count = torch.zeros(img_full_shape, dtype=int)
        for patch_corners, patch in zip(patches_corners, patches):
            img_patch = img[(...,) + patch_corners]
            if operation=='max':
                img_patch[:] = torch.where(img_patch>patch, img_patch, patch)
            elif operation=='min':
                img_patch[:] = torch.where(img_patch<patch, img_patch, patch)
            elif operation=='mean':
                img_patch[:] = img_patch + patch
                img_count[(...,) + patch_corners] += 1

        if operation=='mean':
            mask = img_count>0
            img[mask] = img[mask]/img_count[mask]

        return img 

def generate_patches_corners_for_dataset(dataset, patch_shape, stride, is_3d, img_shape=None):
    '''If img_shape is None, generates indices by opening each image to get the
    respective shape. This is useful when images have distinct sizes. If img_shape
    is not None, uses that shape and the images are not opened, which is much faster.'''

    if img_shape is None:
        must_open = True
    else:
        must_open = False

    patches_corners = []
    patches_location = []
    for img_idx, img_file_path in enumerate(dataset.img_file_paths):
        if must_open:
            try:
                img, _ = dataset.get_img_item(img_idx)
            except Exception:
                raise Exception(f'Cannot get image {img_file_path}\n')
            # Instantiate class in order to disconsider channel information when getting the shape
            img_shape = PatchedImage(img, patch_shape, stride, is_3d=is_3d).img_shape

        patches_corners_img = PatchedImage.generate_patches_corners_for_image(img_shape, patch_shape, stride)
        num_patches = len(patches_corners)
        location = slice(num_patches, num_patches+len(patches_corners_img))
        patches_corners.extend(zip([img_idx]*len(patches_corners_img), patches_corners_img))
        patches_location.append(location)

    return patches_corners, patches_location

class CacheManager:

    def __init__(self, sizeof_func, max_size):
        
        self.sizeof_func = sizeof_func
        self.max_size = max_size
        self.data = {}
        self.cache_size = 0

    def __setitem__(self, key, value):

        data = self.data
        data_size = self.sizeof_func(value)
        old_size = 0
        is_update = False
        if key in data:
            old_size = data[key][1]
            is_update = True

        new_cache_size = self.cache_size + data_size - old_size
        if new_cache_size<=self.max_size:
            data[key] = (value, data_size)
        elif is_update:
            # Cannot add new data with same key. Need to remove old data
            del data[key]        

        self.cache_size = new_cache_size

    def _setitem__cyclic(self, key, value):
        """Set item so that old items are removed if max_size has been reached. Usually this is not
        desired since we will just keep removing and adding new items if the whole dataset does not 
        fit in the memory."""

        data = self.data
        data_size = self.sizeof_func(value)
        if key in data:
            old_size = data[key][1]
            del data[key]
        else:
            old_size = 0
        data[key] = (value, data_size)
        cache_size = self.cache_size + data_size - old_size
        if cache_size>self.max_size:
            items_to_remove = []
            for key, (value, size) in data.items():
                items_to_remove.append(key)
                cache_size -= size
                if cache_size<=self.max_size:
                    break
            for key in items_to_remove:
                del data[key]

        self.cache_size = cache_size

    def __getitem__(self, key):
        return self.data[key][0]

    def __len__(self):
        return len(self.data)
                
    def __contains__(self, key):
        return key in self.data

    @classmethod
    def sizeof_count(cls, img):
        return 1

    @classmethod
    def sizeof_pil(cls, img):
        img_np = np.array(img)
        return cls.sizeof_numpy(img_np)
        
    @classmethod
    def sizeof_numpy(cls, array):
        return array.nbytes

    @classmethod
    def sizeof_torch(cls, tensor):
        return tensor.element_size()*tensor.numel()
        


    