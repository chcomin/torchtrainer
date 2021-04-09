'''
Dataset storage class
'''

from pathlib import Path
from PIL import Image
import random
from torch.utils.data import dataset as torch_dataset
from torch.utils.data import dataloader as torch_dataloader
import torch
import bisect
import copy
from .transforms import TransfToTensor, TransfToPil

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
    on_memory : bool
        If True, all images will be read on memory. If False, images are read from the disk when requested.
    """

    def __init__(self, img_dir, name_to_label_map, filename_filter=None, img_opener=None,
                 transforms=None, on_memory=False):

        if isinstance(img_dir, str):
            img_dir = Path(img_dir)
        if isinstance(filename_filter, list):
            filename_filter = set(filename_filter)
        if transforms is None:
            transforms = []
        if img_opener is None:
            img_opener = Image.open

        self.img_dir = img_dir
        self.name_to_label_map = name_to_label_map
        self.img_opener = img_opener
        self.transforms = transforms
        self.apply_transform = True
        self.on_memory = on_memory

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

        if on_memory:
            self.load_on_memory()

    def __getitem__(self, idx):

        if self.on_memory:
            img, label = self.cache['imgs'][idx], self.cache['labels'][idx]
            img = TransfToPil()(img)
        else:
            img, label = self.get_item(idx)

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

        img_dir = self.img_dir
        name_to_label_map = self.name_to_label_map
        img_opener = self.img_opener
        transforms = copy.copy(self.transforms)
        return self.__class__(img_dir, name_to_label_map, filename_filter=filename_filter, img_opener=img_opener,
                              transforms=transforms)

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

        shapes = []
        for img_idx, img_file_path in enumerate(img_file_paths):
            # Check if all data can be obtained
            try:
                ret_vals = self.__getitem__(img_idx)     # May return multiple items
            except Exception:
                raise Exception(f'Cannot get image {img_file_paths[img_idx]} at index {img_idx}\n')

            for idx, ret_val in enumerate(ret_vals):
                # Check if data has the same shape
                if not isinstance(ret_val, torch.Tensor):
                    ret_val = torch.tensor(ret_val)
                    if ret_val.ndim==0:
                        ret_val = torch.tensor([ret_val]) 
                if len(shapes)<(idx+1):
                    shapes.append(ret_val.shape)
                elif ret_val.shape!=shapes[idx]:
                    raise Exception(f"Data has different shape at index {img_idx}")

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

        img_file_paths = self.img_file_paths
        ret_vals = self.__getitem__(0)           # Open first image to get shape
        if not isinstance(ret_vals[1], torch.Tensor):   # In case label is not a tensor
            ret_vals[1] = torch.tensor(ret_vals[1])

        num_tensors = len(img_file_paths)
        tensors = [torch.zeros((num_tensors, *val.shape), dtype=val.dtype) for val in ret_vals]

        for file_idx, img_file_path in enumerate(img_file_paths):
            ret_vals = self.__getitem__(file_idx)
            for idx, ret_val in enumerate(ret_vals):
                tensors[idx][file_idx] = ret_val

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
    """Image dataset storage class for segmentation tasks.

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
                 label_opener=None, transforms=None, weight_func=None, on_memory=False):

        super().__init__(img_dir, name_to_label_map, filename_filter=filename_filter, img_opener=img_opener,
                 transforms=transforms, on_memory=on_memory)

        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        if label_opener is None:
            label_opener = Image.open

        self.label_dir = label_dir
        self.label_opener = label_opener
        self.weight_func = weight_func

    def __getitem__(self, idx):
        '''Returns one item from the dataset. Will return an image and label if weight_func was
        not defined during class instantiation or an aditional weight image otherwise.'''

        if self.on_memory:
            item = self.cache['imgs'][idx], self.cache['labels'][idx]
            item = TransfToPil()(*item)
        else:
            item = self.get_item(idx)

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

        img_dir = self.img_dir
        label_dir = self.label_dir
        name_to_label_map = self.name_to_label_map
        img_opener = self.img_opener
        label_opener = self.label_opener
        transforms = copy.copy(self.transforms)
        weight_func = self.weight_func
        return self.__class__(img_dir, label_dir, name_to_label_map, filename_filter=filename_filter, img_opener=img_opener,
                              label_opener=label_opener, transforms=transforms, weight_func=weight_func)

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
    """Patchwise image dataset storage."""

    def __init__(self, patch_size, *args, stride=None, patch_transforms=None, **kwargs):

        super().__init__(*args, **kwargs)

        if stride is None:
            stride = patch_size
        if isinstance(stride, int):
            stride = (stride,)*len(patch_size)
        if patch_transforms is None:
            patch_transforms = []

        if len(patch_size)!=len(stride):
            raise ValueError('`patch_size` and `stride` must have same length')

        self.patch_size = patch_size
        self.stride = stride
        self.patch_transforms = patch_transforms

        self.generate_patches_corners_for_dataset()

    def __getitem__(self, idx):
        '''Returns one item from the dataset. Will return an image and label if weight_func was
        not defined during class instantiation or an aditional weight image otherwise.'''

        img_index, patch_corners = self.patches_corners[idx]
        ret = super().__getitem__(img_index)
        if self.weight_func is None:
            img, label = ret
        else:
            img, label, weight = ret

        #first_row, first_col, last_row, last_col = patch_corners
        img_patch = self.crop_img(img, patch_corners)
        label_patch = self.crop_img(label, patch_corners)

        if self.weight_func is not None:
            weight_patch = self.crop_img(weight, patch_corners)
            # Apply transforms on patch
            ret_transf = self.apply_transforms(self.patch_transforms, img_patch, label_patch, weight_patch)
        else:
            ret_transf = self.apply_transforms(self.patch_transforms, img_patch, label_patch)

        return ret_transf

    def __len__(self):
        return len(self.patches_corners)

    def subset(self, filename_filter):

        img_dir = copy.copy(self.img_dir)
        label_dir = copy.copy(self.label_dir)
        name_to_label_map = self.name_to_label_map
        img_opener = self.img_opener
        label_opener = self.label_opener
        transforms = copy.copy(self.transforms)
        weight_func = self.weight_func
        patch_size = copy.copy(self.patch_size)
        stride = copy.copy(self.stride)
        patch_transforms = copy.copy(self.patch_transforms)
        return self.__class__(patch_size, img_dir, label_dir, name_to_label_map, stride=stride,
                              filename_filter=filename_filter, img_opener=img_opener, label_opener=label_opener,
                              transforms=transforms, patch_transforms=patch_transforms, weight_func=weight_func)

    def generate_patches_corners_for_dataset(self, img_shape=None):
        '''If img_shape is None, generates indices by opening each image to get the
        respective shape. This is useful when images have distinct sizes. If img_shape
        is not None, uses that shape and the images are not opened, which is much faster.'''

        if img_shape is None:
            must_open = True
        else:
            must_open = False

        self.patches_corners = []
        self.patch_index_accumulator = [0]
        for img_idx, img_file_path in enumerate(self.img_file_paths):
            if must_open:
                try:
                    ret = super().__getitem__(img_idx)
                except Exception:
                    raise Exception(f'Cannot get image {img_file_path}\n')
                img_shape = self.get_shape(ret[0])
                if len(self.patch_size)!=len(img_shape):
                    raise ValueError('Length of `patch_size` must be the same as image dimension')
            patches_corners_img = self.generate_patches_corners_for_image(self.patch_size, self.stride, img_shape)
            self.patches_corners.extend(zip([img_idx]*len(patches_corners_img), patches_corners_img))

    def generate_patches_corners_for_image(self, patch_size, stride, img_shape):

        if len(img_shape)==2:
            patches_corners = self.generate_patches_corners_for_2d_image(patch_size, stride, img_shape)
        elif len(img_shape)==3:
            patches_corners = self.generate_patches_corners_for_3d_image(patch_size, stride, img_shape)
        else:
            raise Exception('Image must be 2D or 3D')

        return patches_corners

    def generate_patches_corners_for_2d_image(self, patch_size, stride, img_shape):

        #patch_count = 0
        patches_corners = []
        for row in range(0, img_shape[0]-patch_size[0]+stride[0], stride[0]):
            if (row+patch_size[0])>=img_shape[0]:
                # Do not go over image border
                row = img_shape[0] - patch_size[0]
            for col in range(0, img_shape[1]-patch_size[1]+stride[1], stride[1]):
                if (col+patch_size[1])>=img_shape[1]:
                    # Do not go over image border
                    col = img_shape[1] - patch_size[1]

                patch_corners = (slice(row, row+patch_size[0]), slice(col, col+patch_size[1]))
                patches_corners.append(patch_corners)
                #patch_count += 1
        #self.patch_index_accumulator.append(self.patch_index_accumulator[-1] + patch_count)
        return patches_corners

    def generate_patches_corners_for_3d_image(self, patch_size, stride, img_shape):

        #patch_count = 0
        patches_corners = []
        for plane in range(0, img_shape[0]-patch_size[0]+stride[0], stride[0]):
            if (plane+patch_size[0])>=img_shape[0]:
                # Do not go over image border
                plane = img_shape[0] - patch_size[0]
            for row in range(0, img_shape[1]-patch_size[1]+stride[1], stride[1]):
                if (row+patch_size[1])>=img_shape[1]:
                    # Do not go over image border
                    row = img_shape[1] - patch_size[1]
                for col in range(0, img_shape[2]-patch_size[2]+stride[2], stride[2]):
                    if (col+patch_size[2])>=img_shape[2]:
                        # Do not go over image border
                        col = img_shape[2] - patch_size[2]

                    patch_corners = (slice(plane, plane+patch_size[0]), slice(row, row+patch_size[1]),
                                     slice(col, col+patch_size[2]))
                    patches_corners.append(patch_corners)

        return patches_corners

    def get_patch_from_index(self, index, img_shape):
        pass

    def get_shape(self, img, warn=True):

        if isinstance(img, Image.Image):
            img_shape = (img.height, img.width)
        elif isinstance(img, torch.Tensor):
            img_shape = img.shape
            if (img.ndim==3):
                if img_shape[-3]<=3:
                    # Consider that third to last dimension is for color
                    img_shape = img_shape[-2:]
                else:
                    img_shape = img_shape[-3:]
            if (img.ndim==4):
                img_shape = img_shape[-3:]
        elif isinstance(img, np.ndarray):
            img_shape = img.shape
            if img.ndim==3:
                if img_shape[-1]<=3:
                    # Consider that last dimension is for color
                    img_shape = img_shape[-3:-1]
                else:
                    img_shape = img_shape[-3:]
            elif img.ndim==4:
                img_shape = img_shape[-3:]
        else:
            raise AttributeError("Image is not a PIL, Tensor or ndarray. Cannot safely infer shape")

        if min(img_shape)<=3:
            print(f'Warning, inferred shape {img_shape} is probably incorrect. Sizes smaller than 4 are being discarded')
            img_shape = filter(lambda v:v>3, img_shape)

        return img_shape

    def crop_img(self, img, patch_corners):
        '''TODO: Decide if we should include imgaug crop. Otherwise crop will not work if last
        transform of self.img_transforms is from imgaug.'''

        if isinstance(img, Image.Image):
            first_row, last_row = patch_corners[0].start, patch_corners[0].stop
            first_col, last_col = patch_corners[1].start, patch_corners[1].stop
            img_patch = img.crop([first_col, first_row, last_col, last_row])
        if isinstance(img, torch.Tensor):
            # Crop from trailing dimensions
            img_patch = img[(...,)+patch_corners]
        else:
            try:
                img_patch = img[patch_corners]
            except Exception:
                raise IndexError(f'Cannot crop image of type {type(img)}')

        #import pdb; pdb.set_trace()

        return img_patch

    @classmethod
    def get_image_from_patches(cls, patches, stride, img_shape, operation='max'):

        # Remove single channel dimension
        if patches[0].ndim==4:
            is_3d = True
        else:
            is_3d = False

        if not isinstance(patches[0], torch.Tensor):
            patches = [transforms.transf_to_tensor(patch, is_3d=is_3d) for patch in patches]

        #if patches[0].shape[0]==1:
        #    patches = [patch[0] for patch in patches]

        patch_size = patches[0].shape[1:]
        img = torch.zeros(img_shape, dtype=patches[0].dtype)
        patches_corners = cls.generate_patches_corners_for_image(patch_size, stride, img_shape)

        if operation=='mean':
            img_count = torch.zeros(img_shape, dtype=int)
        for patch_corners, patch in zip(patches_corners, patches):
            img_patch = img[patch_corners]
            if operation=='max':
                img_patch[:] = torch.where(img_patch>patch, img_patch, patch)
            elif operation=='min':
                img_patch[:] = torch.where(img_patch<patch, img_patch, patch)
            elif operation=='mean':
                img_patch[:] = img_patch + patch
                img_count[patch_corners] += 1

        if operation=='mean':
            mask = img_count>0
            img[mask] = img[mask]/img_count[mask]

        return img

    # Functions with some ideas, not necessary for class
    def crop_2d_img(self, img, patch_corners):

        first_row, first_col, last_row, last_col = patch_corners
        if isinstance(img, Image.Image):
            img_patch = img.crop([first_col, first_row, last_col, last_row])
        else:
            img_patch = img[first_row:last_row, first_col:last_col]

        return img_patch

    def crop_3d_img(self, img, patch_corners):

        first_plane, first_row, first_col, last_plane, last_row, last_col = patch_corners

        if isinstance(img, Image.Image):
            raise ValueError('Cannot interpret PIL image as 3D')
        else:
            img_patch = img[first_plane:last_plane, first_row:last_row, first_col:last_col]

        return img_patch

    def generate_patch_index_accumulator(self, patch_size, stride=None, img_shape=None):
        '''If img_shape is None, generates indices by opening each image to get the
        respective shape. This is useful when images have distinct sizes. If img_shape
        is not None, uses that shape and the images are not opened, which is much faster.'''

        if img_shape is None:
            must_open = True
        else:
            must_open = False

        index_accumulator = [0]
        img_shapes = []
        for img_file_path in self.img_file_paths:
            if must_open:
                try:
                    img = self.img_opener(img_file_path)
                except Exception:
                    raise Exception(f'Cannot open image {img_file_path}\n')

            img_shape = self.get_shape(img)
            num_patches = self._num_patches_in_img(patch_size, stride, img_shape)

            img_shapes.append(img_shape)
            index_accumulator.append(index_accumulator[-1] + num_patches)

        self.img_shapes = img_shapes
        self.patch_index_accumulator = index_accumulator

    def get_patch_from_global_index(self, index):

        img_index = bisect.bisect(self.patch_index_accumulator, index) - 1
        img_shape = self.img_shapes[img_index]
        patch_index = index - self.patch_index_accumulator[img_index]
        get_patch_from_index(self, patch_index, img_shape)

    def _num_patches_in_img(self, patch_size, stride, img_shape):

        num_p_rows = (img_shape[0]-patch_size[0])//stride[0] + 1
        if num_p_rows*stride[0]!=img_shape[0]:
            # If patches do not fit perfectly
            num_p_rows += 1
        num_p_cols = (img_shape[1]-patch_size[1])//stride[1] + 1
        if num_p_cols*stride[1]!=img_shape[1]:
            # If patches do not fit perfectly
            num_p_cols += 1

        return num_p_rows*num_p_cols
