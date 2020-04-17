'''
Dataset storage class
'''

from pathlib import Path
from PIL import Image
import random
from torch.utils.data import dataset as torch_dataset
import torch

class ImageDataset(torch_dataset.Dataset):
    """Dataset storage class.

    Receives an input image and label directory and stores respective images. Images can be
    retrieved as follows:

    image_ds = ImageDataset(...)
    img, label = image_ds[0]

    or, in case weight_func is not None:

    img, label, weight = image_ds[0]

    Parameters
    ----------
    img_dir : string or pathlib path
        Directory containing the images to be read
    label_dir : string or pathlib path
        Directory containing the labels (segmentations) to be read
    name_2_label_map : function
        Function with signature name_2_label_map(img_filename) that translates image filenames into
        labels filenames. Receives an image filename and returns the filename of an image containing
        the respective label
    filename_filter : list or function
        If list, contains names of the image files that should be kept, other images are ignored
        If function, has signature filename_filter(img_filename), receives an image filename and returns
        True if the image should be kept. The image is discarded otherwise
    img_opener : function
        Function with signature img_opener(img_path) for opening the images. Receives an image path
        and returns a PIL.Image object. Images should be have uint8 type.
    label_opener: function
        Function with signature label_opener(label_path) for opening the labels. Receives an label path
        and returns a PIL.Image object. The image should contain class indices and have uint8 type
    transforms : list of functions
        List of functions to be applied for image augmentation. Each function should have the signature
        transform(img, label, weight=None) and return a tuple (img, label) in case `weight_func` is None
        or (img, label, weight) otherwise. The arguments of the first transform should be PIL.Image objects,
        while the return values of the last transform should be float32 torch.Tensor objects.
    weight_func : function
        Function for generating weights associated to each image. Those can be used for defining masks
        or, for instance, weighting the loss function. Must have signature
        weight_func(pil_img, pil_label, img_path=None) and return a PIL.Image with F (float32) type
    """

    def __init__(self, img_dir, label_dir, name_2_label_map, filename_filter=None, img_opener=None,
                 label_opener=None, transforms=None, weight_func=None):

        if isinstance(img_dir, str):
            img_dir = Path(img_dir)
        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        if isinstance(filename_filter, list):
            filename_filter = set(filename_filter)
        if transforms is None:
            transforms = []
        if img_opener is None:
            img_opener = Image.open
        if label_opener is None:
            label_opener = Image.open

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.name_2_label_map = name_2_label_map
        self.img_opener = img_opener
        self.label_opener = label_opener
        self.transforms = transforms
        self.weight_func = weight_func

        img_file_paths = []
        for img_file_path in img_dir.iterdir():
            img_filename = img_file_path.name
            if filename_filter is None:
                img_file_paths.append(img_file_path)
            elif isinstance(filename_filter, set):
                if img_file_path.stem in filename_filter: img_file_paths.append(img_file_path)
            elif filename_filter(img_filename):
                img_file_paths.append(img_file_path)

        self.img_file_paths = img_file_paths

    def __getitem__(self, idx):
        '''Returns one item from the dataset. Will return an image and label if weight_func was
        not defined during class instantiation or an aditional weight image otherwise.'''

        img_file_path = self.img_file_paths[idx]

        img = self.img_opener(img_file_path)
        label_file_path = self.label_path_from_image_path(img_file_path)
        label = self.label_opener(label_file_path)

        if self.weight_func is not None:
            weight = self.weight_func(img, label, img_file_path)
            ret_transf = self.apply_transforms(img, label, weight)
        else:
            ret_transf = self.apply_transforms(img, label)

        return ret_transf

    def __len__(self):

        return len(self.img_file_paths)

    def label_path_from_image_path(self, img_file_path):
        '''Translates image path to label path.'''

        img_filename = img_file_path.name
        return self.label_dir/self.name_2_label_map(img_filename)

    def check_dataset(self):
        '''Check if all images in the dataset can be read, and if the transformations
        can be successfully applied. It is usefull to call this function right after
        dataset creation.
        '''

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
                if len(shapes)<(idx+1):
                    shapes.append(ret_val.shape)
                elif ret_val.shape!=shapes[idx]:
                    raise Exception(f"Data has different shape at index {img_idx}")

        print('All images read')

    def split_train_val(self, valid_set=0.2):
        '''Split dataset into train and validation. Returns two new datasets.

        Parameters
        ----------
        valid_set : float or list
            If float, a fraction `valid_set` of the dataset will be used for validation,
            the rest will be used for training.
            If list, should containg the names of the files used for validation. The remaining
            images will be used for training.

        Returns
        -------
        train_dataset : ImageDataset
            Dataset to be used for training
        valid_dataset : ImageDataset
            Dataset to be used for validation
        '''

        img_file_paths_train, img_file_paths_valid = self.split_train_val_paths(valid_set)
        # Hacky way to get parameters passed to __init__ during class construction
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
        init_pars_valid['filename_filter'] = img_file_paths_valid

        train_dataset = ImageDataset(**init_pars_train)
        valid_dataset = ImageDataset(**init_pars_valid)

        return train_dataset, valid_dataset

    def split_train_val_paths(self, valid_set=0.2):
        '''Generates image names to be used for spliting the dataset.

        Parameters
        ----------
        valid_set : float or list
            If float, a fraction `valid_set` of the dataset will be used for validation,
            the rest will be used for training.
            If list, should containg the names of the files used for validation. The remaining
            images will be used for training.

        Returns
        -------
        img_file_paths_train : list
            Images used for training
        img_file_paths_valid : list
            Images used for validation
        '''

        img_file_paths = self.img_file_paths
        num_images = len(img_file_paths)

        img_file_paths_train = []
        img_file_paths_valid = []

        if isinstance(valid_set, list):

            valid_set_set = set(valid_set)
            for file_idx, img_file_path in enumerate(img_file_paths):
                if img_file_path.stem in valid_set_set:
                    img_file_paths_valid.append(img_file_path)
                else:
                    img_file_paths_train.append(img_file_path)

            if (len(img_file_paths_train)+len(img_file_paths_valid))!=len(img_file_paths):
                print('Warning, some files in validation set not found')

        elif isinstance(valid_set, float):
            num_images_valid = int(num_images*valid_set)
            num_images_train = num_images - num_images_valid

            ind_all = list(range(num_images))
            random.shuffle(ind_all)
            ind_train = ind_all[0:num_images_train]
            ind_valid = ind_all[num_images_train:]

            img_file_paths_train = [img_file_paths[ind] for ind in ind_train]
            img_file_paths_valid = [img_file_paths[ind] for ind in ind_valid]

        img_file_paths_train = [file.stem for file in img_file_paths_train]
        img_file_paths_valid = [file.stem for file in img_file_paths_valid]

        return img_file_paths_train, img_file_paths_valid

    def as_tensor(self):
        '''Converts all images in the dataset to a single torch tensor.

        Returns
        -------
        tensors : torch.Tensor
            Tensor with dimensions (num images, num channels, height, width)
        '''

        img_file_paths = self.img_file_paths
        ret_vals = self.__getitem__(0)           # Open first image to get shape

        num_tensors = len(img_file_paths)
        tensors = [torch.zeros((num_tensors, *val.shape), dtype=val.dtype) for val in ret_vals]

        for file_idx, img_file_path in enumerate(img_file_paths):
            ret_vals = self.__getitem__(file_idx)
            for idx, ret_val in enumerate(ret_vals):
                tensors[idx][file_idx] = ret_val

        return tensors

    def apply_transforms(self, img, label, weight=None):
        '''Apply transformations stored in self.transforms

        Parameters
        ----------
        img, label, weight : Image-like
            Images to be processed

        Returns
        -------
        vals : Image-like
            Resulting images. Either (img, label) if weight is None or (img, label, weight) otherwise
        '''

        if weight is None:
            vals = [img, label]
        else:
            vals = [img, label, weight]
        for transform in self.transforms:
                vals = transform(*vals)
        return vals