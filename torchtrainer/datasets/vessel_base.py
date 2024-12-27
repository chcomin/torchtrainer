"""The objective of the dataset classes here is to provide a minimal code to load the
images from the respective datasets."""

import os
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from ..util.data_util import search_files


class RetinaDataset(Dataset):
    """Create a dataset object for holding a typical retina blood vessel dataset. 

    Note: __getitem__ returns a numpy array and not a pillow image because pillow
    does not support a negative ignore index (e.g. -100).
    """

    _HAS_TEST = None

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        channels: str = "all",
        keepdim: bool = False,
        return_mask: bool = False,
        ignore_index: int | None = None,
        normalize: bool = True,
        files: list = None,
        transforms: Callable | None = None,
    ):
        """
        root
            Root directory.
        split
            The split to use. Possible values are "train", "test" and "all"
        channels
            Image channels to use. Options are:
            "all": Use all channels
            "green": Use only the green channel
            "gray": Convert the image to grayscale
        keepdim
            If True, keeps the channel dimension in case `channels` is "green" or "gray"
        return_mask
            If True, also returns the retina mask
        ignore_index
            Index to put at the labels for pixels outside the mask (the retina). 
            If None, do nothing.
        normalize
            If True, divide the labels by 255 in case label.max()==255.
        files
            List of files to keep from the split. If None, use all files.
        transforms
            Transformations to apply to the images and the labels. If `return_mask` 
            is True, the transform needs to also accept the mask image as input.
        """
        
        self.root = Path(root)

        if split not in ["train", "test", "all"]:
            raise ValueError("Invalid split value. Must be 'train', 'test' or 'all'.")
        if channels not in ["all", "green", "gray"]:    
            raise ValueError("Invalid channels value. Must be 'all', 'green' or 'gray'.")
        
        if split=="test" and not self._HAS_TEST:
            raise ValueError("This dataset does not have a test split.")

        if split=="all":
            images, labels, masks = self._get_files(split="train")
            if self._HAS_TEST:
                images_t, labels_t, masks_t = self._get_files(split="test")
                images += images_t
                labels += labels_t
                masks += masks_t
        else:
            images, labels, masks = self._get_files(split=split)

        # Filter files if needed
        if files is not None:
            indices = search_files(files, images)
            images = [images[idx] for idx in indices]
            labels = [labels[idx] for idx in indices]
            masks = [masks[idx] for idx in indices]

        self.channels = channels
        self.keepdim = keepdim
        self.return_mask = return_mask
        self.ignore_index = ignore_index
        self.normalize = normalize

        self.images = sorted(images)
        self.labels = sorted(labels)
        self.masks = sorted(masks)
        self.classes = ["background", "vessel"]
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple:
        
        image = Image.open(self.images[idx])
        label = np.array(Image.open(self.labels[idx]), dtype=int)
        mask = np.array(Image.open(self.masks[idx]))

        # Select green channel or convert to gray
        if self.channels=="gray":
            image = image.convert('L')
        image = np.array(image)
        if self.channels=="green":
            image = image[:,:,1]
        if self.keepdim and image.ndim==2:
            image = np.expand_dims(image, axis=2)

        # Normalize label to [0,1] if in range [0,255]
        if self.normalize and label.max()==255:
            label = label//255
            mask = mask//255

        # Keep only first label channel if it is a color image
        if label.ndim==3:
            diff_pix = (label[:,:,0]!=label[:,:,1]).sum()
            diff_pix += (label[:,:,0]!=label[:,:,2]).sum()
            if diff_pix>0:
                raise ValueError("Label has multiple channels and they differ.")
            label = label[:,:,0]

        # Put ignore_index outside mask
        if self.ignore_index is not None:
            label[mask==0] = self.ignore_index

        output = image, label
        # Remember to also transform mask
        if self.return_mask:
            output += (mask,)
        if self.transforms is not None:
            output = self.transforms(*output)

        return output

    def __len__(self) -> int:
        return len(self.images)
    
    def _get_files(self, split: str):

        raise NotImplementedError

class DRIVE(RetinaDataset):
    """Create a dataset object for holding the DRIVE data. The dataset must 
    be organized as
    root
      training
        images
        1st_manual
        mask
      test
        images
        1st_manual
        mask

    See the RetinaDataset docs for an explanation of the parameters.
    """

    _HAS_TEST = True
    
    def _get_files(self, split: str) -> tuple[list, list, list]:

        if split=="train":
            root_split = self.root/"training"
            mask_str = "training"
        elif split=="test":
            root_split = self.root/"test"
            mask_str = "test"

        root_imgs = root_split/"images"
        root_labels = root_split/"1st_manual"
        root_masks = root_split/"mask"

        files = os.listdir(root_imgs)
        images = []
        labels = []
        masks = []
        for file in files:

            num, _ = file.split('_')
            images.append(root_imgs/file)
            labels.append(root_labels/f"{num}_manual1.gif")
            masks.append(root_masks/f"{num}_{mask_str}_mask.gif")

        return images, labels, masks
    
class CHASEDB1(RetinaDataset):
    """Create a dataset object for holding the CHASEDB1 data. The dataset must 
    be organized as
    root
      images
      labels
      mask

    See the RetinaDataset docs for an explanation of the parameters. Note that
    the CHASE dataset does not have a train/test split.
    """

    _HAS_TEST = False
    
    def _get_files(self, split: str) -> tuple[list, list, list]:

        root_imgs = self.root/"images"
        root_labels = self.root/"labels"
        root_masks = self.root/"mask"

        files = os.listdir(root_imgs)
        images = []
        labels = []
        masks = []
        for file in files:

            filename, _ = file.split('.')
            images.append(root_imgs/file)
            labels.append(root_labels/f"{filename}_1stHO.png")
            masks.append(root_masks/f"{filename}.png")

        return images, labels, masks
    
class STARE(RetinaDataset):
    """Create a dataset object for holding the STARE data. The dataset must 
    be organized as
    root
      images
      labels
      mask

    See the RetinaDataset docs for an explanation of the parameters. Note that
    the STARE dataset does not have a train/test split.
    """

    _HAS_TEST = False
    
    def _get_files(self, split: str) -> tuple[list, list, list]:

        root_imgs = self.root/"images"
        root_labels = self.root/"labels"
        root_masks = self.root/"mask"

        files = os.listdir(root_imgs)
        images = []
        labels = []
        masks = []
        for file in files:

            filename, _ = file.split('.')
            images.append(root_imgs/file)
            labels.append(root_labels/f"{filename}.ah.png")
            masks.append(root_masks/f"{filename}.png")

        return images, labels, masks
    
class HRF(RetinaDataset):
    """Create a dataset object for holding the HRF data. The dataset must 
    be organized as
    root
      images
      labels
      mask

    See the RetinaDataset docs for an explanation of the parameters. Note that
    the HRF dataset does not have a train/test split.
    """

    _HAS_TEST = False
    
    def _get_files(self, split: str) -> tuple[list, list, list]:

        root_imgs = self.root/"images"
        root_labels = self.root/"labels"
        root_masks = self.root/"mask"

        files = os.listdir(root_imgs)
        images = []
        labels = []
        masks = []
        for file in files:

            filename, _ = file.split('.')
            images.append(root_imgs/file)
            labels.append(root_labels/f"{filename}.tif")
            masks.append(root_masks/f"{filename}_mask.tif")

        return images, labels, masks
    
class FIVES(RetinaDataset):
    """Create a dataset object for holding the FIVES data. The dataset must 
    be organized as
    root
      train
        images
        labels
      test
        images
        labels
    mask.png

    See the RetinaDataset docs for an explanation of the parameters.
    """

    _HAS_TEST = True
    
    def _get_files(self, split: str) -> tuple[list, list, list]:

        if split=="train":
            root_split = self.root/"train"
        elif split=="test":
            root_split = self.root/"test"

        root_imgs = root_split/"images"
        root_labels = root_split/"labels"

        files = os.listdir(root_imgs)
        images = []
        labels = []
        masks = []
        for file in files:

            filname, _ = file.split('.')
            images.append(root_imgs/file)
            labels.append(root_labels/f"{filname}.png")
            masks.append(self.root/"mask.png")

        return images, labels, masks
    
class VessMAP(Dataset):
    """Create a dataset object for holding the VessMAP data. 
    """
    def __init__(
        self,
        root: str | Path,
        keepdim: bool = False,
        normalize: bool = True,
        files: list = None,
        transforms: Callable | None = None,
    ):
        """
        root
            Root directory.
        keepdim
            If True, keeps the channel dimension of the image
        normalize
            If True, divide the labels by 255 in case label.max()==255.
        files
            List of dataset files to use. If None, use all files.
        transforms
            Transformations to apply to the images and the labels.
        """
        
        self.root = Path(root)

        images, labels = self._get_files()

        # Filter files if needed
        if files is not None:
            indices = search_files(files, images)
            images = [images[idx] for idx in indices]
            labels = [labels[idx] for idx in indices]

        self.keepdim = keepdim
        self.normalize = normalize

        self.images = sorted(images)
        self.labels = sorted(labels)
        self.classes = ["background", "vessel"]
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple[NDArray, NDArray]:
            
        image = np.array(Image.open(self.images[idx]))
        label = np.array(Image.open(self.labels[idx]), dtype=int)

        if self.keepdim and image.ndim==2:
            image = np.expand_dims(image, axis=2)

        # Normalize label to [0,1] if in range [0,255]
        if self.normalize and label.max()==255:
            label = label//255

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)
    
    def _get_files(self) -> tuple[list, list]:

        root_imgs = self.root/"images"
        root_labels = self.root/"annotator1"/"labels"

        files = os.listdir(root_imgs)
        images = []
        labels = []
        for file in files:

            filename, _ = file.split('.')
            images.append(root_imgs/file)
            labels.append(root_labels/f"{filename}.png")

        return images, labels
    
class CORTEX(Dataset):
    """Create a dataset object for holding the CORTEX data. """
    def __init__(
        self,
        root: str | Path,
        keepdim: bool = False,
        normalize: bool = True,
        files: list = None,
        transforms: Callable | None = None,
    ):
        """
        root
            Root directory.
        keepdim
            If True, keeps the channel dimension of the image
        normalize
            If True, divide the labels by 255 in case label.max()==255.
        files
            List of dataset files to use. If None, use all files.
        transforms
            Transformations to apply to the images and the labels.
        """
        
        self.root = Path(root)

        images, labels = self._get_files()

        # Filter files if needed
        if files is not None:
            indices = search_files(files, images)
            images = [images[idx] for idx in indices]
            labels = [labels[idx] for idx in indices]

        self.keepdim = keepdim
        self.normalize = normalize

        self.images = sorted(images)
        self.labels = sorted(labels)
        self.classes = ["background", "vessel"]
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple[NDArray, NDArray]:
            
        image = np.array(Image.open(self.images[idx]))
        label = np.array(Image.open(self.labels[idx]), dtype=int)

        if self.keepdim and image.ndim==2:
            image = np.expand_dims(image, axis=2)

        # Normalize label to [0,1] if in range [0,255]
        if self.normalize and label.max()==255:
            label = label//255

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)
    
    def _get_files(self) -> tuple[list, list]:

        root_imgs = self.root/"images"
        root_labels = self.root/"labels"

        files = os.listdir(root_imgs)
        images = []
        labels = []
        for file in files:

            filename, _ = file.split('.')
            images.append(root_imgs/file)
            labels.append(root_labels/f"{filename}.png")

        return images, labels
    

