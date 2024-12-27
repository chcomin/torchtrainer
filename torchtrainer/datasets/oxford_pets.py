import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from ..util.train_util import Subset


class OxfordIIITPetSeg(Dataset):
    """Oxford Pets segmentation dataset."""

    def __init__(self, root, transforms=None, ignore_val=2):
        """

        Parameters
        ----------
        root
            Dataset root folder
        transforms
            Transformations to be applied to the images and targets
        ignore_val
            Value to be used for ignored pixels
        """

        root = Path(root)
        images_folder = root / "images"
        segs_folder = root / "annotations/trimaps"
        anns_file = root / "annotations/list.txt"

        images = []
        segs = []
        with open(anns_file) as file:
            for line in file.read().splitlines():
                if line[0]!="#":   # Discards file comments
                    name, class_id, species_id, breed_id = line.strip().split()
                    images.append(images_folder/f"{name}.jpg")
                    segs.append(segs_folder/f"{name}.png")

        self.classes = ("Background", "Foreground")
        self.images = images
        self.segs = segs
        self.transforms = transforms
        self.ignore_val = ignore_val

    def __getitem__(self, idx, apply_transform=True):

        image = Image.open(self.images[idx]).convert("RGB")
        target_or = Image.open(self.segs[idx])

        # Oxford Pets uses 1 for foreground, 2 for background and 3 for undefined pixels
        # Change 2->0 and 3->ignore_val
        target_np = np.array(target_or)
        target_np[target_np==2] = 0

        if self.ignore_val!=3:
            target_np[target_np==3] = self.ignore_val

        target = Image.fromarray(target_np, mode="L")

        if self.transforms and apply_transform:
            image, target = self.transforms(image, target)

        return image, target
    
    def __len__(self):
        return len(self.images)

class TransformsTrain:

    def __init__(self, resize_size=(384, 384)):
    
        transforms = transf.Compose([
            transf.PILToTensor(),   
            transf.RandomResizedCrop(size=resize_size, scale=(0.5,1.), 
                                     ratio=(0.9,1.1), antialias=True),
            #transf.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.01),
            transf.RandomHorizontalFlip(),
            transf.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}),
            transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0))
        ])

        self.transforms = transforms

    def __call__(self, img, target):
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data
        target = target.squeeze()
        return img, target

class TransformsEval:

    def __init__(self, resize_size=(384, 384)):

        transforms = transf.Compose([
            transf.PILToTensor(),   
            transf.Resize(size=resize_size, antialias=True),
            transf.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}),
            transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0))
        ])

        self.transforms = transforms

    def __call__(self, img, target):
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data
        target = target.squeeze()
        return img, target
    
def cat_list(images, fill_value=0):
    """Concatenate a list of images into a single tensor.

    Parameters
    ----------
    images
        List of images
    fill_value, optional
       How to pad the images
    Returns
    -------
        Batched images as a tensor
    """

    is_target = images[0].ndim==2

    num_rows, num_cols = zip(*[img.shape[-2:] for img in images])
    r_max, c_max = max(num_rows), max(num_cols)
    if is_target:
        batch_shape = (len(images), r_max, c_max)
    else:
        batch_shape = (len(images), 3, r_max, c_max)

    batched_imgs = torch.full(batch_shape, fill_value, dtype=images[0].dtype)
    for idx in range(len(images)):
        img = images[idx]
        if is_target:
            batched_imgs[idx, :img.shape[0], :img.shape[1]] = img
        else:
            batched_imgs[idx, :, :img.shape[1], :img.shape[2]] = img

    return batched_imgs

def collate_fn(batch, img_fill=0, target_fill=2):

    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=img_fill)
    batched_targets = cat_list(targets, fill_value=target_fill)

    return batched_imgs, batched_targets

def unormalize(img):
    img = img.permute(1, 2, 0)
    mean = torch.tensor([122.7, 114.6, 100.9])
    std = torch.tensor([59.2, 58.4, 59.0])
    img = img*std + mean
    img = img.to(torch.uint8)

    return img

def get_dataset(dataset_path, split=0.2, resize_size=(384, 384)):

    ds = OxfordIIITPetSeg(dataset_path)
    n = len(ds)
    n_valid = int(n*split)

    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    
    ds_train = Subset(ds, indices[n_valid:], TransformsTrain(resize_size))
    ds_valid = Subset(ds, indices[:n_valid], TransformsEval(resize_size))

    class_weights = (0.33, 0.67)
    ignore_index = 2

    return ds_train, ds_valid, class_weights, ignore_index, collate_fn