"""The objective of the dataset classes here is to provide the minimal code to load the
images from the respective datasets. """

from pathlib import Path
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from ..datasets.vessel_base import DRIVE
from ..util.train_util import Subset


import os.path as osp
import pandas as pd
from skimage import measure
import torch
from torchvision.transforms import v2 as tv_transf
from torchvision.transforms.v2 import functional as tv_transf_F
from torchvision import tv_tensors

class TrainDataset(Dataset):
    def __init__(self, csv_path, transforms=None, channels='all'):
        df = pd.read_csv(csv_path)
        self.root = osp.dirname(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths
        self.mask_list = df.mask_paths
        self.transforms = transforms
        self.channels = channels
        self.label_values = (0, 255)  # for use in label_encoding

    def label_encoding(self, gdt):
        gdt_gray = np.array(gdt.convert('L'))
        classes = np.arange(len(self.label_values))
        for i in classes:
            gdt_gray[gdt_gray == self.label_values[i]] = classes[i]
        return Image.fromarray(gdt_gray)

    def crop_to_fov(self, img, target, mask):
        minr, minc, maxr, maxc = measure.regionprops(np.array(mask))[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        tg_crop = Image.fromarray(np.array(target)[minr:maxr, minc:maxc])
        mask_crop = Image.fromarray(np.array(mask)[minr:maxr, minc:maxc])
        return im_crop, tg_crop, mask_crop

    def __getitem__(self, index):
        # load image and labels
        img = Image.open(osp.join(self.root,self.im_list[index]))
        target = Image.open(osp.join(self.root,self.gt_list[index]))
        mask = Image.open(osp.join(self.root,self.mask_list[index])).convert('L')

        if self.channels=='gray':
            img = img.convert('L')
        elif self.channels=='green':
            img = Image.fromarray(np.array(img)[:,:,1])

        img, target, mask = self.crop_to_fov(img, target, mask)

        target = np.array(self.label_encoding(target))

        target[np.array(mask) == 0] = 0
        target = Image.fromarray(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # QUICK HACK FOR PSEUDO_SEG IN VESSELS, BUT IT SPOILS A/V
        if len(self.label_values)==2: # vessel segmentation case
            target = target.float()
            if torch.max(target)>1:
                target= target.float()/255

        return img, target

    def __len__(self):
        return len(self.im_list)

class TrainTransforms:

    def __init__(self, tg_size):

        self.tg_size = tg_size

        scale = tv_transf.RandomAffine(degrees=0, scale=(0.95, 1.20))
        transl = tv_transf.RandomAffine(degrees=0, translate=(0.05, 0))
        rotate = tv_transf.RandomRotation(degrees=45)
        scale_transl_rot = tv_transf.RandomChoice((scale, transl, rotate))

        #brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
        #jitter = tv_transf.ColorJitter(brightness, contrast, saturation, hue)

        hflip = tv_transf.RandomHorizontalFlip()
        vflip = tv_transf.RandomVerticalFlip()

        to_dtype = tv_transf.ToDtype(
            {
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64
            },
            scale=True   # Mask is not scaled
        )

        unwrap = tv_transf.ToPureTensor()

        self.transform = tv_transf.Compose((
            scale_transl_rot,
            #jitter,
            hflip,
            vflip,
            to_dtype,
            unwrap
        ))

    def __call__(self, img, target):

        img = torch.from_numpy(img)
        target = torch.from_numpy(target)

        img = tv_transf_F.resize(img, self.tg_size)
        # NEAREST_EXACT has a 0.01 better Dice score than NEAREST. The
        # object oriented version of resize uses NEAREST, thus we need to use
        # the functional interface
        target = tv_transf_F.resize(target, self.tg_size, interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)

        img, target = self.transform(img, target)
        target = target[0]

        return img, target

class ValidTransforms:

    def __init__(self, tg_size):
        
        self.tg_size = tg_size

        to_dtype = tv_transf.ToDtype(
            {
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64
            },
            scale=True   # Mask is not scaled
        )

        unwrap = tv_transf.ToPureTensor()

        self.transform = tv_transf.Compose((
            to_dtype,
            unwrap
        ))

    def __call__(self, img, target):

        img = torch.from_numpy(img)
        target = torch.from_numpy(target)
        
        img = tv_transf_F.resize(img, self.tg_size)
        # NEAREST_EXACT has a 0.01 better Dice score than NEAREST. The
        # object oriented version of resize uses NEAREST, thus we need to use
        # the functional interface
        target = tv_transf_F.resize(target, self.tg_size, interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)

        img, target = self.transform(img, target)
        target = target[0]

        return img, target


def get_dataset_drive_train(dataset_path, split_strategy="train_0.2", resize_size=(512, 512), channels="all"):
    """Get the DRIVE dataset for training.
    Parameters
    ----------
    dataset_path
        Path to the dataset root folder
    split_strategy
        Strategy to split the dataset. Possible values are:
        "train_<split>": Use <split> fraction of the train images to validate
        "use_test": Use the test images of the dataset for validation
        "file": Use the train.csv and val.csv files to split the dataset
    resize_size
        Size to resize the images
    channels
        Image channels to use. Options are:
        "all": Use all channels
        "green": Use only the green channel
        "gray": Convert the image to grayscale
    """

    class_weights = (0.13, 0.87)
    ignore_index = 2
    collate_fn = None

    dataset_path = Path(dataset_path)

    drive_params = {
        'channels':channels, 'keepdim':True, 'ignore_index':ignore_index
    }
    if "file" in split_strategy:
        files_train = open(dataset_path/'train.csv').read().splitlines()
        files_valid = open(dataset_path/'val.csv').read().splitlines()
        ds_train = DRIVE(dataset_path, files=files_train, **drive_params)
        ds_valid = DRIVE(dataset_path, files=files_valid, **drive_params)

    elif "train" in split_strategy:
        ds = DRIVE(dataset_path, **drive_params)
        split = float(split_strategy.split("_")[1])
        n = len(ds)
        n_valid = int(n*split)

        indices = list(range(n))
        random.shuffle(indices)
        
        class_atts = {
            'images':ds.images, 'labels':ds.labels, 'masks':ds.masks, 'classes':ds.classes
        }
        ds_train = Subset(ds, indices[n_valid:], **class_atts)
        ds_valid = Subset(ds, indices[:n_valid], **class_atts)

    elif split_strategy=="use_test":
        ds_train = DRIVE(dataset_path, split="train", **drive_params)
        ds_valid = DRIVE(dataset_path, split="test", **drive_params)

    ds_train.transforms = TrainTransforms(resize_size)
    ds_valid.transforms = ValidTransforms(resize_size)

    return ds_train, ds_valid, class_weights, ignore_index, collate_fn