"""Classes for creating artificial data for testing purposes."""

import random

import torch
from skimage import draw
from torch.utils.data import Dataset


class FakeClassificationData(Dataset):
    """Dataset that returns a circle or disk image. Used for testing purposes."""

    def __init__(self, img_size=224, n_samples=100, transforms=None):
        
        half = img_size//2
        quarter = img_size//4
        
        img_square = torch.zeros((img_size, img_size), dtype=torch.uint8)
        inds = draw.rectangle((quarter, quarter), extent=(half, half))
        img_square[inds] = 255
        img_square = img_square.tile((3,1,1)).permute(1, 2, 0)

        img_disk = torch.zeros((img_size, img_size), dtype=torch.uint8)
        inds = draw.disk((half, half), quarter)
        img_disk[inds] = 255
        img_disk = img_disk.tile((3,1,1)).permute(1, 2, 0)

        self.imgs = [img_square.numpy(), img_disk.numpy()]
        self.n_samples = n_samples
        self.transforms = transforms

    def __getitem__(self, idx):
        """Ignore index and return a random image."""
        idx = random.randint(0, 1)
        img = self.imgs[idx]
        if self.transforms is not None:
            img = self.transforms(img)

        return img, idx
    
    def __len__(self):
        return self.n_samples

class FakeSegmentationData(FakeClassificationData):
    """Segmentation dataset that returns a circle or disk image. Used for testing purposes."""

    def __getitem__(self, idx):
        idx = random.randint(0, 1)
        img = self.imgs[idx]
        target = img[:,:,0]//255
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
class Transforms:
    """Simple tranform for the fake data."""

    def __call__(self, img, target):
        
        img = torch.from_numpy(img).permute(2, 0, 1).float()/255
        target = torch.from_numpy(target).to(torch.int64)

        return img, target

def get_dataset(img_size=224):
    """Return a fake segmentation dataset."""

    transforms = Transforms()

    ds_train = FakeSegmentationData(img_size, n_samples=80, transforms=transforms)
    ds_valid = FakeSegmentationData(img_size, n_samples=20, transforms=transforms)

    return ds_train, ds_valid, (0.5, 0.5), None, None