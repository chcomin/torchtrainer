import random
from skimage import draw
import torch
from torch.utils.data import Dataset

class FakeClassificationData(Dataset):
    """Dataset that returns a circle or disk image. Used for testing purposes."""

    def __init__(self, img_size=224, n_samples=100):
        
        half = img_size//2
        quarter = img_size//4
        
        img_square = torch.zeros((img_size, img_size), dtype=torch.uint8)
        inds = draw.rectangle((quarter, quarter), extent=(half, half))
        img_square[inds] = 1
        img_square = img_square.tile((3,1,1))

        img_disk = torch.zeros((img_size, img_size), dtype=torch.uint8)
        inds = draw.disk((half, half), quarter)
        img_disk[inds] = 1
        img_disk = img_disk.tile((3,1,1))

        self.imgs = [img_square, img_disk]
        self.n_samples = n_samples

    def __getitem__(self, idx):
        """Ignore index and return a random image."""
        idx = random.randint(0, 1)
        return self.imgs[idx], idx
    
    def __len__(self):
        return len(self.n_samples)

class FakeSegmentationData(FakeClassificationData):

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return img, img[0].to(torch.int64)
    
def get_dataset(img_size=224):

    ds_train = FakeSegmentationData(img_size, n_samples=80)
    ds_valid = FakeSegmentationData(img_size, n_samples=20)

    return ds_train, ds_valid, (0.5, 0.5), None, None