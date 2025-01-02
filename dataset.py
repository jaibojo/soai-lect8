import torch
from torchvision import datasets
import numpy as np
from albumentations import (
    Compose, CoarseDropout, Normalize, HorizontalFlip, 
    RandomBrightnessContrast, ShiftScaleRotate, GaussNoise
)

class AlbumentationsTransform:
    def __init__(self, mean, std):
        # CIFAR-10 mean values for each channel
        self.transform = Compose([
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            GaussNoise(
                var_limit=(5.0, 30.0),
                mean=0,
                p=0.3
            ),
            CoarseDropout(
                max_holes=2,
                max_height=8,
                max_width=8,
                min_holes=1,
                min_height=2,
                min_width=2,
                fill_value=(int(mean[0] * 255), int(mean[1] * 255), int(mean[2] * 255)),
                p=0.5
            ),
            Normalize(
                mean=mean,
                std=std,
                max_pixel_value=255.0,
            )
        ])

    def __call__(self, img):
        img = np.array(img)
        transformed = self.transform(image=img)
        img_transformed = transformed['image']
        tensor = torch.from_numpy(img_transformed)
        tensor = tensor.float()
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        return tensor

class CIFAR10Dataset(datasets.CIFAR10):
    """
    Custom CIFAR10 Dataset class
    Classes:
    0: Airplane
    1: Automobile
    2: Bird
    3: Cat
    4: Deer
    5: Dog
    6: Frog
    7: Horse
    8: Ship
    9: Truck
    """
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, download=download)
        
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.transform is not None:
            img = self.transform(img)
        return img, target 