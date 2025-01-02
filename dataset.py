import torch
from torchvision import datasets
import numpy as np
from albumentations import (
    Compose, CoarseDropout, Normalize, HorizontalFlip, 
    RandomBrightnessContrast, ShiftScaleRotate, GaussNoise,
    PadIfNeeded, RandomCrop, Blur
)

class AlbumentationsTransform:
    def __init__(self, mean, std, train=True):
        if train:
            self.transform = Compose([
                # Spatial transforms
                PadIfNeeded(min_height=36, min_width=36, border_mode=4, p=1.0),
                RandomCrop(height=32, width=32, p=1.0),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    border_mode=4,  # BORDER_REFLECT_101
                    p=0.7
                ),
                
                # Color transforms
                RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5
                ),
                
                # Noise and dropout
                GaussNoise(
                    var_limit=(5.0, 20.0),
                    mean=0,
                    per_channel=True,
                    p=0.3
                ),
                Blur(
                    blur_limit=3,
                    p=0.2
                ),
                CoarseDropout(
                    max_holes=2,
                    max_height=6,
                    max_width=6,
                    min_holes=1,
                    min_height=2,
                    min_width=2,
                    fill_value=None,  # Using None will use random values
                    p=0.5
                ),
                
                # Normalization
                Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                )
            ])
        else:
            # Test transform - only normalization
            self.transform = Compose([
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