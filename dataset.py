import torch
from torchvision import datasets, transforms
import numpy as np

class CIFAR10Transform:
    def __init__(self, mean, std, train=True):
        transforms_list = []
        
        if train:
            # Spatial transforms (before tensor conversion)
            transforms_list.extend([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=15,
                    fill=(int(mean[0]*255), int(mean[1]*255), int(mean[2]*255))
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # Increased translation
                    scale=(0.9, 1.1),      # Increased scale range
                    fill=(int(mean[0]*255), int(mean[1]*255), int(mean[2]*255))
                ),
                transforms.ColorJitter(
                    brightness=0.2,    # Increased color augmentation
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),  # Added grayscale
            ])
        
        # Convert to tensor and normalize
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        if train:
            # Cutout (after tensor conversion)
            transforms_list.extend([
                transforms.RandomErasing(
                    p=0.3,             # Reduced probability
                    scale=(0.02, 0.2),  # Increased max size
                    ratio=(0.3, 3.3),
                    value='random'      # Use random values instead of 0
                ),
                transforms.RandomErasing(  # Second erasing for more robustness
                    p=0.2,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value='random'
                )
            ])
        
        self.transform = transforms.Compose(transforms_list)
    
    def __call__(self, img):
        return self.transform(img)

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