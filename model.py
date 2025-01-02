import torch.nn as nn
import torch

class CIFAR10Model(nn.Module):
    """
    CNN model for CIFAR10 image classification
    Target Classes:
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
    def __init__(self):
        super().__init__()
        
        # Input Block
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),    # in: 32x32x3 -> out: 32x32x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05)
        )   # params: (3*16*3*3) + (16*2) = 464

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),    # in: 32x32x16 -> out: 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False),    # in: 32x32x32 -> out: 16x16x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )   # params: (16*32*3*3) + (32*2) + (32*32*3*3) + (32*2) = 18,624

        # Transition Block 1 (channel reduction)
        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=(1, 1), bias=False),    # in: 16x16x32 -> out: 16x16x24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.05)
        )   # params: (32*24) + (24*2) = 816

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=24,
                out_channels=48,
                kernel_size=(3, 3),
                padding=2,  # Padding increased to maintain spatial dimensions with dilation
                dilation=2,  # Add dilation to capture larger context
                bias=False
            ),    # in: 16x16x24 -> out: 16x16x48 (with larger receptive field)
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, stride=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.05)
        )   # params: (24*48*3*3) + (48*2) + (48*48*3*3) + (48*2) = 42,624

        # Transition Block 2 (channel reduction)
        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(1, 1), bias=False),    # in: 8x8x48 -> out: 8x8x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )   # params: (48*32) + (32*2) = 1,600

        # Block 3
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),    # in: 8x8x32 -> out: 8x8x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            DepthwiseSeparableConv(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False
            ),    # in: 8x8x64 -> out: 4x4x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )   # params: (32*1*3*3 + 32*64*1*1) + (64*2) + (64*1*3*3 + 64*64*1*1) + (64*2) = 4,864

        # Transition Block 3 (channel reduction)
        self.transition3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=(1, 1), bias=False),    # in: 4x4x64 -> out: 4x4x48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.05)
        )   # params: (64*48) + (48*2) = 3,168

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=112, kernel_size=(3, 3), padding=1, bias=False),    # in: 4x4x48 -> out: 4x4x112
            nn.BatchNorm2d(112),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(in_channels=112, out_channels=112, kernel_size=(3, 3), padding=1, stride=2, bias=False),    # in: 4x4x112 -> out: 2x2x112
            nn.BatchNorm2d(112),
            nn.ReLU(),
            nn.Dropout(0.05)
        )   # params: (48*112*3*3) + (112*2) + (112*112*3*3) + (112*2) = 120,736

        # Global Average Pooling and FC layer
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.05)
        )    # in: 2x2x112 -> out: 1x1x112
        self.fc = nn.Linear(112, 10)                # in: 112 -> out: 10 (10 CIFAR10 classes)
        # params: (112*10) + 10 = 1,130

    def forward(self, x):
        x = self.input_block(x)      # 32x32x16
        identity = x
        x = self.block1(x)           # 16x16x32
        x = self.transition1(x)      # 16x16x24
        x = self.block2(x)           # 8x8x48
        x = self.transition2(x)      # 8x8x32
        x = self.block3(x)           # 4x4x64
        x = self.transition3(x)      # 4x4x48
        x = self.block4(x)           # 2x2x112
        x = self.gap(x)              # 1x1x112
        x = x.view(x.size(0), -1)    # 112
        x = self.fc(x)               # 10 (CIFAR10 classes)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Each input channel is convolved separately
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Parameter count breakdown:
# Input Block:
# - Conv2d(3, 16): (3*16*3*3) + (16*2) = 464

# Block 1:
# - Conv2d(16, 32): (16*32*3*3) + (32*2)
# - Conv2d(32, 32): (32*32*3*3) + (32*2)
# Total: 18,624

# Transition 1:
# - Conv2d(32, 24): (32*24) + (24*2) = 816

# Block 2:
# - Conv2d(24, 48): (24*48*3*3) + (48*2)
# - Conv2d(48, 48): (48*48*3*3) + (48*2)
# Total: 42,624

# Transition 2:
# - Conv2d(48, 32): (48*32) + (32*2) = 1,600

# Block 3 (Depthwise Separable):
# - DepthwiseConv(32): (32*1*3*3) + (32*64*1*1)
# - DepthwiseConv(64): (64*1*3*3) + (64*64*1*1)
# - BatchNorm: (64*2) * 2
# Total: 4,864

# Transition 3:
# - Conv2d(64, 48): (64*48) + (48*2) = 3,168

# Block 4:
# - Conv2d(48, 112): (48*112*3*3) + (112*2)
# - Conv2d(112, 112): (112*112*3*3) + (112*2)
# Total: 120,736

# FC Layer:
# - Linear(112, 10): (112*10) + 10 = 1,130

# Total Parameters: 194,026
    
