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
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),    # in: 32x32x3 -> out: 32x32x16, RF: 3x3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.01)
        )   # params: Conv(3*16*3*3=432) + BN(16*2=32) = 464

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),    # in: 32x32x16 -> out: 32x32x32, RF: 5x5
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False),    # in: 32x32x32 -> out: 16x16x32, RF: 10x10 (5*2 + 0)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.01)
        )   # params: Conv1(16*32*3*3=4,608) + BN1(32*2=64) + Conv2(32*32*3*3=9,216) + BN2(32*2=64) = 13,952

        # Transition Block 1
        self.transition_block1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),    # in: 16x16x32 -> out: 16x16x16, RF: 10x10 (unchanged)
        )   # params: Conv(32*16*1*1=512)

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=2, dilation=2, bias=False),    # in: 16x16x16 -> out: 16x16x32, RF: 18x18 (10 + 2*(5-1))
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=2, bias=False),    # in: 16x16x32 -> out: 8x8x64, RF: 38x38 (18*2 + 2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.01)
        )   # params: Conv1(16*32*3*3=4,608) + BN1(32*2=64) + Conv2(32*64*3*3=18,432) + BN2(64*2=128) = 23,232

        # Transition Block 2
        self.transition_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),    # in: 8x8x64 -> out: 8x8x32, RF: 38x38 (unchanged)
        )   # params: Conv(64*32*1*1=2,048)

        # Block 3
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),    # in: 8x8x32 -> out: 8x8x64, RF: 40x40
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.01),
            DepthwiseSeparableConv(in_channels=64, out_channels=96, kernel_size=3, padding=1, stride=2, bias=False),    # in: 8x8x64 -> out: 4x4x96, RF: 82x82 (40*2 + 2)
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.01)
        )   # params: DSConv1((32*1*3*3=288)+(32*64*1*1=2,048)) + BN1(64*2=128) + DSConv2((64*1*3*3=576)+(64*96*1*1=6,144)) + BN2(96*2=192) = 9,376

        # Transition Block 3
        self.transition_block3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),    # in: 4x4x96 -> out: 4x4x32, RF: 82x82 (unchanged)
        )   # params: Conv(96*32*1*1=3,072)

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3, 3), padding=1, bias=False),    # in: 4x4x32 -> out: 4x4x96, RF: 84x84
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), padding=1, stride=2, bias=False),    # in: 4x4x96 -> out: 2x2x96, RF: 170x170 (84*2 + 2)
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.01)
        )   # params: Conv1(32*96*3*3=27,648) + BN1(96*2=192) + Conv2(96*96*3*3=82,944) + BN2(96*2=192) = 110,976

        # Global Average Pooling and FC layer
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),    # in: 2x2x96 -> out: 1x1x96, RF: 170x170 (unchanged)
        )   # params: 0 (no trainable parameters)
        
        self.fc = nn.Linear(96, 10)    # in: 96 -> out: 10
        # params: (96*10=960) + 10 = 970

    def forward(self, x):
        x = self.input_block(x)      # 32x32x16
        x = self.block1(x)           # 16x16x32
        x = self.transition_block1(x) # 16x16x16
        x = self.block2(x)           # 8x8x64
        x = self.transition_block2(x) # 8x8x32
        x = self.block3(x)           # 4x4x96
        x = self.transition_block3(x) # 4x4x32
        x = self.block4(x)           # 2x2x96
        x = self.gap(x)              # 1x1x96
        x = x.view(x.size(0), -1)    # 96
        x = self.fc(x)               # 10 (CIFAR10 classes)
        return x

"""
Receptive Field Calculation:
1. Input Block:  RF = 3x3
2. Block 1:      RF = 5x5 -> 10x10 (with stride 2)
3. Trans 1:      RF = 10x10 (unchanged)
4. Block 2:      RF = 18x18 (dilated) -> 38x38 (with stride 2)
5. Trans 2:      RF = 38x38 (unchanged)
6. Block 3:      RF = 40x40 -> 82x82 (with stride 2)
7. Trans 3:      RF = 82x82 (unchanged)
8. Block 4:      RF = 84x84 -> 170x170 (with stride 2)
Final RF: 170x170 (covers more than input 32x32)

Parameter Count Breakdown:
1. Input Block:       464
2. Block 1:           13,952
3. Transition 1:      512
4. Block 2:           23,232
5. Transition 2:      2,048
6. Block 3:           9,376
7. Transition 3:      3,072
8. Block 4:           110,976
9. FC Layer:          970
Total Parameters:     164,602
"""

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
    
