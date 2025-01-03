# CIFAR-10 Image Classification

A custom CNN model implementation for CIFAR-10 image classification achieving 83%+ test accuracy.

## Architecture Features

- **Input Block**: Initial convolution with 16 filters
- **Block 1**: Standard convolutions with stride-2 reduction
- **Block 2**: Dilated convolution (dilation=2) for expanded receptive field
- **Block 3**: Depthwise Separable Convolutions for efficient computation
- **Block 4**: Standard convolutions with final feature extraction
- **Global Average Pooling**: Instead of flattening
- **Total Parameters**: ~164K
- **Final Receptive Field**: 170x170

## Training Features

- **Learning Rate Schedule**:
  - Dynamic warmup (5% of total epochs)
  - Cyclic learning rate with 40-epoch cycles
  - Three-phase learning within each cycle:
    - First 30%: Aggressive learning (0.15-1.0)
    - Middle 40%: Balanced learning (0.1-0.7)
    - Last 30%: Fine-tuning (0.075-0.4)
  - 10% LR reduction between cycles

- **Regularization**:
  - BatchNorm after each convolution
  - Lightweight Dropout (p=0.01)
  - Label smoothing (0.1)
  - Weight decay (6e-4)

## Data Augmentation

Using torchvision transforms:
- Random crop (32x32 with padding=4)
- Random horizontal flip
- Random rotation (±15 degrees)
- Random affine transforms (translation & scale)
- Color jitter (brightness & contrast)
- Random grayscale conversion
- Random erasing (cutout)
- Normalization with CIFAR-10 mean/std

## Requirements

```
torch
torchvision
numpy
prettytable
```

## Project Structure

```
.
├── model.py          # Model architecture definition
├── train.py          # Training script
├── dataset.py        # Dataset and transforms
├── utils.py          # Logging utilities
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start training:
```bash
python train.py
```

## Training Progress

The training script will show progress with:
- Epoch number
- Current step
- Training accuracy
- Test accuracy
- Loss value
- Learning rate
- Time per epoch

## Results

- Best Test Accuracy: 83.32%
- Training Time: ~50s per epoch on CPU
- Total Epochs: 120
- Batch Size: 128 