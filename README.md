# CIFAR-10 Image Classification

A PyTorch implementation of a CNN model for CIFAR-10 image classification with modern architecture features and training techniques.

## Architecture Features

- **Input Block**: Initial convolution with 16 filters
- **Multiple Blocks with Progressive Feature Extraction**:
  - Block 1: Standard convolutions (32 channels)
  - Block 2: Dilated convolutions for larger receptive field (48 channels)
  - Block 3: Depthwise separable convolutions (64 channels)
  - Block 4: Standard convolutions (112 channels)
- **Transition Blocks**: 1x1 convolutions for channel reduction
- **Modern Techniques**:
  - BatchNorm after every convolution
  - Dropout (p=0.05) for regularization
  - Global Average Pooling
  - Dilated Convolutions
  - Depthwise Separable Convolutions

Total Parameters: ~194k

## Training Features

- **Optimizer**: AdamW with weight decay (1e-3)
- **Learning Rate**: Cosine annealing with warmup
- **Regularization**:
  - Label smoothing (0.3)
  - Weight decay
  - Gradient clipping
  - Dropout

## Data Augmentation

Strong augmentation pipeline using albumentations:
- Horizontal Flip
- ShiftScaleRotate
- RandomBrightnessContrast
- GaussNoise
- CoarseDropout with mean-value filling

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
numpy<2.0
```

## Project Structure

- `model.py`: CNN architecture implementation
- `dataset.py`: CIFAR-10 dataset and augmentation setup
- `train.py`: Training loop and configuration
- `utils.py`: Training logger and utilities

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python train.py
```

The training progress will be displayed in a table format showing:
- Current epoch
- Training accuracy
- Test accuracy
- Loss
- Learning rate

## Model Architecture Details

1. **Input Block** (32x32x3 → 32x32x16)
   - Conv2d(3→16, 3x3)
   - BatchNorm
   - ReLU
   - Dropout(0.05)

2. **Block 1** (32x32x16 → 16x16x32)
   - Two Conv2d layers
   - Stride 2 in second conv
   - BatchNorm, ReLU, Dropout after each

3. **Block 2** (16x16x24 → 8x8x48)
   - Dilated Conv2d (dilation=2)
   - Regular Conv2d with stride 2
   - BatchNorm, ReLU, Dropout

4. **Block 3** (8x8x32 → 4x4x64)
   - Two Depthwise Separable Convs
   - Stride 2 in second conv
   - BatchNorm, ReLU, Dropout

5. **Block 4** (4x4x48 → 2x2x112)
   - Two Conv2d layers
   - Stride 2 in second conv
   - BatchNorm, ReLU, Dropout

6. **Output** (2x2x112 → 10)
   - Global Average Pooling
   - Dropout
   - Linear(112→10)

## Training Configuration

- Batch size: 128
- Initial learning rate: 0.0005
- Weight decay: 1e-3
- Label smoothing: 0.3
- Warmup epochs: 5
- Total epochs: 50
- Gradient clipping: 1.0 