# Model Architecture Guide

This guide explains how to train and test diffusion models with different architectures.

## Available Model Architectures

### DRUNet (Default)
- **Model Name**: `DRUnet`
- **Type**: U-Net based architecture with residual blocks
- **Parameters**: 
  - `--base_channels`: Base number of channels (default: 8)
- **Best for**: General purpose, good for MNIST and smaller images
- **Example**:
  ```bash
  python train_diffusion.py --dataset mnist --model_name DRUnet --base_channels 8
  ```

### JiT Models (Joint-Embedding Transformer)
Vision Transformer-based architectures with different sizes:

#### JiT-B/16 (Base, patch size 16)
- **Model Name**: `JiT-B/16`
- **Parameters**:
  - `--attn_dropout`: Attention dropout rate (default: 0.0)
  - `--proj_dropout`: Projection dropout rate (default: 0.0)
- **Best for**: Medium-sized images (64x64 to 128x128)
- **Example**:
  ```bash
  python train_diffusion.py --dataset flowers --model_name JiT-B/16 --img_size 128
  ```

#### JiT-B/32 (Base, patch size 32)
- **Model Name**: `JiT-B/32`
- **Best for**: Larger images with lower computational cost

#### JiT-L/16 (Large, patch size 16)
- **Model Name**: `JiT-L/16`
- **Best for**: High-quality generation, requires more memory

#### JiT-L/32 (Large, patch size 32)
- **Model Name**: `JiT-L/32`

#### JiT-H/16 (Huge, patch size 16)
- **Model Name**: `JiT-H/16`
- **Best for**: State-of-the-art quality, requires significant GPU memory

#### JiT-H/32 (Huge, patch size 32)
- **Model Name**: `JiT-H/32`

## Training Examples

### Training DRUNet on MNIST
```bash
python train_diffusion.py \
    --dataset mnist \
    --model_name DRUnet \
    --base_channels 8 \
    --img_size 32 \
    --batch_size 512 \
    --epochs 100 \
    --lr 1e-4
```

### Training JiT-B/16 on Flowers102
```bash
python train_diffusion.py \
    --dataset flowers \
    --model_name JiT-B/16 \
    --img_size 128 \
    --batch_size 64 \
    --epochs 1000 \
    --lr 1e-4 \
    --attn_dropout 0.0 \
    --proj_dropout 0.0
```

### Training JiT-L/16 on Flowers102 (requires more memory)
```bash
python train_diffusion.py \
    --dataset flowers \
    --model_name JiT-L/16 \
    --img_size 128 \
    --batch_size 32 \
    --epochs 1000 \
    --lr 1e-4
```

## Testing / Generation

### Generate with automatic model detection
The model architecture will be loaded from the checkpoint's config:
```bash
python test_diffusion.py \
    --dataset mnist \
    --num_samples 10 \
    --cfg_scale 2.0
```

### Override model architecture (if config is missing)
```bash
python test_diffusion.py \
    --dataset flowers \
    --model_name JiT-B/16 \
    --num_samples 10 \
    --cfg_scale 2.0
```

## Model Selection Guidelines

### For MNIST (32x32, grayscale):
- **Recommended**: DRUnet (fast, efficient)
- **Alternative**: JiT-B/16 or JiT-B/32

### For Flowers102 (128x128, RGB):
- **Fast Training**: DRUnet with base_channels=16
- **Good Quality**: JiT-B/16
- **Best Quality**: JiT-L/16 or JiT-H/16 (if you have enough GPU memory)

## Configuration Files

All training configurations are automatically saved to `{output_dir}/config.json`, including:
- `model_name`: Which architecture was used
- `base_channels`: For DRUNet models
- `attn_dropout`, `proj_dropout`: For JiT models
- All other hyperparameters

This allows the test script to automatically load the correct architecture.

## Memory Requirements

Approximate VRAM requirements (batch_size=1):
- **DRUnet**: ~500MB - 2GB (depending on base_channels)
- **JiT-B/16**: ~2GB - 4GB
- **JiT-L/16**: ~4GB - 8GB
- **JiT-H/16**: ~8GB - 16GB

Scale linearly with batch size.
