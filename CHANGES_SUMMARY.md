# Diffusion Model Architecture Support - Summary of Changes

## Overview
Modified the training and testing scripts to support multiple model architectures, specifically **DRUNet** and **JiT (Joint-Embedding Transformer)** variants.

## Files Modified

### 1. train_diffusion.py
**Changes:**
- Added `--model_name` argument to select model architecture
  - Choices: `DRUnet`, `JiT-B/16`, `JiT-B/32`, `JiT-L/16`, `JiT-L/32`, `JiT-H/16`, `JiT-H/32`
- Reorganized architecture-specific parameters:
  - `--base_channels`: Only for DRUNet models
  - `--attn_dropout`, `--proj_dropout`: Only for JiT models
- Updated model initialization to pass `model_name` to Denoiser
- Updated training output to display the selected model architecture
- Config file now saves `model_name` for automatic loading during testing

### 2. test_diffusion.py
**Changes:**
- Added `--model_name` optional argument to override model architecture
- Modified `load_model()` function to:
  - Accept `model_name_override` parameter
  - Load model architecture from checkpoint config automatically
  - Allow manual override via command-line if needed
- Added default values for JiT-specific parameters in fallback config
- Improved error handling when config is missing

## New Features

### Automatic Model Detection
The test script now automatically detects which model architecture was used during training by reading the saved config.json file.

### Manual Override
You can override the model architecture when testing if the config is missing or you want to experiment:
```bash
python test_diffusion.py --dataset flowers --model_name JiT-B/16
```

### Config Persistence
All model-specific parameters are saved in the checkpoint config, including:
- `model_name`
- `base_channels` (DRUNet)
- `attn_dropout`, `proj_dropout` (JiT)

## Usage Examples

### Training

**DRUNet on MNIST:**
```bash
python train_diffusion.py --dataset mnist --model_name DRUnet --base_channels 8
```

**JiT-B/16 on Flowers:**
```bash
python train_diffusion.py --dataset flowers --model_name JiT-B/16 --img_size 128
```

**JiT-L/16 on Flowers (high quality):**
```bash
python train_diffusion.py --dataset flowers --model_name JiT-L/16 --img_size 128 --batch_size 32
```

### Testing

**Automatic (recommended):**
```bash
python test_diffusion.py --dataset mnist --cfg_scale 2.0
```

**With manual override:**
```bash
python test_diffusion.py --dataset flowers --model_name JiT-B/16 --cfg_scale 2.0
```

## Additional Files Created

1. **README_MODELS.md** - Comprehensive guide on model architectures
2. **examples_train.sh** - Bash script with training examples
3. **examples_train.bat** - Windows batch file with training examples

## Backward Compatibility

- Existing checkpoints without `model_name` in config will default to `DRUnet`
- All previous command-line arguments remain functional
- Default behavior unchanged when `--model_name` is not specified (defaults to `DRUnet`)

## Model Architecture Summary

| Model | Type | Patch Size | Best For | Memory |
|-------|------|------------|----------|---------|
| DRUnet | U-Net | N/A | General purpose, fast | Low |
| JiT-B/16 | ViT | 16 | Medium images | Medium |
| JiT-B/32 | ViT | 32 | Large images, efficient | Medium |
| JiT-L/16 | ViT | 16 | High quality | High |
| JiT-L/32 | ViT | 32 | High quality, efficient | High |
| JiT-H/16 | ViT | 16 | SOTA quality | Very High |
| JiT-H/32 | ViT | 32 | SOTA quality, efficient | Very High |

## Testing Recommendations

1. For MNIST (32x32): Use DRUnet or JiT-B/16
2. For Flowers102 (128x128): Use JiT-B/16 for good quality, JiT-L/16 for best quality
3. Adjust batch size based on GPU memory (JiT models require more memory)

## Next Steps

You can now:
1. Train models with different architectures
2. Compare performance between DRUNet and JiT variants
3. Experiment with different configurations for optimal results
