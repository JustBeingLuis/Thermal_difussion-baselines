# Diffusion Quickstart

This project is configured for unconditional diffusion training and testing.
Class labels are ignored in both training and evaluation.

## 1) Create environment

Use the provided Conda environment file:

```powershell
conda env create -f env.yml
```

## 2) Activate environment

```powershell
conda activate diffusion
```

## 3) Run training

Single GPU / default run:

```powershell
python train_diffusion.py --dataset flowers --model_name JiT-Flowers
```

Multi-GPU (Windows PowerShell):

```powershell
$env:USE_LIBUV="0"; torchrun --nproc_per_node=2 train_diffusion.py --dataset flowers --model_name JiT-Flowers
```

Checkpoints are saved in checkpoints_flowers by default.

## 4) Run test

Run generation + FID (uses train split for supported datasets):

```powershell
python test_diffusion.py --dataset flowers --model_name JiT-Flowers --device cuda
```

Generated grid images are saved in generated_flowers.

## Notes

- If you do not want FID, add --no-compute_metrics.
