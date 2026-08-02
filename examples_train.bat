:: Example training scripts for different model architectures (Windows batch file)

:: Example 1: Train DRUNet on MNIST
:: Creates checkpoint_DRUnet.pth and config_DRUnet.json
echo Training DRUNet on MNIST...
python train_diffusion.py ^
    --dataset mnist ^
    --model_name DRUnet ^
    --base_channels 8 ^
    --img_size 32 ^
    --batch_size 512 ^
    --epochs 100 ^
    --lr 1e-4 ^
    --output_dir ./checkpoints_mnist

:: Example 2: Train JiT-B/16 on MNIST  
:: Creates checkpoint_JiT-B_16.pth and config_JiT-B_16.json
echo Training JiT-B/16 on MNIST...
python train_diffusion.py ^
    --dataset mnist ^
    --model_name JiT-B/16 ^
    --img_size 32 ^
    --batch_size 256 ^
    --epochs 100 ^
    --lr 1e-4 ^
    --attn_dropout 0.0 ^
    --proj_dropout 0.0 ^
    --output_dir ./checkpoints_mnist

:: Example 3: Train DRUNet on Flowers102
echo Training DRUNet on Flowers102...
python train_diffusion.py ^
    --dataset flowers ^
    --model_name DRUnet ^
    --base_channels 16 ^
    --img_size 128 ^
    --batch_size 64 ^
    --epochs 1000 ^
    --lr 1e-4 ^
    --output_dir ./checkpoints_flowers

:: Example 4: Train JiT-B/16 on Flowers102
echo Training JiT-B/16 on Flowers102...
python train_diffusion.py ^
    --dataset flowers ^
    --model_name JiT-B/16 ^
    --img_size 128 ^
    --batch_size 64 ^
    --epochs 1000 ^
    --lr 1e-4 ^
    --attn_dropout 0.0 ^
    --proj_dropout 0.0 ^
    --output_dir ./checkpoints_flowers

:: Test: Generate samples with DRUNet
echo Generating samples with DRUNet...
python test_diffusion.py ^
    --dataset mnist ^
    --model_name DRUnet ^
    --num_samples 10 ^
    --cfg_scale 2.0

:: Test: Generate samples with JiT-B/16
echo Generating samples with JiT-B/16...
python test_diffusion.py ^
    --dataset flowers ^
    --model_name JiT-B/16 ^
    --num_samples 10 ^
    --cfg_scale 2.0

echo Done!
pause
