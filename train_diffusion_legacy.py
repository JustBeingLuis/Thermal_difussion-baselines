"""
Diffusion Model Training with Multiple Architectures
Supports DRUNet and JiT (Joint-Embedding Transformer) models.
Supports MNIST, Flowers102, and CelebA datasets.
"""

import argparse
import torch
torch.set_float32_matmul_precision('high')

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pathlib import Path
import copy
import json
from tqdm import tqdm
from denoiser import Denoiser
from PIL import Image


@torch.no_grad()
def generate_samples(model, device, output_path, num_classes=10, num_samples=None, class_label=0):
    """Generate samples and save as grid."""
    model.eval()
    
    if num_samples is None:
        num_samples = num_classes

    # Generate one sample per class or multiple samples of a single class
    if num_classes <= 1:
        labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    else:
        labels = torch.arange(num_classes, device=device)
    samples = model.generate(labels)
    
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    # Save as horizontal grid
    nrow = num_samples if num_classes <= 1 else num_classes
    save_image(samples, output_path, nrow=nrow, padding=2)
    model.train()


class HuggingFaceDatasetWrapper(Dataset):
    """Wrapper for HuggingFace datasets to work with PyTorch DataLoader."""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class UnconditionalDatasetWrapper(Dataset):
    """Dataset wrapper that ignores targets and returns a dummy label."""

    def __init__(self, dataset, label=0):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.label


def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for x, labels in pbar:
        x = x.to(device)
        labels = labels.to(device)
        
        # Normalize to [-1, 1]
        x = x * 2 - 1
        
        # Forward pass (loss computation is inside the model)
        loss = model(x, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA
        model.update_ema()
        
        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix({'loss': f'{loss_val:.4f}'})
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, path, config=None):
    """Save model checkpoint."""
    ema_state_dict1 = {name: model.ema_params1[i] for i, (name, _) in enumerate(model.named_parameters())}
    ema_state_dict2 = {name: model.ema_params2[i] for i, (name, _) in enumerate(model.named_parameters())}
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'model_ema1': ema_state_dict1,
        'model_ema2': ema_state_dict2,
        'optimizer': optimizer.state_dict(),
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # Load EMA parameters
    ema_state_dict1 = checkpoint['model_ema1']
    ema_state_dict2 = checkpoint['model_ema2']
    model.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model.named_parameters()]
    model.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model.named_parameters()]
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser(description='Simple MNIST Diffusion Training')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='celeba', choices=['mnist', 'flowers', 'celeba'],
                        help='Dataset to train on (mnist, flowers, or celeba)')
    
    # Model architecture
    parser.add_argument('--model_name', type=str, default='JiT-Celeba',
                        choices=['JiT-B/16', 'JiT-Flowers', 'JiT-Celeba'],
                        help='Model architecture to use')
    parser.add_argument('--img_size', type=int, default=None, help='Image size (auto-set if None)')
    
    # DRUNet-specific parameters
    parser.add_argument('--base_channels', type=int, default=8, help='Base channels in DRUNet (only for DRUnet)')
    
    # JiT-specific parameters
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate (only for JiT models)')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate (only for JiT models)')
    
    
    # Training
    parser.add_argument('--epochs', type=int, default=60000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--ema_decay1', type=float, default=0.9999, help='EMA decay rate 1')
    parser.add_argument('--ema_decay2', type=float, default=0.9996, help='EMA decay rate 2')
    
    # Diffusion parameters
    parser.add_argument('--P_mean', type=float, default=-0.8, help='Time sampling mean (log-space)')
    parser.add_argument('--P_std', type=float, default=0.8, help='Time sampling std (log-space)')
    parser.add_argument('--t_eps', type=float, default=1e-3, help='Minimum time to avoid division by zero')
    parser.add_argument('--noise_scale', type=float, default=1.0, help='Noise scale')
    parser.add_argument('--label_drop_prob', type=float, default=0.1, help='Label dropout for CFG')
    
    # Sampling (for generation)
    parser.add_argument('--sampling_method', type=str, default='heun', help='Sampling method')
    parser.add_argument('--num_sampling_steps', type=int, default=40, help='Number of sampling steps')
    parser.add_argument('--cfg', type=float, default=1.0, help='Classifier-free guidance scale')
    parser.add_argument('--interval_min', type=float, default=0.0, help='CFG interval min')
    parser.add_argument('--interval_max', type=float, default=1.0, help='CFG interval max')
    
    # Dataclear
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--class_num', type=int, default=None, help='Number of classes (auto-set if None)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (auto-set if None)')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=1, help='Save frequency (epochs)')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Configure dataset-specific parameters
    if args.dataset == 'mnist':
        if args.img_size is None:
            args.img_size = 32
        if args.class_num is None:
            args.class_num = 10
        if args.output_dir is None:
            args.output_dir = './checkpoints_mnist'
        in_channels = 1
    elif args.dataset == 'flowers':
        if args.img_size is None:
            args.img_size = 128
        if args.class_num is None:
            args.class_num = 102
        if args.output_dir is None:
            args.output_dir = './checkpoints_flowers'
        in_channels = 3
    elif args.dataset == 'celeba':
        if args.img_size is None:
            args.img_size = 64
        if args.class_num is None:
            args.class_num = 1
        if args.output_dir is None:
            args.output_dir = './checkpoints_celeba'
        in_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Set resume path default if empty
    if not args.resume:
        resume_path = Path(args.output_dir) / f'checkpoint_{args.model_name.replace("/", "_")}.pth'
        if resume_path.exists():
            args.resume = str(resume_path)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration as JSON
    config = vars(args)
    model_name_safe = args.model_name.replace("/", "_")
    config_path = Path(args.output_dir) / f'config_{model_name_safe}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")
    
    print("=" * 60)
    print(f"Diffusion Training - {args.dataset.upper()}")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Channels: {in_channels}")
    print(f"Classes: {args.class_num}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    # Load dataset
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform
        )
    elif args.dataset == 'flowers':
        # For flowers, resize and center crop to ensure consistent dimensions
        transform = transforms.Compose([
            transforms.Resize( (150, 150) ),
            transforms.RandomCrop( args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        # Load from HuggingFace
        from datasets import load_dataset
        hf_dataset = load_dataset('nelorth/oxford-flowers', split='train', trust_remote_code=True)
        dataset = HuggingFaceDatasetWrapper(hf_dataset, transform=transform)
    elif args.dataset == 'celeba':
        transform = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = datasets.CelebA(
            root=args.data_path,
            split='train',
            download=True,
            transform=transform,
        )
        dataset = UnconditionalDatasetWrapper(dataset, label=0)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create model using Denoiser from denoiser.py
    device = torch.device(args.device)
    # Add in_channels to args for model initialization
    args.in_channels = in_channels
    model = Denoiser(args, model_name=args.model_name).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Initialize EMA parameters
    model.ema_params1 = copy.deepcopy(list(model.parameters()))
    model.ema_params2 = copy.deepcopy(list(model.parameters()))
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device) + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    model_name_safe = args.model_name.replace("/", "_")
    checkpoint_path = Path(args.output_dir) / f'checkpoint_{model_name_safe}.pth'
    
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch, args)
        print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')
        
        # Generate conditional samples (1 per class)
        output_image_path = Path(args.output_dir) / f'out_{model_name_safe}.png'
        if args.dataset == 'celeba':
            generate_samples(
                model,
                device,
                output_image_path,
                num_classes=args.class_num,
                num_samples=10,
                class_label=0
            )
        else:
            generate_samples(model, device, output_image_path, num_classes=args.class_num)
        print(f'Generated samples saved to {output_image_path}')
        
        # Always save last checkpoint (overwrite)
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, epoch, checkpoint_path, config=config)
    
    print("\nTraining completed!")
    print(f"Final checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
