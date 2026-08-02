"""
Diffusion Model Training with Multiple Architectures
Supports DRUNet and JiT (Joint-Embedding Transformer) models.
Supports MNIST, Flowers102, and CelebA datasets.
"""

# default command
# $env:USE_LIBUV="0"; torchrun --nproc_per_node=8 train_diffusion.py --epochs 1000 --batch_size 64

import argparse
import csv
import subprocess
import sys
import torch
torch.set_float32_matmul_precision('medium')
import numpy as np
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pathlib import Path
import copy
import json
from tqdm import tqdm
from denoiser import Denoiser
from PIL import Image
from util import misc

# Import the metrics computation function and CSV helper
try:
    from test_diffusion import compute_fid_metrics, _append_results_csv
except ImportError:
    compute_fid_metrics = None
    _append_results_csv = None


def _resolve_amp_dtype(amp_dtype, device):
    """Resolve AMP dtype from user config with CUDA capability fallback."""
    if device.type != 'cuda':
        return torch.float32, 'fp32'

    if amp_dtype == 'auto':
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, 'bf16'
        return torch.float16, 'fp16'

    if amp_dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, 'bf16'
        print('bf16 requested but not supported on this GPU; falling back to fp16')
        return torch.float16, 'fp16'

    return torch.float16, 'fp16'


def _build_lr_scheduler(optimizer, start_lr, end_lr, total_epochs, start_epoch):
    """Create a cosine annealing scheduler from start_lr to end_lr."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - 1, 1),
        eta_min=end_lr,
        last_epoch=start_epoch - 1,
    )


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


def train_epoch(model, model_without_ddp, dataloader, optimizer, device, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not misc.is_main_process())
    
    for x, labels in pbar:
        x = x.to(device)
        labels = labels.to(device)
        
        # Normalize to [-1, 1]
        x = x * 2 - 1
        
        # Forward pass (loss computation is inside the model)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA
        model_without_ddp.update_ema()
        
        loss_val = loss.item()
        total_loss += loss_val
        if misc.is_main_process():
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})

    avg_loss = total_loss / len(dataloader)
    if misc.is_dist_avail_and_initialized():
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (avg_loss_tensor / misc.get_world_size()).item()

    return avg_loss


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
    
    misc.save_on_master(checkpoint, path)
    print(f'Checkpoint saved to {path}')


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model_state = checkpoint['model']

    # Support checkpoints saved from both DDP and non-DDP models.
    if any(name.startswith('module.') for name in model_state.keys()):
        model_state = {name.replace('module.', '', 1): value for name, value in model_state.items()}

    model.load_state_dict(model_state)
    
    # Load EMA parameters
    ema_state_dict1 = checkpoint['model_ema1']
    ema_state_dict2 = checkpoint['model_ema2']
    model.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model.named_parameters()]
    model.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model.named_parameters()]
    
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0)


def run_periodic_eval(args, device, epoch):
    """Run test_diffusion.py as a subprocess for periodic progress checks."""
    repo_root = Path(__file__).resolve().parent
    test_script = repo_root / 'test_diffusion.py'

    cmd = [
        sys.executable,
        str(test_script),
        '--dataset', args.dataset,
        '--model_name', args.model_name,
        '--device', str(device),
        '--num_sampling_steps', str(args.num_sampling_steps),
    ]

    if args.eval_skip_grid:
        cmd.append('--skip_grid')

    print(f"\n[Eval] Epoch {epoch + 1}: running {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(repo_root), check=False)
    if completed.returncode != 0:
        print(f"[Eval] test_diffusion.py failed at epoch {epoch + 1} with return code {completed.returncode}")


@torch.no_grad()
def run_distributed_eval(model, args, device, epoch):
    """Run evaluation directly in distributed mode.
    
    Computes FID metrics with distributed image generation (images generated 
    across all ranks, metrics computed on rank 0). Results are saved to CSV.
    """
    is_main = misc.is_main_process()
    
    if is_main:
        print(f"\n[Eval] Epoch {epoch + 1}: Computing FID metrics...")
    
    model.eval()
    
    # Set up AMP
    use_amp = device.type == 'cuda'
    amp_dtype, _ = _resolve_amp_dtype('bf16', device)
    
    # Compute FID metrics in distributed mode
    if compute_fid_metrics is not None and hasattr(args, 'compute_metrics') and args.compute_metrics:
        try:
            metrics_dict = compute_fid_metrics(
                model=model,
                args=args,
                device=device,
                cfg_scale=1.0,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            
            if is_main and metrics_dict is not None:
                fid = metrics_dict.get('frechet_inception_distance')
                if fid is not None:
                    print(f"[Eval] Epoch {epoch + 1}: FID = {fid:.4f}")
                    
                    # Save to CSV if available
                    if _append_results_csv is not None:
                        results_row = {
                            'dataset': args.dataset,
                            'model': args.model_name,
                            'epoch': epoch,
                            'pred': getattr(args, 'pred', 'x'),
                            'w': getattr(args, 'w', 'v'),
                            'learning rate': getattr(args, 'lr', ''),
                            'batch size': getattr(args, 'batch_size', ''),
                            'fid': fid,
                        }
                        results_path = Path(__file__).resolve().parent / 'results.csv'
                        _append_results_csv(results_path, results_row)
                        print(f"Results appended to: {results_path}")
        
        except Exception as exc:
            if is_main:
                print(f"[Eval] FID computation failed ({exc}); continuing training")
        
        # Sync across ranks after metrics
        if misc.is_dist_avail_and_initialized():
            dist.barrier()
    
    model.train()


def main():
    parser = argparse.ArgumentParser(description='Simple MNIST Diffusion Training')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='celeba', choices=['mnist', 'flowers', 'celeba'],
                        help='Dataset to train on (mnist, flowers, or celeba)')
    
    # Model architecture
    parser.add_argument('--model_name', type=str, default='JiT-CelebaB',
                        choices=['JiT-B/4', 'JiT-Flowers', 'JiT-Celeba', 'JiT-CelebaB', 'DRUnet', 'Unet'],
                        help='Model architecture to use')
    parser.add_argument('--img_size', type=int, default=None, help='Image size (auto-set if None)')
    parser.add_argument('--loss', type=str, default='sup', choices=['sup', 'noisy', 'noise2noise', 'gr2r'], help='Loss type')
    
    # DRUNet-specific parameters
    parser.add_argument('--base_channels', type=int, default=64, help='Base channels in DRUNet (only for DRUnet)')
    
    # JiT-specific parameters
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate (only for JiT models)')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate (only for JiT models)')
    
    
    # Training
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--end_lr', type=float, default=1e-5,
                        help='Final learning rate for linear scheduling')
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
    parser.add_argument('--sampling_method', type=str, default='euler', help='Sampling method')
    parser.add_argument('--num_sampling_steps', type=int, default=50, help='Number of sampling steps')
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
    parser.add_argument('--eval_freq', type=int, default=200,
                        help='Run test_diffusion.py every N epochs (0 disables periodic eval)')
    parser.add_argument('--eval_skip_grid', dest='eval_skip_grid', action='store_true',
                        help='Pass --skip_grid to test_diffusion.py during periodic eval (default: enabled)')
    parser.add_argument('--no_eval_skip_grid', dest='eval_skip_grid', action='store_false',
                        help='Do not pass --skip_grid to test_diffusion.py during periodic eval')
    parser.set_defaults(eval_skip_grid=True)
    
    # Metrics (FID computation during distributed training eval)
    parser.add_argument('--compute_metrics', action='store_true', default=True,
                        help='Compute FID metrics during distributed eval (default: enabled)')
    parser.add_argument('--num_gen_images', type=int, default=52000,
                        help='Number of generated images for FID metrics')
    parser.add_argument('--num_real_images', type=int, default=-1,
                        help='Number of real images for FID metrics (-1 for full dataset)')
    parser.add_argument('--metrics_batch_size', type=int, default=500,
                        help='Batch size per rank for FID generation')
    parser.add_argument('--metrics_seed', type=int, default=42,
                        help='Seed for reproducible real image subset selection')

    parser.set_defaults(compute_metrics=True)  # Enable metrics by default for distributed eval

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility / distributed training
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank for torchrun')
    parser.add_argument('--dist_on_itp', action='store_true', help='Distributed launch on ITP')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')

    # pred mode args
    parser.add_argument('--pred', type=str, default='x', choices=['x', 'v'], help='parameterization: x or v')
    parser.add_argument('--w', type=str, default='v', choices=['x', 'v'], help='Loss weight: x or v')
    
    args = parser.parse_args()

    if args.eval_freq < 0:
        raise ValueError('eval_freq must be >= 0')

    misc.init_distributed_mode(args)

    if not getattr(args, 'distributed', False):
        args.gpu = 0

    if str(args.device).startswith('cuda') and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    is_main_rank = misc.is_main_process()
    
    # Configure dataset-specific parameters
    if args.dataset == 'mnist':
        if args.img_size is None:
            args.img_size = 32
        if args.output_dir is None:
            args.output_dir = './checkpoints_mnist'
        in_channels = 1
    elif args.dataset == 'flowers':
        if args.img_size is None:
            args.img_size = 128
        if args.output_dir is None:
            args.output_dir = './checkpoints_flowers'
        in_channels = 3
    elif args.dataset == 'celeba':
        if args.img_size is None:
            args.img_size = 64
        if args.output_dir is None:
            args.output_dir = './checkpoints_celeba'
        in_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Always train unconditionally: ignore dataset class labels.
    if is_main_rank and args.class_num not in (None, 1):
        print(f"Forcing unconditional mode: overriding class_num from {args.class_num} to 1")
    args.class_num = 1
    
    # Set resume path default if empty
    if not args.resume:
        resume_path = Path(args.output_dir) / f'checkpoint_{args.model_name.replace("/", "_")}.pth'
        if resume_path.exists():
            args.resume = str(resume_path)
    
    # Create output directory
    if is_main_rank:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if misc.is_dist_avail_and_initialized():
        dist.barrier()
    
    # Save configuration as JSON
    config = vars(args)
    model_name_safe = args.model_name.replace("/", "_")
    config_path = Path(args.output_dir) / f'config_{model_name_safe}.json'
    if is_main_rank:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")
    
    if is_main_rank:
        print("=" * 60)
        print(f"Diffusion Training - {args.dataset.upper()}")
        print("=" * 60)
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model_name}")
        print(f"Device: {device}")
        print(f"Image size: {args.img_size}x{args.img_size}")
        print(f"Channels: {in_channels}")
        print(f"Classes: {args.class_num}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"End learning rate: {args.end_lr}")
        print(f"Eval frequency: {args.eval_freq if args.eval_freq > 0 else 'disabled'}")
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
        # Prefer torchvision's native Flowers102 dataset to avoid HuggingFace/pyarrow issues on Windows.
        transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        try:
            dataset = datasets.Flowers102(
                root=args.data_path,
                split='train',
                download=True,
                transform=transform,
            )
        except Exception as exc:
            # Fallback to local prepared folders if they exist.
            local_train_dir = Path('./true_flowers_train')
            if local_train_dir.exists():
                dataset = datasets.ImageFolder(local_train_dir, transform=transform)
                if is_main_rank:
                    print(f"Flowers102 download failed ({exc}). Falling back to local folder: {local_train_dir}")
            else:
                raise RuntimeError(
                    "Failed to load Flowers dataset via torchvision Flowers102 and no local fallback was found at "
                    f"'{local_train_dir}'. Original error: {exc}"
                ) from exc
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

    if not isinstance(dataset, UnconditionalDatasetWrapper):
        dataset = UnconditionalDatasetWrapper(dataset, label=0)
    
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True,
            seed=args.seed,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.distributed,
    )
    
    if is_main_rank:
        print(f"Dataset size: {len(dataset)}")
    
    # Add in_channels to args for model initialization
    args.in_channels = in_channels
    model = Denoiser(args, model_name=args.model_name).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    model_without_ddp = model.module if args.distributed else model
    
    n_params = sum(p.numel() for p in model_without_ddp.parameters())
    if is_main_rank:
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Initialize EMA parameters
    model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
    model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model_without_ddp, optimizer, args.resume, device) + 1
        if misc.is_dist_avail_and_initialized():
            dist.barrier()
        print(f"Resumed from epoch {start_epoch}")

    scheduler = _build_lr_scheduler(
        optimizer=optimizer,
        start_lr=args.lr,
        end_lr=args.end_lr,
        total_epochs=args.epochs,
        start_epoch=start_epoch,
    )
    
    # Training loop
    if is_main_rank:
        print("\nStarting training...")
    model_name_safe = args.model_name.replace("/", "_")
    checkpoint_path = Path(args.output_dir) / f'checkpoint_{model_name_safe}.pth'
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        avg_loss = train_epoch(model, model_without_ddp, dataloader, optimizer, device, epoch, args)
        if is_main_rank:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f} | LR = {current_lr:.6g}')
        
        # Generate conditional samples (1 per class)
        if is_main_rank:
            output_image_path = Path(args.output_dir) / f'out_{model_name_safe}.png'
            generate_samples(
                model_without_ddp,
                device,
                output_image_path,
                num_classes=1,
                num_samples=10,
                class_label=0,
            )
            print(f'Generated samples saved to {output_image_path}')

        should_eval = args.eval_freq > 0 and ((epoch + 1) % args.eval_freq == 0)
        
        # Always save last checkpoint (overwrite)
        should_save = ((epoch + 1) % args.save_freq == 0) or (epoch == args.epochs - 1) or should_eval
        if should_save and is_main_rank:
            save_checkpoint(model_without_ddp, optimizer, epoch, checkpoint_path, config=config)

        # Keep all ranks in lockstep around external eval subprocess execution.
        if args.distributed:
            dist.barrier()

        if should_eval:
            if args.distributed:
                # In distributed mode, run eval directly
                run_distributed_eval(model_without_ddp, args, device, epoch)
            else:
                # In non-distributed mode, use subprocess
                if is_main_rank:
                    run_periodic_eval(args, device, epoch)
            
            if args.distributed:
                dist.barrier()

        scheduler.step()
    
    if is_main_rank:
        print("\nTraining completed!")
        print(f"Final checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
