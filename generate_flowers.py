
import argparse
import torch
from pathlib import Path
import json
from denoiser import Denoiser
import copy
from torchvision.utils import save_image
from tqdm import tqdm
import math
import os

torch.manual_seed(0)

def load_checkpoint(model, path, device):
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

    return model


def load_model(checkpoint_path, device, config_path):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load config from checkpoint first, then from file
    config = None
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Loaded config from checkpoint")
    elif config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        raise FileNotFoundError(f"Config file not found at {config_path} and not embedded in checkpoint")
    
    # Create args object from config
    class Args:
        pass
    
    args = Args()
    for key, value in config.items():
        setattr(args, key, value)
    
    # Get model_name from config
    model_name = args.model_name
    print(f"Using model architecture: {model_name}")
    
    # Create model
    model = Denoiser(args, model_name=model_name).to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    return model

@torch.no_grad()
def generate_batch(model, labels, cfg_scale=1.0, device='cuda'):
    """Generate conditional samples for a batch of labels."""
    # Set CFG scale
    original_cfg = model.cfg_scale
    model.cfg_scale = cfg_scale
    
    # Generate
    samples = model.generate(labels)
    
    # Restore original CFG
    model.cfg_scale = original_cfg
    
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    return samples

def main():
    parser = argparse.ArgumentParser(description="Generate fake flowers for FID evaluation.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_flowers/checkpoint_last.pth", help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional if in checkpoint).")
    parser.add_argument("--output_dir", type=str, default="fake_flowers", help="Directory found save generated images.")
    parser.add_argument("--n_images", type=int, default=10, help="Number of images per class to generate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation.")
    parser.add_argument("--cfg_scale", type=float, default=1.5, help="Classifier-free guidance scale.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Ensure checkpoint path is correct
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Optional: try to prepend checkpoints_flowers if not found
        alt_path = os.path.join('checkpoints_flowers', checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
             print(f"Checkpoint not found: {checkpoint_path}")
             return

    # Load Model
    try:
        model = load_model(checkpoint_path, device, args.config)
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return
    
    num_classes = model.num_classes
    print(f"Model has {num_classes} classes.")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Generating {args.n_images} images per class (Total: {args.n_images * num_classes})...")
    print(f"CFG Scale: {args.cfg_scale}")
    
    # Create generation tasks
    tasks = []
    for c in range(num_classes):
        for i in range(args.n_images):
            filename = f"class_{c}_img_{i}.png"
            tasks.append((c, filename))
            
    # Process in batches
    total_images = len(tasks)
    num_batches = math.ceil(total_images / args.batch_size)
    
    for b in tqdm(range(num_batches), desc="Generating batches"):
        start_idx = b * args.batch_size
        end_idx = min((b + 1) * args.batch_size, total_images)
        
        batch_tasks = tasks[start_idx:end_idx]
        batch_labels = torch.tensor([t[0] for t in batch_tasks], device=device, dtype=torch.long)
        
        # Generate
        images = generate_batch(model, batch_labels, cfg_scale=args.cfg_scale, device=device)
        
        # Save images
        for i, (label, filename) in enumerate(batch_tasks):
            save_path = os.path.join(args.output_dir, filename)
            save_image(images[i], save_path)
            
    print(f"Done. Saved {total_images} images to '{args.output_dir}'.")

if __name__ == "__main__":
    main()
