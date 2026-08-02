"""
Test Diffusion Model - Generate samples
Supports multiple architectures: DRUNet and JiT models.
Runs unconditional generation and metrics evaluation.
"""

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from denoiser import Denoiser
from util import misc

torch.manual_seed(0)


class InMemoryUInt8ImageDataset(Dataset):
    """Torch dataset backed by in-RAM uint8 CHW images for torch_fidelity."""

    def __init__(self, images_u8):
        if not torch.is_tensor(images_u8):
            raise TypeError('images_u8 must be a torch.Tensor')
        if images_u8.dtype != torch.uint8:
            raise TypeError('images_u8 must use torch.uint8 dtype')
        if images_u8.ndim != 4:
            raise ValueError('images_u8 must have shape [N, C, H, W]')

        self.images_u8 = images_u8.contiguous()

    def __len__(self):
        return self.images_u8.shape[0]

    def __getitem__(self, index):
        return self.images_u8[index]


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


def _autocast_context(device, use_amp, amp_dtype):
    """Return autocast context for CUDA generation, or a no-op context otherwise."""
    if use_amp and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=amp_dtype)
    return nullcontext()


def _strip_module_prefix(state_dict):
    """Support checkpoints from both DDP and non-DDP training."""
    if any(name.startswith('module.') for name in state_dict.keys()):
        return {name.replace('module.', '', 1): value for name, value in state_dict.items()}
    return state_dict


def _choose_checkpoint_path(checkpoint_dir, model_name, checkpoint_suffix):
    """Resolve checkpoint path using a directly provided checkpoint suffix."""
    model_name_safe = model_name.replace('/', '_')
    model_variants = [model_name_safe]

    # Keep compatibility with legacy typo in existing CelebA checkpoints.
    if 'Celeba' in model_name_safe:
        model_variants.append(model_name_safe.replace('Celeba', 'Celeb'))

    suffix_aliases = {
        'sup': ['sup'],
        'sup_xloss': ['sup_xloss', 'sup'],
        'noisy': ['noisy'],
        'noise2noise': ['n2n', 'noise2noise'],
        'n2n': ['n2n', 'noise2noise'],
        'gr2r': ['gr2r'],
    }
    suffix_variants = suffix_aliases.get(checkpoint_suffix, [checkpoint_suffix])

    candidates = []
    for model_variant in model_variants:
        for suffix in suffix_variants:
            candidates.append(checkpoint_dir / f'checkpoint_{model_variant}_{suffix}.pth')
        candidates.append(checkpoint_dir / f'checkpoint_{model_variant}.pth')

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No checkpoint found for model={model_name}, checkpoint_suffix={checkpoint_suffix} in {checkpoint_dir}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def _choose_config_path(checkpoint_dir, model_name):
    """Resolve config path with compatibility for legacy naming variants."""
    model_name_safe = model_name.replace('/', '_')
    candidates = [checkpoint_dir / f'config_{model_name_safe}.json']
    if 'Celeba' in model_name_safe:
        candidates.append(checkpoint_dir / f"config_{model_name_safe.replace('Celeba', 'Celeb')}.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _load_runtime_config(checkpoint, config_path):
    """Load config from checkpoint when available, else from config file."""
    if 'config' in checkpoint:
        print('Loaded config from checkpoint')
        return checkpoint['config']
    if config_path is not None:
        with open(config_path, 'r') as f:
            print(f'Loaded config from {config_path}')
            return json.load(f)
    raise FileNotFoundError('Config not found in checkpoint and no config file was resolved')


def _apply_checkpoint_to_model(model, checkpoint, device, weight_source='auto'):
    """Load model params and optionally swap runtime weights to EMA params."""
    model_state = _strip_module_prefix(checkpoint['model'])
    model.load_state_dict(model_state)

    ema_state_dict1 = _strip_module_prefix(checkpoint['model_ema1'])
    ema_state_dict2 = _strip_module_prefix(checkpoint['model_ema2'])
    model.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model.named_parameters()]
    model.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model.named_parameters()]

    effective_source = weight_source
    if weight_source == 'auto':
        # For denoising objectives, EMA tends to be more stable at generation time.
        effective_source = 'ema1' if model.loss in {'noisy', 'noise2noise', 'gr2r'} else 'raw'

    if effective_source == 'ema1':
        selected_params = model.ema_params1
    elif effective_source == 'ema2':
        selected_params = model.ema_params2
    else:
        selected_params = None

    if selected_params is not None:
        for param, ema_param in zip(model.parameters(), selected_params):
            param.data.copy_(ema_param.data)

    return effective_source


def load_model(checkpoint_path, device, config_path, weight_source='auto'):
    """Load trained model and return (model, runtime_config, checkpoint_epoch)."""
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = _load_runtime_config(checkpoint, config_path)

    class Args:
        pass

    args = Args()
    for key, value in config.items():
        setattr(args, key, value)

    model_name = args.model_name
    print(f"Using model architecture: {model_name}")

    model = Denoiser(args, model_name=model_name).to(device)
    loaded_weight_source = _apply_checkpoint_to_model(model, checkpoint, device, weight_source=weight_source)

    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Using weight source: {loaded_weight_source}")

    return model, config, checkpoint['epoch']


def _append_results_csv(results_path, row):
    """Append one experiment row to results.csv, creating header on first write."""
    fieldnames = ['dataset', 'model', 'epoch', 'pred', 'w', 'learning rate', 'batch size', 'fid']
    write_header = not results_path.exists()

    with open(results_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _is_a100(device):
    """Detect whether the selected CUDA device is an NVIDIA A100."""
    if device.type != 'cuda':
        return False

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    return 'A100' in device_name.upper()


def _maybe_compile_for_generation(model, args, device):
    """Compile JiT model with reduce-overhead mode for faster repeated generation."""
    if not args.torch_compile:
        return model, False
    if device.type != 'cuda' or not args.model_name.startswith('JiT'):
        return model, False
    if not hasattr(torch, 'compile'):
        print('torch.compile is not available in this PyTorch build; using eager mode')
        return model, False

    try:
        compiled_model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
        return compiled_model, True
    except Exception as exc:
        print(f'torch.compile failed ({exc}); falling back to eager mode')
        return model, False


@torch.no_grad()
def generate_samples(
    model,
    num_samples=9,
    cfg_scale=1.0,
    device='cuda',
    class_label=0,
    use_amp=False,
    amp_dtype=torch.float16,
):
    """Generate unconditional samples with a fixed dummy class label."""
    print(f"\nGenerating {num_samples} samples (CFG scale: {cfg_scale})...")

    target_device = torch.device(device) if not isinstance(device, torch.device) else device

    original_cfg = model.cfg_scale
    model.cfg_scale = cfg_scale

    labels = torch.full((num_samples,), class_label, device=target_device, dtype=torch.long)

    with _autocast_context(target_device, use_amp=use_amp, amp_dtype=amp_dtype):
        samples = model.generate(labels)

    model.cfg_scale = original_cfg

    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    return samples, labels


def save_image_grid(images, path, nrow=3):
    """Save a grid of generated images."""
    save_image(images, path, nrow=nrow, padding=2)
    print(f"Saved: {path}")


def _build_celeba_eval_dataset(data_path, img_size):
    """Build deterministic CelebA eval dataset consistent with train preprocessing."""
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return datasets.CelebA(
        root=data_path,
        split='train',
        download=True,
        transform=transform,
    )


def _build_flowers_eval_dataset(data_path, img_size):
    """Build deterministic Flowers102 train eval dataset."""
    transform = transforms.Compose([
        # Keep train-time geometry while removing random augmentations for stable FID.
        transforms.Resize((150, 150)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
    ])
    try:
        return datasets.Flowers102(
            root=data_path,
            split='train',
            download=True,
            transform=transform,
        )
    except Exception as exc:
        local_train_dir = Path('./true_flowers_train')
        if local_train_dir.exists():
            print(f"Flowers102 download failed ({exc}). Falling back to local folder: {local_train_dir}")
            return datasets.ImageFolder(local_train_dir, transform=transform)
        raise RuntimeError(
            "Failed to load Flowers train split via torchvision Flowers102 and no local fallback was found at "
            f"'{local_train_dir}'. Original error: {exc}"
        ) from exc


def _build_eval_dataset(dataset_name, data_path, img_size):
    """Build deterministic eval dataset for metric computation."""
    if dataset_name == 'celeba':
        return _build_celeba_eval_dataset(data_path, img_size)
    if dataset_name == 'flowers':
        return _build_flowers_eval_dataset(data_path, img_size)
    raise ValueError(f'Metrics mode does not support dataset={dataset_name}')


def _fixed_random_indices(total_size, num_samples, seed):
    """Return reproducible random indices without replacement."""
    if num_samples <= 0:
        raise ValueError('num_samples must be > 0')
    if num_samples > total_size:
        raise ValueError(f'num_samples ({num_samples}) exceeds dataset size ({total_size})')

    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total_size, generator=generator)[:num_samples].tolist()


def _save_real_subset_images(real_dir, dataset, num_real_images, seed):
    """Save a fixed random subset of real images for metric computation."""
    if num_real_images == -1:
        indices = range(len(dataset))
    else:
        indices = _fixed_random_indices(len(dataset), num_real_images, seed)

    for i, idx in enumerate(indices):
        image, _ = dataset[idx]
        save_image(image, real_dir / f'real_{i:06d}.png')


def _resolve_real_cache_dir(dataset_name, num_real_images):
    """Return the persistent cache directory for real train images."""
    if dataset_name == 'celeba':
        if num_real_images == -1:
            dir_name = 'real_celeba_full'
        else:
            dir_name = f'real_celeba_{num_real_images}imgs'
    elif dataset_name == 'flowers':
        if num_real_images == -1:
            dir_name = 'real_flowers_train_full'
        else:
            dir_name = f'real_flowers_train_{num_real_images}imgs'
    else:
        raise ValueError(f'Metrics cache path is not supported for dataset={dataset_name}')

    # Keep cache location anchored to this repository's local ./data folder.
    path = Path(__file__).resolve().parent / 'data' / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _count_cached_real_images(real_dir):
    """Count cached real image files."""
    return len(list(real_dir.glob('real_*.png')))


def _gather_shard_payload_to_rank0(payload):
    """Gather shard payloads to rank 0, with all_gather_object fallback."""
    if not misc.is_dist_avail_and_initialized():
        return [payload]

    world_size = misc.get_world_size()
    rank = misc.get_rank()

    gather_list = [None for _ in range(world_size)] if rank == 0 else None
    try:
        dist.gather_object(
            obj=payload,
            object_gather_list=gather_list,
            dst=0,
        )
        return gather_list if rank == 0 else None
    except Exception as exc:
        if rank == 0:
            print(f'dist.gather_object failed ({exc}); falling back to dist.all_gather_object')
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, payload)
        return gathered if rank == 0 else None


@torch.no_grad()
def _generate_ram_dataset(
    model,
    num_gen_images,
    batch_size,
    cfg_scale,
    device,
    class_label=0,
    use_amp=False,
    amp_dtype=torch.float16,
):
    """Generate images in distributed shards and assemble rank-0 RAM dataset."""
    if num_gen_images <= 0:
        raise ValueError('num_gen_images must be > 0')
    if batch_size <= 0:
        raise ValueError('metrics_batch_size must be > 0')

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            'tqdm is required to show generation progress. Install it with: pip install tqdm'
        ) from exc

    rank = misc.get_rank()
    world_size = misc.get_world_size()
    is_main = misc.is_main_process()

    original_cfg = model.cfg_scale
    model.cfg_scale = cfg_scale

    global_batch_size = batch_size * world_size
    num_steps = (num_gen_images + global_batch_size - 1) // global_batch_size
    gathered_shards = [] if is_main else None

    if is_main and num_gen_images > 0:
        channels = model.img_channels if hasattr(model, 'img_channels') else 3
        approx_bytes = num_gen_images * channels * model.img_size * model.img_size
        approx_gib = approx_bytes / (1024 ** 3)
        print(f'Estimated RAM for generated uint8 tensors: {approx_gib:.2f} GiB')
        print(f'Distributed fake generation: world_size={world_size}, per_rank_batch={batch_size}')

    try:
        with tqdm(
            total=num_gen_images,
            desc='Generating fake images in RAM (all ranks)',
            unit='img',
            disable=not is_main,
        ) as pbar:
            for step in range(num_steps):
                global_start = step * global_batch_size
                global_end = min(global_start + global_batch_size, num_gen_images)

                local_start = global_start + rank * batch_size
                local_end = min(local_start + batch_size, global_end)

                local_payload = None
                if local_start < global_end:
                    cur_batch = local_end - local_start
                    labels = torch.full((cur_batch,), class_label, device=device, dtype=torch.long)

                    with _autocast_context(device, use_amp=use_amp, amp_dtype=amp_dtype):
                        samples = model.generate(labels)

                    samples_u8 = ((samples + 1) / 2).clamp(0, 1)
                    samples_u8 = samples_u8.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

                    # Immediately transfer each rank shard to host RAM to avoid GPU OOM.
                    local_payload = (local_start, samples_u8.cpu())
                    del samples
                    del samples_u8

                step_payloads = _gather_shard_payload_to_rank0(local_payload)

                if is_main:
                    generated_this_step = 0
                    active_ranks = []
                    for rank_idx, shard_payload in enumerate(step_payloads):
                        if shard_payload is None:
                            continue
                        shard_start, shard_images = shard_payload
                        gathered_shards.append((shard_start, shard_images))
                        generated_this_step += shard_images.shape[0]
                        active_ranks.append(rank_idx)

                    #if active_ranks:
                    #    print(
                    #        f'Generation step {step + 1}/{num_steps}: '
                    #        f'active_ranks={active_ranks}, generated={generated_this_step}'
                    #    )

                    pbar.update(generated_this_step)
    finally:
        model.cfg_scale = original_cfg

    if not is_main:
        return None

    gathered_shards.sort(key=lambda item: item[0])
    if not gathered_shards:
        raise RuntimeError('No fake-image shards were gathered on rank 0')

    expected_start = 0
    ordered_batches = []
    for shard_start, shard_images in gathered_shards:
        if shard_start != expected_start:
            raise RuntimeError(
                f'Gathered shard ordering mismatch at index {expected_start}: '
                f'got shard starting at {shard_start}'
            )
        ordered_batches.append(shard_images)
        expected_start += shard_images.shape[0]

    ram_images_u8 = torch.cat(ordered_batches, dim=0)
    if ram_images_u8.shape[0] != num_gen_images:
        raise RuntimeError(
            f'Generated image count mismatch: expected {num_gen_images}, got {ram_images_u8.shape[0]}'
        )

    print(f'Assembled rank-0 in-memory fake dataset with {ram_images_u8.shape[0]} samples')
    return InMemoryUInt8ImageDataset(ram_images_u8)


def compute_fid_metrics(model, args, device, cfg_scale, use_amp=False, amp_dtype=torch.float16):
    """Compute FID with fixed random real subset and generated samples."""
    if args.dataset not in {'celeba', 'flowers'}:
        raise ValueError('Metrics mode is currently implemented only for --dataset celeba or --dataset flowers')

    rank = misc.get_rank()
    is_main = misc.is_main_process()
    is_distributed = misc.is_dist_avail_and_initialized()

    real_dir = None

    if is_main:
        eval_dataset = _build_eval_dataset(args.dataset, args.data_path, model.img_size)
        resolved_num_real_images = len(eval_dataset) if args.num_real_images == -1 else args.num_real_images
        metrics_cfg_scale = 1.0
        metrics_mode = 'unconditional'

        print('\n' + '=' * 60)
        print('Computing FID from in-memory generated tensors (distributed -> rank 0)')
        print(f'Dataset: {args.dataset} (train split)')
        print(f'Real images: {resolved_num_real_images}')
        print(f'Generated images: {args.num_gen_images}')
        print(f'Generation mode for metrics: {metrics_mode}')
        print(f'Metrics cfg_scale: {metrics_cfg_scale}')
        print(f'Metrics seed (base): {args.metrics_seed}')
        print('Metrics seeding policy: per-rank offset (seed = base + rank)')
        print(f'Metrics batch size per rank: {args.metrics_batch_size}')
        print('=' * 60)

        real_dir = _resolve_real_cache_dir(args.dataset, args.num_real_images)

        cached_real_count = _count_cached_real_images(real_dir)

        if cached_real_count == resolved_num_real_images:
            print(f"Using cached real images in {real_dir} ({cached_real_count} images)")
        else:
            if cached_real_count > 0:
                print(
                    f"Cached real images count mismatch in {real_dir}: "
                    f"found {cached_real_count}, expected {resolved_num_real_images}. Rebuilding cache."
                )
                for stale_file in real_dir.glob('real_*.png'):
                    stale_file.unlink()

            _save_real_subset_images(real_dir, eval_dataset, args.num_real_images, args.metrics_seed)
            print(f"Saved real images to {real_dir}")

    if is_distributed:
        dist.barrier()

    # Make generation deterministic and explicit across distributed ranks.
    metrics_seed = args.metrics_seed + rank
    torch.manual_seed(metrics_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(metrics_seed)

    metrics_cfg_scale = 1.0

    fake_dataset = _generate_ram_dataset(
        model=model,
        num_gen_images=args.num_gen_images,
        batch_size=args.metrics_batch_size,
        cfg_scale=metrics_cfg_scale,
        device=device,
        class_label=0,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    metrics_dict = None
    try:
        if is_main:
            try:
                import torch_fidelity
            except ImportError as exc:
                raise ImportError(
                    'torch_fidelity is required for --compute_metrics. Install it with: pip install torch-fidelity'
                ) from exc

            print(f'Generated in-memory dataset with {len(fake_dataset)} samples')
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=fake_dataset,
                input2=str(real_dir),
                cuda=(device.type == 'cuda'),
                isc=False,
                fid=True,
                kid=False,
                prc=False,
                verbose=True,
            )
    finally:
        if is_distributed:
            dist.barrier()

    if not is_main:
        return None

    fid = metrics_dict['frechet_inception_distance']
    #is_mean = metrics_dict['inception_score_mean']
    # is_std = metrics_dict.get('inception_score_std', None)

    print('\n-- Metrics summary --')
    print(f'FID: {fid}')

    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description='Test Diffusion Model')

    parser.add_argument('--dataset', type=str, default='celeba', choices=['mnist', 'flowers', 'celeba'],
                        help='Dataset to generate from (mnist, flowers, or celeba)')
    parser.add_argument('--model_name', type=str, default='JiT-CelebaB',
                        choices=['DRUnet', 'JiT-B/4', 'JiT-Flowers', 'JiT-Celeba', 'JiT-CelebaB'],
                        help='Model architecture used during training')
    parser.add_argument('--checkpoint_suffix', type=str, default='',
                        choices=['sup_xloss', 'sup', 'noisy', 'noise2noise', 'n2n', 'gr2r'],
                        help='Checkpoint suffix to load (e.g. sup_xloss, sup, n2n, gr2r)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Generation settings (3x3 by default)
    parser.add_argument('--grid_size', type=int, default=3,
                        help='Grid dimension (3 -> 3x3)')
    parser.add_argument('--num_sampling_steps', type=int, default=50,
                        help='Sampling steps override (defaults to checkpoint config)')
    parser.add_argument('--cfg_scale', type=float, default=None,
                        help='CFG scale override (ignored; unconditional mode always uses 1.0)')
    parser.add_argument('--sampling_method', type=str, default='euler',
                        choices=['euler', 'heun'], help='ODE sampling method')
    parser.add_argument('--weight_source', type=str, default='raw', choices=['auto', 'raw', 'ema1', 'ema2'],
                        help='Runtime weights source: raw model, ema1, ema2, or auto by loss')
    parser.add_argument('--mixed_precision', dest='mixed_precision', action='store_true',
                        help='Enable CUDA autocast during generation (default: enabled)')
    parser.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false',
                        help='Disable CUDA autocast during generation')
    parser.set_defaults(mixed_precision=True)
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['auto', 'fp16', 'bf16'],
                        help='Autocast dtype for generation when mixed precision is enabled')
    parser.add_argument('--torch_compile', dest='torch_compile', action='store_true',
                        help='Enable torch.compile(mode=reduce-overhead) for JiT CUDA generation')
    parser.add_argument('--no_torch_compile', dest='torch_compile', action='store_false',
                        help='Disable torch.compile generation optimization')
    parser.set_defaults(torch_compile=True)

    # Data and metrics settings
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset root (used by metrics for real train images)')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute FID metrics (supported: celeba, flowers)')
    parser.set_defaults(compute_metrics=True)
    parser.add_argument('--num_real_images', type=int, default=-1,
                        help='Fixed number of random real train images for metrics; use -1 for full train split')
    parser.add_argument('--num_gen_images', type=int, default=52000,
                        help='Fixed number of generated images for metrics')
    parser.add_argument('--metrics_batch_size', type=int, default=500,
                        help='Generation batch size for metrics computation')
    parser.add_argument('--metrics_seed', type=int, default=42,
                        help='Seed used for fixed random real subset and generated samples')
    parser.add_argument('--skip_grid', action='store_true',
                        help='Skip saving the qualitative 3x3 grid image')

    # Reproducibility / distributed generation
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank for torchrun')
    parser.add_argument('--dist_on_itp', action='store_true', help='Distributed launch on ITP')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')

    # pred mode args
    parser.add_argument('--pred', type=str, default='x', choices=['x', 'v'], help='parameterization: x or v')
    parser.add_argument('--w', type=str, default='v', choices=['x', 'v'], help='Loss weight: x or v')

    args = parser.parse_args()

    if args.grid_size <= 0:
        raise ValueError('grid_size must be > 0')
    if args.num_real_images == 0 or args.num_real_images < -1:
        raise ValueError('num_real_images must be > 0, or -1 to use all train images')
    if args.num_gen_images <= 0:
        raise ValueError('num_gen_images must be > 0')
    if args.metrics_batch_size <= 0:
        raise ValueError('metrics_batch_size must be > 0')

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

    # Create output directory
    model_name_safe = args.model_name.replace("/", "_")
    output_dir = Path(f"./generated_{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(f"./checkpoints_{args.dataset}")
    checkpoint = _choose_checkpoint_path(checkpoint_dir, args.model_name, args.checkpoint_suffix)
    config_path = _choose_config_path(checkpoint_dir, args.model_name)

    if misc.is_main_process():
        print("=" * 60)
        print(f"{args.dataset.upper()} Diffusion Model - Image Generation")
        print("=" * 60)
        print(f"Model: {args.model_name}")
        print(f"Checkpoint suffix: {args.checkpoint_suffix}")
        print(f"Device: {device}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Config: {config_path}")
        print(f"Sampling method: {args.sampling_method}")
        print(f"Sampling steps: {args.num_sampling_steps if args.num_sampling_steps is not None else 'from config'}")
        print(f"CFG scale: {args.cfg_scale if args.cfg_scale is not None else 'from config'}")
        print(f"Weight source: {args.weight_source}")
        print(f"Compute metrics: {args.compute_metrics}")
        print("=" * 60)

    if device.type == 'cpu' and args.model_name.startswith('JiT'):
        raise ValueError(
            'JiT models currently require CUDA in this codebase due to CUDA-bound RoPE buffers. '
            'Use --device cuda, or use DRUnet for CPU runs.'
        )

    use_amp = bool(args.mixed_precision and device.type == 'cuda')
    amp_dtype, amp_dtype_label = _resolve_amp_dtype(args.amp_dtype, device)
    if args.mixed_precision and device.type != 'cuda' and misc.is_main_process():
        print('Mixed precision requested but CUDA is not active; running fp32 generation')
    if misc.is_main_process():
        print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")
        print(f"AMP dtype: {amp_dtype_label if use_amp else 'n/a'}")
        if args.torch_compile and device.type == 'cuda' and args.model_name.startswith('JiT'):
            print('torch.compile: enabled (mode=reduce-overhead)')
            if _is_a100(device):
                print('Detected A100 GPU; enabling ViT-oriented compile path')
        else:
            print('torch.compile: disabled')

    model, runtime_config, checkpoint_epoch = load_model(
        checkpoint,
        device,
        config_path,
        weight_source=args.weight_source,
    )

    # Update sampling parameters
    model.method = args.sampling_method
    if args.num_sampling_steps is not None:
        model.steps = args.num_sampling_steps

    model, compile_enabled = _maybe_compile_for_generation(model, args, device)
    if misc.is_main_process() and args.torch_compile and args.model_name.startswith('JiT'):
        print(f'torch.compile active: {compile_enabled}')

    if args.cfg_scale not in (None, 1.0) and misc.is_main_process():
        print(f'Forcing unconditional mode: overriding cfg_scale from {args.cfg_scale} to 1.0')
    cfg_scale = 1.0
    output_path = None

    if not args.skip_grid and misc.is_main_process():
        num_samples = args.grid_size * args.grid_size
        samples, labels = generate_samples(
            model,
            num_samples=num_samples,
            cfg_scale=cfg_scale,
            device=device,
            class_label=0,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        output_path = output_dir / f'grid_{model_name_safe}_{args.checkpoint_suffix}_{args.grid_size}x{args.grid_size}.png'
        save_image_grid(samples, output_path, nrow=args.grid_size)
        print(f"Labels used: {labels.tolist()}")

    if args.compute_metrics:
        metrics_dict = compute_fid_metrics(
            model=model,
            args=args,
            device=device,
            cfg_scale=cfg_scale,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        if misc.is_main_process() and metrics_dict is not None:
            fid = metrics_dict.get('frechet_inception_distance')
            results_row = {
                'dataset': args.dataset,
                'model': runtime_config.get('model_name', args.model_name),
                'epoch': checkpoint_epoch,
                'pred': runtime_config.get('pred', args.pred),
                'w': runtime_config.get('w', args.w),
                'learning rate': runtime_config.get('lr', ''),
                'batch size': runtime_config.get('batch_size', ''),
                'fid': fid,
            }
            results_path = Path(__file__).resolve().parent / 'results.csv'
            _append_results_csv(results_path, results_row)
            print(f'Results appended to: {results_path}')

    if misc.is_main_process():
        print("\n" + "=" * 60)
        print("Generation completed!")
        if output_path is not None:
            print(f"Image saved to: {output_path}")
        print("=" * 60)


if __name__ == '__main__':
    main()
