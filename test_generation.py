"""
Visual generation evaluation for CelebA.

Loads a trained diffusion model from checkpoints/config using a direct
checkpoint suffix selector, then captures 10 equally
spaced snapshots along the generation trajectory and saves a grid with
rows=image index and columns=timestep.
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from denoiser import Denoiser


torch.manual_seed(0)


def _strip_module_prefix(state_dict):
	"""Support checkpoints from both DDP and non-DDP training."""
	if any(name.startswith('module.') for name in state_dict.keys()):
		return {name.replace('module.', '', 1): value for name, value in state_dict.items()}
	return state_dict


def _choose_checkpoint_path(checkpoint_dir, model_name, suffix=None):
	"""Resolve checkpoint path using an optional direct checkpoint suffix."""
	model_name_safe = model_name.replace('/', '_')
	model_variants = [model_name_safe]

	if 'Celeba' in model_name_safe:
		model_variants.append(model_name_safe.replace('Celeba', 'Celeb'))

	candidates = []
	for model_variant in model_variants:
		if suffix:
			candidates.append(checkpoint_dir / f'checkpoint_{model_variant}_{suffix}.pth')
		candidates.append(checkpoint_dir / f'checkpoint_{model_variant}.pth')

	for candidate in candidates:
		if candidate.exists():
			return candidate

	raise FileNotFoundError(
		f"No checkpoint found for model={model_name}, suffix={suffix} in {checkpoint_dir}. "
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
		with open(config_path, 'r') as file_handle:
			print(f'Loaded config from {config_path}')
			return json.load(file_handle)
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
	"""Load trained model from checkpoint with config."""
	print(f'Loading checkpoint from {checkpoint_path}')
	checkpoint = torch.load(checkpoint_path, map_location=device)

	config = _load_runtime_config(checkpoint, config_path)

	class Args:
		pass

	args = Args()
	for key, value in config.items():
		setattr(args, key, value)

	model_name = args.model_name
	print(f'Using model architecture: {model_name}')

	model = Denoiser(args, model_name=model_name).to(device)
	loaded_weight_source = _apply_checkpoint_to_model(model, checkpoint, device, weight_source=weight_source)
	model.eval()

	print(f"Model loaded from epoch {checkpoint['epoch']}")
	print(f'Using weight source: {loaded_weight_source}')
	return model


def _build_snapshot_indices(num_steps, num_snapshots):
	"""Build equally spaced trajectory indices including first and final states."""
	if num_steps < num_snapshots:
		raise ValueError(f'num_steps must be >= {num_snapshots} to capture unique equally spaced snapshots')
	if num_snapshots < 2:
		raise ValueError('num_snapshots must be >= 2')

	return [(i * num_steps) // (num_snapshots - 1) for i in range(num_snapshots)]


@torch.no_grad()
def generate_trajectory_snapshots(model, labels, cfg_scale, num_steps, num_snapshots=10):
	"""Generate and return z/x_pred snapshots [T, N, C, H, W] over the trajectory."""
	original_cfg = model.cfg_scale
	original_steps = model.steps

	model.cfg_scale = cfg_scale
	model.steps = num_steps

	try:
		device = labels.device
		batch_size = labels.size(0)
		z = model.noise_scale * torch.randn(
			batch_size,
			model.in_channels,
			model.img_size,
			model.img_size,
			device=device,
		)

		timesteps = torch.linspace(0.0, 1.0, model.steps + 1, device=device)
		timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, batch_size, -1, -1, -1)

		if model.method == 'euler':
			stepper = model._euler_step
		elif model.method == 'heun':
			stepper = model._heun_step
		else:
			raise NotImplementedError(f'Unsupported sampling method: {model.method}')

		capture_indices = _build_snapshot_indices(num_steps=model.steps, num_snapshots=num_snapshots)
		capture_set = set(capture_indices)
		captured_z = {}
		captured_x_pred = {}

		def _capture_state(current_state_index, current_z, current_t):
			if current_state_index in capture_set:
				v_pred = model._forward_sample(current_z, current_t, labels)
				x_pred = current_z + (1.0 - current_t) * v_pred
				captured_z[current_state_index] = current_z.clone()
				captured_x_pred[current_state_index] = x_pred.clone()

		# State index 0 corresponds to the initial latent noise.
		state_index = 0
		_capture_state(state_index, z, timesteps[0])

		for i in range(model.steps - 1):
			z = stepper(z, timesteps[i], timesteps[i + 1], labels)
			state_index += 1
			_capture_state(state_index, z, timesteps[i + 1])

		# Keep last step consistent with Denoiser.generate.
		z = model._euler_step(z, timesteps[-2], timesteps[-1], labels)
		state_index += 1
		_capture_state(state_index, z, timesteps[-1])

		z_snapshots = torch.stack([captured_z[idx] for idx in capture_indices], dim=0)
		x_pred_snapshots = torch.stack([captured_x_pred[idx] for idx in capture_indices], dim=0)

		z_snapshots = ((z_snapshots + 1.0) / 2.0).clamp(0.0, 1.0)
		x_pred_snapshots = ((x_pred_snapshots + 1.0) / 2.0).clamp(0.0, 1.0)

		return z_snapshots, x_pred_snapshots, capture_indices
	finally:
		model.cfg_scale = original_cfg
		model.steps = original_steps


def _build_labels(model, num_images, device, class_label=0):
	"""CelebA-compatible label logic, aligned with test_diffusion.py."""
	if model.num_classes <= 1:
		return torch.full((num_images,), class_label, device=device, dtype=torch.long)
	return torch.arange(num_images, device=device, dtype=torch.long) % model.num_classes


def save_trajectory_grid(z_snapshots, x_pred_snapshots, output_path):
	"""Save interleaved rows: z then x_pred for each image, cols=timesteps."""
	num_snapshots, num_images, channels, height, width = z_snapshots.shape
	z_rows = z_snapshots.permute(1, 0, 2, 3, 4)
	x_pred_rows = x_pred_snapshots.permute(1, 0, 2, 3, 4)
	interleaved = torch.stack((z_rows, x_pred_rows), dim=1)
	grid_images = interleaved.reshape(num_images * 2 * num_snapshots, channels, height, width)
	save_image(grid_images, output_path, nrow=num_snapshots, padding=2)
	print(f'Saved: {output_path}')


def main():
	parser = argparse.ArgumentParser(description='Visual trajectory evaluation for CelebA generation')
	parser.add_argument('--dataset', type=str, default='celeba', choices=['celeba'],
						help='Dataset to evaluate (CelebA-only in this script)')
	parser.add_argument('--model_name', type=str, default='JiT-CelebaB',
						choices=['DRUnet', 'JiT-Celeba', 'JiT-CelebaB'],
						help='Model architecture used during training')
	parser.add_argument('--suffix', type=str, default='',
						help='Checkpoint suffix used directly in checkpoint_<model>_<suffix>.pth')
	parser.add_argument('--loss', dest='suffix', type=str,
						help='Deprecated alias of --suffix')
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
						help='Device for inference')
	parser.add_argument('--sampling_method', type=str, default='euler', choices=['euler', 'heun'],
						help='ODE sampling method')
	parser.add_argument('--cfg_scale', type=float, default=None,
						help='CFG scale override (defaults to checkpoint config)')
	parser.add_argument('--weight_source', type=str, default='ema1', choices=['auto', 'raw', 'ema1', 'ema2'],
						help='Runtime weights source: raw model, ema1, ema2, or auto by loss')
	parser.add_argument('--num_steps', type=int, default=50,
						help='Number of generation steps (defaults to checkpoint config)')
	parser.add_argument('--num_images', type=int, default=1,
						help='Number of generated images (rows in output grid)')
	parser.add_argument('--num_snapshots', type=int, default=5,
						help='Number of equally spaced snapshots over trajectory')
	parser.add_argument('--seed', type=int, default=0,
						help='Random seed for deterministic generation')

	# pred mode args
	parser.add_argument('--pred', type=str, default='v', choices=['x', 'v'], help='parameterization: x or v')
	parser.add_argument('--w', type=str, default='v', choices=['x', 'v'], help='Loss weight: x or v')

	args = parser.parse_args()

	if args.num_images <= 0:
		raise ValueError('num_images must be > 0')
	if args.num_snapshots <= 1:
		raise ValueError('num_snapshots must be > 1')

	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device(args.device)
	if device.type == 'cpu' and args.model_name.startswith('JiT'):
		raise ValueError(
			'JiT models currently require CUDA in this codebase due to CUDA-bound RoPE buffers. '
			'Use --device cuda, or use DRUnet for CPU runs.'
		)

	model_name_safe = args.model_name.replace('/', '_')
	output_dir = Path('./generated_celeba')
	output_dir.mkdir(parents=True, exist_ok=True)

	checkpoint_dir = Path('./checkpoints_celeba')
	checkpoint_path = _choose_checkpoint_path(checkpoint_dir, args.model_name, args.suffix)
	config_path = _choose_config_path(checkpoint_dir, args.model_name)

	print('=' * 60)
	print('CELEBA Diffusion - Trajectory Visualization')
	print('=' * 60)
	print(f'Model: {args.model_name}')
	print(f'Suffix: {args.suffix}')
	print(f'Device: {args.device}')
	print(f'Checkpoint: {checkpoint_path}')
	print(f'Config: {config_path}')
	print(f'Sampling method: {args.sampling_method}')
	print(f'Num steps: {args.num_steps if args.num_steps is not None else "from config"}')
	print(f'CFG scale: {args.cfg_scale if args.cfg_scale is not None else "from config"}')
	print(f'Weight source: {args.weight_source}')
	print(f'Num images: {args.num_images}')
	print(f'Num snapshots: {args.num_snapshots}')
	print(f'Seed: {args.seed}')
	print('=' * 60)

	model = load_model(checkpoint_path, device, config_path, weight_source=args.weight_source)

	model.method = args.sampling_method
	if args.num_steps is not None:
		model.steps = args.num_steps

	if model.steps < args.num_snapshots:
		raise ValueError(
			f'num_steps must be >= num_snapshots ({args.num_snapshots}). '
			f'Current num_steps={model.steps}'
		)

	cfg_scale = args.cfg_scale if args.cfg_scale is not None else model.cfg_scale
	labels = _build_labels(model, num_images=args.num_images, device=device, class_label=0)

	z_snapshots, x_pred_snapshots, snapshot_indices = generate_trajectory_snapshots(
		model=model,
		labels=labels,
		cfg_scale=cfg_scale,
		num_steps=model.steps,
		num_snapshots=args.num_snapshots,
	)

	output_path = output_dir / (
		f'trajectory_{model_name_safe}_{args.suffix}_imgs{args.num_images}_'
		f'steps{model.steps}_snaps{args.num_snapshots}.png'
	)
	save_trajectory_grid(z_snapshots, x_pred_snapshots, output_path)

	print(f'Labels used: {labels.tolist()}')
	print(f'Snapshot indices over states [0..{model.steps}]: {snapshot_indices}')
	print('Row layout: interleaved per image -> z row then x_pred row (timesteps left-to-right)')
	print(f'Total grid rows: {args.num_images * 2} (for {args.num_images} images)')
	print('=' * 60)
	print('Trajectory visualization completed!')
	print(f'Image saved to: {output_path}')
	print('=' * 60)


if __name__ == '__main__':
	main()
