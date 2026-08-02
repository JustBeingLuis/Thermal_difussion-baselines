import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

EPS_LOWER = 0.05
NOISE_LEVEL = 0.0

# -------------------
# data: 2D spiral
# -------------------
def make_spiral(
    n: int = 20000,
    n_turns: float = 2.0,
    r_start: float = 0.1,
    r_end: float = 1.0,
    noise_scale: float = 0.0,
) -> np.ndarray:
    """
    Generates a standardized 2D spiral dataset.

    Parameters:
        n: Number of data points to generate (must be positive, default: 20000)
        n_turns: Total number of spiral turns (1 turn = 2π radians, default: 2.0)
        r_start: Starting radius of the spiral (default: 0.1)
        r_end: Ending radius of the spiral (default: 1.0)
        noise_scale: Standard deviation of Gaussian noise added to coordinates (default: 0.0)

    Returns:
        Numpy array of shape `(n, 2)` where rows are `(x, y)` coordinates (standardized).
    """
    assert isinstance(n, int) and n > 0, f"Number of data points 'n' must be a positive integer. Current input: {n}"
    assert r_start > 0 and r_end > 0, f"Radii 'r_start' and 'r_end' must be positive. Current inputs: {r_start}, {r_end}"
    assert n_turns > 0, f"Number of spiral turns 'n_turns' must be positive. Current input: {n_turns}"
    assert noise_scale >= 0, f"Noise standard deviation 'noise_scale' cannot be negative. Current input: {noise_scale}"

    t = np.linspace(0, 2 * np.pi * n_turns, n)  # Angles: 0 to (turns × 2π)
    r = np.linspace(r_start, r_end, n)          # Radii: linear increase from start to end
    x = r * np.cos(t)
    y = r * np.sin(t)

    # Standardize to unit scale (global std), preserving overall variance
    data = np.stack([x, y], axis=1, dtype=np.float32)
    data_std = data.std()
    if data_std > 0:  # Avoid division by zero in edge cases
        data /= data_std

    return data


class DenoiseMLP(nn.Module):
    """
    5-Layers MLP denoiser for DDPM.

    Parameters:
        dim: Feature dimension of input samples (D).
        config: Configuration dictionary containing the noise schedule.
            Required key: `"T"` (total number of timesteps).
        hidden: Number of hidden units in each layer (default: 256).
    """

    def __init__(self, dim, config, hidden=256):
        super().__init__()
        n = config['hidden_layers']
        assert n >= 2, f"Hidden layers 'n' must be at least 2. Current input: {n}"
        self.net = nn.ModuleList([
            nn.Linear(dim + 1, hidden),
            nn.ReLU(),
        ])
        for _ in range(n - 2):
            self.net.extend([
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ])
        self.net.append(nn.Linear(hidden, dim))

    def forward(self, x, t):
        """
        Predicts the clean sample x0 at timestep `t`.

        - Takes time `t` already in [0, 1]
        - Concatenates time with features and feeds the MLP

        Parameters:
            x: Input features of shape `(B, D)`
            t: Float timesteps of shape `(B,)` with values in `[0, 1]`

        Returns:
            Tensor of shape `(B, D)` predicting the clean sample x0
        """
        t = t.float().unsqueeze(1)
        h = torch.cat([x, t], dim=1)
        for layer in self.net:
            h = layer(h)
        return h


# -------------------
# training for x-pred parameterization
# -------------------
def train_one(
        D, data2d, config,
        num_steps=4000, batch_size=512, lr=1e-3,
        val_split=0.2,
        patience=5,
        min_delta=1e-5,
        label="",
        loss="sup"
):
    """
    Train a denoiser model using x-pred parameterization.
    
    Parameters:
        D: Feature dimension (2 for this case)
        data2d: Training data of shape (N, 2)
        config: Diffusion config with schedule parameters
        num_steps: Number of training steps
        batch_size: Batch size for training
        lr: Learning rate
        val_split: Fraction of data for validation
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        label: Label for the training run (e.g., "ground-truth" or "noisy")
    
    Returns:
        model: Trained DenoiseMLP
        P: Projection matrix (identity for D=2)
    """
    device = config['device']
    data_tensor = torch.from_numpy(data2d).to(device)
    N = data_tensor.shape[0]
    val_size = int(N * val_split)

    perm = torch.randperm(N, device=device)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    # For D=2, P is identity, but we keep it for consistency
    rand = torch.randn(D, 2)
    # P, _ = torch.linalg.qr(rand)
    P = torch.eye(D, 2, device=device)  # Identity projection for D=2

    P = P.to(device)
    train_x0_D = data_tensor[train_idx] @ P.T
    val_x0_D = data_tensor[val_idx] @ P.T

    model = DenoiseMLP(D, config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.8)

    best_val_loss = float('inf')
    patience_counter = 0

    train_idx_all = torch.arange(train_x0_D.shape[0], device=device)
    with tqdm(total=num_steps, desc=f"Train D={D}, {label}", leave=False) as pbar:
        for step in range(1, num_steps + 1):
            idx = train_idx_all[torch.randint(0, train_x0_D.shape[0], (batch_size,), device=device)]
            x0 = train_x0_D[idx]

            # Sample t uniformly in [0, 1]
            t = torch.rand(batch_size, device=device)

            if loss == "sup":
                y1 = x0
                y2 = y1

            elif loss == "noisy":
                y1 = x0 #  + torch.randn_like(x0) * noise_level  # Add some noise to the ground truth
                y2 = y1
            
            elif loss == "n2n":
                y1 = x0 + torch.randn_like(x0) * NOISE_LEVEL  # Noisy version of x0
                y2 = x0 + torch.randn_like(x0) * NOISE_LEVEL  # Another independent noisy version of x0
            
            elif loss == "r2r":
                alpha = 0.5

                y = x0
                eps = torch.randn_like(x0)
                y1 = y + eps * NOISE_LEVEL * alpha
                y2 = y - eps * NOISE_LEVEL / alpha

            
            # Forward diffusion: xt = t * x0 + (1 - t) * noise
            noise = torch.randn_like(x0)

            xt_y1 = t.view(-1, 1) * y1 + (1 - t.view(-1, 1)) * noise
            xt_y2 = t.view(-1, 1) * y2 + (1 - t.view(-1, 1)) * noise
            


            v_true =  (y2 - xt_y2) / ( 1 - t.view(-1, 1) ).clamp(min=EPS_LOWER)
            # Model predicts x0
            x0_pred = model(xt_y1, t)
            v_pred = (x0_pred - xt_y2) / ( 1 - t.view(-1, 1) ).clamp(min=EPS_LOWER)
            
            # Loss: MSE between predicted and true x0
            # loss = F.mse_loss(v_pred, v_true)
            loss = F.mse_loss(x0_pred, y2)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 500 == 0 or step == num_steps:
                val_loss = compute_val_loss(model, val_x0_D, config, batch_size=batch_size)
                sch.step(val_loss)
                pbar.set_postfix({"train_loss": f"{loss.item():.4f}", "val_loss": f"{val_loss:.4f}"})

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        pbar.write(f"Early stopping triggered at step {step}! Best val loss: {best_val_loss:.4f}")
                        break
            else:
                pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

            pbar.update(1)

    return model, P



# -------------------
# training for x-pred only as image denoising
# -------------------
def train_denoiser(
        D, data2d, config,
        num_steps=4000, batch_size=512, lr=1e-3,
        val_split=0.2,
        patience=5,
        min_delta=1e-5,
        label="",
        loss="sup",
        save_every=1000,
        out_dir="plots"
):
    """
    Train a denoiser model using x-pred parameterization.

    Adds periodic validation and plotting for 'n2n' and 'r2r' when `save_every` > 0.
    """
    device = config['device']
    data_tensor = torch.from_numpy(data2d).to(device)
    N = data_tensor.shape[0]
    val_size = int(N * val_split)

    perm = torch.randperm(N, device=device)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    # For D=2, P is identity, but we keep it for consistency
    rand = torch.randn(D, 2)
    # P, _ = torch.linalg.qr(rand)
    P = torch.eye(D, 2, device=device)  # Identity projection for D=2

    P = P.to(device)
    train_x0_D = data_tensor[train_idx] @ P.T
    val_x0_D = data_tensor[val_idx] @ P.T

    model = DenoiseMLP(D, config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=100000, factor=0.8)

    best_val_loss = float('inf')
    patience_counter = 0

    train_idx_all = torch.arange(train_x0_D.shape[0], device=device)
    with tqdm(total=num_steps, desc=f"Train D={D}, {label}", leave=False) as pbar:
        for step in range(1, num_steps + 1):
            idx = train_idx_all[torch.randint(0, train_x0_D.shape[0], (batch_size,), device=device)]
            x0 = train_x0_D[idx]

            # Construct t as zeros (denoising)
            t = torch.zeros(x0.shape[0], device=device)

            if loss == "sup":
                y1 = x0 + torch.randn_like(x0) * NOISE_LEVEL  # Add some noise to the ground truth
                y2 = x0

            elif loss == "noisy":
                y1 = x0 + torch.randn_like(x0) * NOISE_LEVEL  # Add some noise to the ground truth
                y2 = y1

            elif loss == "n2n":
                y1 = x0 + torch.randn_like(x0) * NOISE_LEVEL  # Noisy version of x0
                y2 = x0 + torch.randn_like(x0) * NOISE_LEVEL  # Another independent noisy version of x0

            elif loss == "r2r":
                alpha = 0.5
                y = x0
                eps = torch.randn_like(x0)
                y1 = y + eps * NOISE_LEVEL * alpha
                y2 = y - eps * NOISE_LEVEL / alpha

            x0_pred = model(y1, t)

            # Loss: MSE between predicted and target (y2 or x0 depending on setup)
            batch_loss = F.mse_loss(x0_pred, y2)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            # Periodic validation and plotting
            if (save_every and (step % save_every == 0)) or step == num_steps:
                val_loss, denoised_np, noisy_np = compute_val_denoiser_loss(model, val_x0_D, config, loss, batch_size=batch_size)
                # sch.step(val_loss)

                # Compute plot limits from all points
                val_np = val_x0_D.cpu().numpy()
                all_pts = [val_np]
                if noisy_np is not None and noisy_np.size:
                    all_pts.append(noisy_np)
                if denoised_np is not None and denoised_np.size:
                    all_pts.append(denoised_np)
                all_stack = np.vstack(all_pts)
                min_x, max_x = all_stack[:, 0].min(), all_stack[:, 0].max()
                min_y, max_y = all_stack[:, 1].min(), all_stack[:, 1].max()
                cx = 0.5 * (min_x + max_x)
                cy = 0.5 * (min_y + max_y)
                half = 0.5 * max(max_x - min_x, max_y - min_y)
                pad = 0.05 * max(max_x - min_x, max_y - min_y)
                half = max(half, 1e-3) + pad
                xlim = (cx - half, cx + half)
                ylim = (cy - half, cy + half)

                # Save plot only for n2n and r2r denoisers (user request)
                if loss in ("n2n", "r2r"):
                    save_denoiser_plot(step, loss, val_np, noisy_np, denoised_np, xlim, ylim, out_dir=out_dir)

                pbar.set_postfix({"train_loss": f"{batch_loss.item():.4f}", "val_loss": f"{val_loss:.4f}"})
            else:
                pbar.set_postfix({"train_loss": f"{batch_loss.item():.4f}"})

            pbar.update(1)

    return model, P

def compute_val_loss(model, val_x0_D, config, batch_size=512):
    """
    Computes the validation loss for x-pred parameterization.

    Parameters:
        model: Trained `DenoiseMLP`
        val_x0_D: Validation set data (N_val, D)
        config: Diffusion configuration
        batch_size: Batch size for validation

    Returns:
        val_loss: Average MSE loss over validation set
    """
    model.eval()
    device = config['device']
    N_val = val_x0_D.shape[0]
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, N_val, batch_size):
            batch_x0 = val_x0_D[i:i + batch_size]
            
            # Sample t uniformly in [0, 1]
            batch_t = torch.rand(batch_x0.shape[0], device=device)
            
            # Forward diffusion: xt = t * x0 + (1 - t) * noise
            noise = torch.randn_like(batch_x0)
            xt = batch_t.view(-1, 1) * batch_x0 + (1 - batch_t.view(-1, 1)) * noise
            

            v_true = batch_x0 - noise
            # Model predicts x0
            x0_pred = model(xt, batch_t)
            v_pred = (x0_pred - xt) / ( 1 - batch_t.view(-1, 1) ).clamp(min=EPS_LOWER)
            
            # Loss: MSE between predicted and true x0
            # batch_loss = F.mse_loss(v_pred, v_true)
            batch_loss = F.mse_loss(x0_pred, batch_x0)
            total_loss += batch_loss.item() * batch_x0.shape[0]

    model.train()
    return total_loss / N_val


# -------------------
# Denoiser evaluation + plotting helpers
# -------------------

def compute_val_denoiser_loss(model, val_x0_D, config, loss_type, batch_size=512, n_avg=10, alpha=0.5):
    """
    Compute validation loss and return denoised + noisy arrays for plotting.

    - For 'n2n': single noisy input per clean target.
    - For 'r2r': average `n_avg` denoised outputs per clean target to form the estimate.

    Returns: (val_loss, denoised_np, noisy_np)
    """
    model.eval()
    device = config['device']
    N = val_x0_D.shape[0]
    total_loss = 0.0
    denoised_list = []
    noisy_list = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = val_x0_D[i:i + batch_size].to(device)
            if loss_type == 'n2n':
                noisy = batch + torch.randn_like(batch) * NOISE_LEVEL
                out = model(noisy, torch.zeros(noisy.shape[0], device=device))
                batch_loss = F.mse_loss(out, batch)
                denoised_list.append(out.cpu().numpy())
                noisy_list.append(noisy.cpu().numpy())
                total_loss += batch_loss.item() * batch.shape[0]

            elif loss_type == 'r2r':
                # For r2r we average multiple denoiser outputs per input
                denoised_acc = torch.zeros_like(batch)
                # keep a representative single noisy sample for plotting
                noisy_rep = batch + torch.randn_like(batch) * NOISE_LEVEL * 0.5
                for _ in range(n_avg):
                    eps = torch.randn_like(batch)
                    y1 = batch + eps * NOISE_LEVEL * alpha
                    denoised_acc += model(y1, torch.zeros(y1.shape[0], device=device))
                denoised_avg = denoised_acc / n_avg
                batch_loss = F.mse_loss(denoised_avg, batch)
                denoised_list.append(denoised_avg.cpu().numpy())
                noisy_list.append(noisy_rep.cpu().numpy())
                total_loss += batch_loss.item() * batch.shape[0]

            else:
                # default supervised/noisy
                noisy = batch + torch.randn_like(batch) * NOISE_LEVEL
                out = model(noisy, torch.zeros(noisy.shape[0], device=device))
                batch_loss = F.mse_loss(out, batch)
                denoised_list.append(out.cpu().numpy())
                noisy_list.append(noisy.cpu().numpy())
                total_loss += batch_loss.item() * batch.shape[0]

    model.train()
    denoised = np.vstack(denoised_list) if len(denoised_list) > 0 else np.zeros((0, val_x0_D.shape[1]))
    noisy_all = np.vstack(noisy_list) if len(noisy_list) > 0 else np.zeros_like(denoised)
    return total_loss / N, denoised, noisy_all


def save_denoiser_plot(step, loss_type, val_x0_np, noisy_np, denoised_np, xlim, ylim, out_dir="plots"):
    """Save a 1x3 panel: noisy input, denoised output, ground truth."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    scatter_panel(axes[0], noisy_np, f"Noisy Input [{loss_type}]", "noisy", xlim=xlim, ylim=ylim, point_size=1.5)
    scatter_panel(axes[1], denoised_np, f"Denoised [{loss_type}]\nstep={step}", "blue", xlim=xlim, ylim=ylim, point_size=1.5)
    scatter_panel(axes[2], val_x0_np, "Ground Truth", "ground-truth", xlim=xlim, ylim=ylim, point_size=1.5)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"denoiser_{loss_type}_{step}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


# -------------------
# convert model output -> (x0_hat, eps_hat)
# -------------------
def model_to_x0(model, x_t, t):
    """
    Converts model output to reconstructed clean sample.

    Parameters:
        model: Denoiser mapping `(x_t, t)` → x0 prediction
        x_t: Noisy samples of shape `(B, D)`
        t: Timesteps of shape `(B,)` with values in [0, 1]

    Returns:
        x0_hat of shape `(B, D)`
    """
    x0_hat = model(x_t, t)
    return x0_hat


# -------------------
# DDIM-style sampling
# -------------------
@torch.no_grad()
def sample(model, D, P, config, n_samples=4000, n_steps=50):
    """
    Generates samples via iterative denoising with linear diffusion and projects to 2D.

    Parameters:
        model: Trained `DenoiseMLP`
        D: Feature dimension used during training
        P: Projection matrix of shape `D×2`
        config: Diffusion configuration and device
        n_samples: Number of samples to generate
        n_steps: Number of denoising steps

    Returns:
        `np.ndarray` of shape `(n_samples, 2)` containing 2D coordinates
    """
    device = config['device']
    
    # Start from pure noise
    x_t = torch.randn(n_samples, D, device=device)
    
    # Iterative denoising: t goes from 0 to 1
    t_steps = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    
    for i in range(len(t_steps) - 1):
        t_curr = t_steps[i]
        t_next = t_steps[i + 1]
        
        # Get x0 prediction from model
        t_batch = torch.full((n_samples,), t_curr, device=device)
        x0_pred = model_to_x0(model, x_t, t_batch)

        v_pred = (x0_pred - x_t) / ( 1 - t_curr ).clamp(min=EPS_LOWER)
        
        # This interpolates between x0_pred and noise
        x_t = x_t + (t_next - t_curr) * v_pred  # Move towards x0_pred as t decreases
    
    # Project back to 2D
    x0_2d = x_t @ P  # (n_samples, 2)
    return x0_2d.cpu().numpy()


def scatter_panel(ax, points, title, color_type="blue", xlim=None, ylim=None, point_size=1):
    """
    Helper to visualize 2D points as a scatter plot without axes ticks.
    Accepts explicit `xlim`/`ylim` and `point_size` so multiple panels can be identical.
    """
    if color_type == "ground-truth":
        color = "orange"
    elif color_type == "noisy":
        color = "red"
    else:
        color = "blue"
    
    ax.scatter(points[:, 0], points[:, 1], s=point_size, color=color, alpha=0.6)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # Create ground truth and noisy datasets
    print("Creating datasets...")
    data_ground_truth = make_spiral(n=20000, noise_scale=0.0)
    data_noisy = make_spiral(n=20000, noise_scale=0.0) 
    data_noisy += np.random.normal(loc=0.0, scale=NOISE_LEVEL, size=data_noisy.shape).astype(np.float32)

    # Configuration for simplified linear diffusion
    config = {
        'hidden_layers': 5,
        'device': device,
    }

    D = 2
    results = {}  # 'ground_truth' or 'noisy' -> (samples_2d, P)

    lr = 1e-3
    epochs = 10000

    print("\nTraining on ground truth data...")
    model_gt, P_gt = train_one(
        D, data_ground_truth, config,
        num_steps=epochs,
        lr=lr,
        val_split=0.2,
        patience=5000,
        min_delta=1e-5,
        label="ground-truth",
        loss="sup"
    )
    samples_gt = sample(model_gt, D, P_gt, config, n_samples=4000)
    results['ground_truth'] = (samples_gt, P_gt)

    print(f"\nTraining on noisy data (σ={NOISE_LEVEL})...")
    model_noisy, P_noisy = train_one(
        D, data_noisy, config,
        num_steps=epochs,
        lr=lr,
        val_split=0.2,
        patience=5000,
        min_delta=1e-5,
        label=f"noisy (σ={NOISE_LEVEL})",
        loss="noisy"
    )
    samples_noisy = sample(model_noisy, D, P_noisy, config, n_samples=4000)
    results['noisy'] = (samples_noisy, P_noisy)



    print("\nTraining Noise2Noise")
    model_n2n, P_n2n = train_one(
        D, data_ground_truth, config,
        num_steps=epochs,
        lr=lr,
        val_split=0.2,
        patience=5000,
        min_delta=1e-5,
        label=f"n2n (σ={NOISE_LEVEL})",
        loss="n2n"
    )
    samples_n2n = sample(model_n2n, D, P_n2n, config, n_samples=4000)
    results['n2n'] = (samples_n2n, P_n2n)


    print("\nTraining R2R")
    model_r2r, P_r2r = train_one(
        D, data_noisy, config,
        num_steps=epochs,
        lr=lr,
        val_split=0.2,
        patience=5000,
        min_delta=1e-5,
        label=f"r2r (σ={NOISE_LEVEL})",
        loss="r2r"
    )
    samples_r2r = sample(model_r2r, D, P_r2r, config, n_samples=4000)
    results['r2r'] = (samples_r2r, P_r2r)



    val_noisy = torch.from_numpy(data_noisy).to(device) #  @ P_noisy.T
    t_val = torch.zeros(val_noisy.shape[0], device=device)

    lr2 = 1e-3

    print("\nTraining denoising N2N")
    model_denoiser_n2n, P_denoiser_n2n = train_denoiser(
        D, data_ground_truth, config,
        num_steps=4000,
        lr=lr2,
        val_split=0.2,
        patience=5000,
        min_delta=1e-5,
        label=f"denoiser n2n (σ={NOISE_LEVEL})",    
        loss="n2n",
        save_every=1000,
        out_dir="plots"
    )

    with torch.no_grad():
        denoised_n2n = model_denoiser_n2n(val_noisy, t_val).cpu().numpy()
    results['denoiser_n2n'] = (denoised_n2n, P_denoiser_n2n)

    print("\nTraining denoising R2R")
    model_denoiser_r2r, P_denoiser_r2r = train_denoiser(
        D, data_noisy, config,
        num_steps=4000,
        lr=lr2,
        val_split=0.2,
        patience=5000,
        min_delta=1e-5,
        label=f"denoiser r2r (σ={NOISE_LEVEL})",    
        loss="r2r",
        save_every=1000,
        out_dir="plots"
    )

    with torch.no_grad():
        n_samples_r2r = 10
        denoised_r2r = 0
        for _ in range(n_samples_r2r):

            y1 = val_noisy + torch.randn_like(val_noisy) * NOISE_LEVEL * 0.5
            denoised_r2r += model_denoiser_r2r(y1, t_val).cpu().numpy()

        denoised_r2r /= n_samples_r2r

    results['denoiser_r2r'] = (denoised_r2r, P_denoiser_r2r)

    # Create comparison visualization
    fig, axes = plt.subplots(1, 8, figsize=(18, 4))
    fig.suptitle("Diffusion Models: Ground Truth vs Noisy Training Data (D=2, x-pred)", fontsize=14)

    # Compute global plot limits so all scatter panels share same size/aspect
    all_points = [data_ground_truth, data_noisy]
    for k in results:
        pts = results[k][0]
        if pts is not None:
            all_points.append(pts)
    all_stack = np.vstack(all_points)
    min_x, max_x = all_stack[:, 0].min(), all_stack[:, 0].max()
    min_y, max_y = all_stack[:, 1].min(), all_stack[:, 1].max()
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half = 0.5 * max(max_x - min_x, max_y - min_y)
    pad = 0.05 * max(max_x - min_x, max_y - min_y)
    half = max(half, 1e-3) + pad
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)
    point_size = 1.5

    # Column 1: Ground Truth Data
    scatter_panel(axes[5], data_ground_truth, "Ground Truth Data", "ground-truth", xlim=xlim, ylim=ylim, point_size=point_size)
    
    # Column 2: Samples from GT-trained model
    scatter_panel(axes[4], samples_gt, "Samples Supervised", "blue", xlim=xlim, ylim=ylim, point_size=point_size)
    
    # Column 3: Noisy Data
    scatter_panel(axes[0], data_noisy, f"Noisy Data (σ={NOISE_LEVEL})", "noisy", xlim=xlim, ylim=ylim, point_size=point_size)
    
    # Column 4: Samples from noisy-trained model
    scatter_panel(axes[1], samples_noisy, "Samples from\nNoisy-trained model", "blue", xlim=xlim, ylim=ylim, point_size=point_size)

    scatter_panel(axes[2], samples_n2n, "Samples from\nN2N-trained model", "blue", xlim=xlim, ylim=ylim, point_size=point_size)

    scatter_panel(axes[3], samples_r2r, "Samples from\nR2R-trained model", "blue", xlim=xlim, ylim=ylim, point_size=point_size)

    scatter_panel(axes[6], denoised_n2n, "Denoised N2N", "blue", xlim=xlim, ylim=ylim, point_size=point_size)
    scatter_panel(axes[7], denoised_r2r, "Denoised R2R", "blue", xlim=xlim, ylim=ylim, point_size=point_size)



    plt.tight_layout()
    plt.savefig("comparison_ground_truth_vs_noisy.png", dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: comparison_ground_truth_vs_noisy.png")
    plt.close()

    # Compute and print simple statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nGround Truth Model (trained on clean data):")
    print(f"  Generated samples mean: {samples_gt.mean(axis=0)}")
    print(f"  Generated samples std:  {samples_gt.std(axis=0)}")
    
    print(f"\nNoisy Model (trained on σ={NOISE_LEVEL} corrupted data):")
    print(f"  Generated samples mean: {samples_noisy.mean(axis=0)}")
    print(f"  Generated samples std:  {samples_noisy.std(axis=0)}")
    
    print(f"\nInput Data Statistics:")
    print(f"  Ground truth data mean: {data_ground_truth.mean(axis=0)}")
    print(f"  Ground truth data std:  {data_ground_truth.std(axis=0)}")
    print(f"  Noisy data mean:        {data_noisy.mean(axis=0)}")
    print(f"  Noisy data std:         {data_noisy.std(axis=0)}")
    print(f"\nNoise level used: σ={NOISE_LEVEL}")
    print("="*60)


if __name__ == "__main__":
    main()
