import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import json

from dataset_thermal import ThermalPairedDataset
from denoiser import Denoiser

def main():
    parser = argparse.ArgumentParser(description='Palette Conditional Diffusion Training')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_name', type=str, default='DRUnet')
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--loss', type=str, default='sup')
    parser.add_argument('--pred', type=str, default='v') # Cambiado a 'v' (Rectified Flow / Flow-Matching)
    parser.add_argument('--w', type=str, default='none') # 'none' garantiza que NO haya divisiones que exploten
    parser.add_argument('--sampling_method', type=str, default='heun')
    parser.add_argument('--num_sampling_steps', type=int, default=50) # ODE solver steps
    parser.add_argument('--noise_scale', type=float, default=1.0)
    parser.add_argument('--P_mean', type=float, default=-1.2) 
    parser.add_argument('--P_std', type=float, default=1.2)
    parser.add_argument('--t_eps', type=float, default=1e-3)
    parser.add_argument('--cfg', type=float, default=1.0) # No classifier free guidance scale required
    parser.add_argument('--interval_min', type=float, default=0.0)
    parser.add_argument('--interval_max', type=float, default=1.0)
    parser.add_argument('--label_drop_prob', type=float, default=0.0) # Not needed for pure Image2Image
    parser.add_argument('--ema_decay1', type=float, default=0.999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Iniciando entrenamiento Fase 3 (Palette Diffusion) en: {device} ===")

    save_dir = "checkpoints_palette"
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 1. Dataset
    print("[*] Preparando dataset térmico...")
    full_dataset = ThermalPairedDataset(scenes_dir="Scenes", cond_folder="120", target_folder="005", patch_size=args.img_size, is_train=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    val_subset.dataset = copy.copy(full_dataset)
    val_subset.dataset.is_train = False

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Inicializar Modelo de Difusión
    # in_channels=6 porque concatenamos [120s (3 canales) + Ruido (3 canales)]
    args.in_channels = 6
    model = Denoiser(args, model_name=args.model_name).to(device)
    
    # EMA params setup
    model.ema_params1 = copy.deepcopy(list(model.parameters()))
    model.ema_params2 = copy.deepcopy(list(model.parameters()))

    # 3. Optimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Bucle de Entrenamiento
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for x_cond, x_target in pbar:
            x_cond, x_target = x_cond.to(device), x_target.to(device)
            # Las imágenes ya vienen en [-1, 1] desde dataset_thermal.py
            
            # Dummy labels
            labels = torch.zeros(x_cond.size(0), dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Pasamos x_target (la imagen limpia que queremos recuperar) y x_cond (la imagen guía de 120s)
                loss = model(x_target, labels, cond=x_cond)
            
            loss.backward()
            optimizer.step()
            model.update_ema()
            
            train_loss += loss.item()
            pbar.set_postfix({"L2_V": f"{loss.item():.4f}"})
            
        avg_loss = train_loss / len(train_loader)
        
        # Validación y Muestreo
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Tomamos un solo batch de validación para generar el grid
                x_cond, x_target = next(iter(val_loader))
                n = min(4, x_cond.size(0))
                x_cond = x_cond[:n].to(device)
                x_target = x_target[:n].to(device)
                labels = torch.zeros(n, dtype=torch.long, device=device)
                
                # --- Generación con PESOS ACTIVOS ---
                pred_active = model.generate(labels, cond=x_cond, rgb=True)

                # --- Generación con PESOS EMA ---
                active_params = [p.data.clone() for p in model.parameters()]
                for targ, ema_p in zip(model.parameters(), model.ema_params1):
                    targ.data.copy_(ema_p.data)
                pred_ema = model.generate(labels, cond=x_cond, rgb=True)
                for targ, act_p in zip(model.parameters(), active_params):
                    targ.data.copy_(act_p.data)
                del active_params
                
                # Desnormalizar de [-1, 1] a [0, 1]
                cond_vis = (x_cond * 0.5 + 0.5).clamp(0, 1)
                active_vis = (pred_active * 0.5 + 0.5).clamp(0, 1)
                ema_vis = (pred_ema * 0.5 + 0.5).clamp(0, 1)
                target_vis = (x_target * 0.5 + 0.5).clamp(0, 1)
                
                # Grid: Fila1=Cond | Fila2=Active | Fila3=EMA | Fila4=GT
                grid = torch.cat([cond_vis, active_vis, ema_vis, target_vis], dim=0)
                sample_path = os.path.join(sample_dir, "latest_palette_sample.png")
                save_image(grid, sample_path, nrow=n)
                
            print(f"   -> Resumen Epoch {epoch}: Train Loss (L2 V) = {avg_loss:.4f}")

        # Checkpoints
        if epoch % 20 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'ema_params1': [p.data.clone() for p in model.ema_params1],
                'ema_params2': [p.data.clone() for p in model.ema_params2],
            }, os.path.join(save_dir, "palette_latest.pt"))

if __name__ == "__main__":
    main()
