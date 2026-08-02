import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm

from dataset_thermal import ThermalPairedDataset
from model_drunet import DRUNet
from model_discriminator import PatchGAN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Iniciando entrenamiento de Fase 2 (Pix2Pix GAN) en: {device} ===")

    # Hiperparámetros de GAN Clásico
    batch_size = 16
    learning_rate = 2e-4  # En GANs es común usar 2e-4 en vez de 1e-4
    epochs = 2000
    lambda_l1 = 100.0     # Peso fuertísimo a la L1 Loss para que no invente formas locas
    
    save_dir = "checkpoints_pix2pix"
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Dataset
    print("[*] Preparando dataset...")
    full_dataset = ThermalPairedDataset(scenes_dir="Scenes", cond_folder="120", target_folder="005", patch_size=256, is_train=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    val_subset.dataset = copy.copy(full_dataset)
    val_subset.dataset.is_train = False

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. Inicializar Modelos (Generador U-Net y Discriminador PatchGAN)
    # Reutilizamos el DRUNet pasando t=0, que ya probamos que funciona como generador puro
    generator = DRUNet(in_channels=3).to(device)
    # El discriminador recibe 6 canales (3 de la condición + 3 de la generada/real)
    discriminator = PatchGAN(in_channels=6).to(device)

    # 3. Optimizadores (Los GANs son muy sensibles, Adam con beta1=0.5 es un estándar de facto)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Funciones de Pérdida
    criterion_GAN = nn.BCEWithLogitsLoss() # Para castigar si es Real (1) o Falso (0)
    criterion_L1 = nn.L1Loss()             # Para la fidelidad de la imagen base

    best_val_loss = float('inf')

    # 4. Bucle de Entrenamiento Adversarial
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        train_loss_G = 0.0
        train_loss_D = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for x_cond, x_target in pbar:
            x_cond, x_target = x_cond.to(device), x_target.to(device)
            t_dummy = torch.zeros(x_cond.size(0), device=device)
            
            # Generar imagen falsa
            fake_target = generator(x_cond, t_dummy)
            
            # ==========================================
            #  Entrenar Discriminador (El Policía)
            # ==========================================
            optimizer_D.zero_grad()
            
            # Loss Real: Trata de clasificar (x_cond + x_target_real) como verdaderos (1)
            pred_real = discriminator(x_cond, x_target)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Loss Fake: Trata de clasificar (x_cond + fake_target) como falsos (0)
            # Usamos .detach() para no pasar el gradiente al generador por error aquí
            pred_fake = discriminator(x_cond, fake_target.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # ==========================================
            #  Entrenar Generador (El Falsificador)
            # ==========================================
            optimizer_G.zero_grad()
            
            # Loss GAN: El Generador quiere que el Discriminador clasifique la imagen falsa como verdadera (1)
            pred_fake_G = discriminator(x_cond, fake_target)
            loss_G_GAN = criterion_GAN(pred_fake_G, torch.ones_like(pred_fake_G))
            
            # Loss L1: A la vez, el generador no puede pintar cosas al azar, debe respetar el calor original
            loss_G_L1 = criterion_L1(fake_target, x_target)
            
            # Loss Total del Generador
            loss_G = loss_G_GAN + (lambda_l1 * loss_G_L1)
            loss_G.backward()
            optimizer_G.step()
            
            # Actualizar barra de progreso
            train_loss_G += loss_G.item()
            train_loss_D += loss_D.item()
            pbar.set_postfix({"L_G": f"{loss_G.item():.3f}", "L_D": f"{loss_D.item():.3f}"})
            
        avg_loss_G = train_loss_G / len(train_loader)
        
        # ==========================================
        #  Validación y Muestreo (Cada 5 Épocas)
        # ==========================================
        if epoch % 5 == 0 or epoch == 1:
            generator.eval()
            val_loss = 0.0
            saved_grid = False
            
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ")
                for x_cond, x_target in pbar_val:
                    x_cond, x_target = x_cond.to(device), x_target.to(device)
                    t_dummy = torch.zeros(x_cond.size(0), device=device)
                    
                    pred = generator(x_cond, t_dummy)
                    # En validación medimos puramente L1 para saber qué tan cerca estamos matemáticamente
                    loss = criterion_L1(pred, x_target) 
                    val_loss += loss.item()
                    
                    if not saved_grid:
                        n = min(4, x_cond.size(0))
                        cond_vis = (x_cond[:n] * 0.5) + 0.5
                        pred_vis = (pred[:n] * 0.5) + 0.5
                        target_vis = (x_target[:n] * 0.5) + 0.5
                        
                        grid = torch.cat([cond_vis, pred_vis, target_vis], dim=0)
                        sample_path = os.path.join(sample_dir, "latest_pix2pix_sample.png")
                        save_image(grid, sample_path, nrow=n)
                        saved_grid = True
                        
            avg_val_loss = val_loss / len(val_loader)
            print(f"   -> Resumen Epoch {epoch}: G_Loss = {avg_loss_G:.4f} | Val L1 Loss = {avg_val_loss:.4f}")
            
            # Guardamos el generador si la Loss L1 general mejora
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(save_dir, "pix2pix_generator_best.pt")
                torch.save(generator.state_dict(), best_path)
                print(f"   [!] Nuevo mejor Generador guardado en {best_path}")

        # Checkpoints (sobreescribiendo para ahorrar espacio)
        if epoch % 20 == 0:
            torch.save(generator.state_dict(), os.path.join(save_dir, "pix2pix_generator_latest.pt"))
            torch.save(discriminator.state_dict(), os.path.join(save_dir, "pix2pix_discriminator_latest.pt"))

if __name__ == "__main__":
    main()
