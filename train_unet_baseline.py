import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm

from dataset_thermal import ThermalPairedDataset
from model_drunet import DRUNet

def main():
    # 1. Configuración de Hardware y Directorios
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Iniciando entrenamiento del Baseline (Pix2Pix) en: {device} ===")

    batch_size = 16        # RTX 3090 tiene 24GB, soporta 16 o 32 sin problema
    learning_rate = 1e-4
    epochs = 2000          # Subimos las épocas porque 1 época son muy pocos pasos
    save_dir = "checkpoints_baseline"
    sample_dir = os.path.join(save_dir, "samples")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # 2. Creación del Dataset y Train/Val Split
    print("[*] Cargando dataset y creando particiones (80% Train, 20% Val)...")
    full_dataset = ThermalPairedDataset(
        scenes_dir="Scenes", 
        cond_folder="120", 
        target_folder="005", 
        patch_size=256, 
        is_train=True
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # random_split devuelve objetos 'Subset' que comparten la referencia al dataset original
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # TRUCO DE PROFESOR: Hacemos una copia del dataset original para la validación.
    # Así podemos apagar el Data Augmentation (is_train=False) SOLO para el subset de validación
    # sin afectar al subset de entrenamiento.
    val_subset.dataset = copy.copy(full_dataset)
    val_subset.dataset.is_train = False

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. Inicializar Modelo (El Generador / U-Net) y Optimizador
    model = DRUNet(in_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    # 4. Bucle de Entrenamiento
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        # tqdm para la barra de progreso de entrenamiento
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for x_cond, x_target in pbar:
            x_cond, x_target = x_cond.to(device), x_target.to(device)
            
            # Dummy timestep para el DRUNet
            t_dummy = torch.zeros(x_cond.size(0), device=device)
            
            pred = model(x_cond, t_dummy)
            loss = F.l1_loss(pred, x_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Actualizar barra con la loss actual
            pbar.set_postfix({"L1 Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validación y Muestreo visual (Cada 5 Épocas)
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_loss = 0.0
            saved_grid = False
            
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ")
                for x_cond, x_target in pbar_val:
                    x_cond, x_target = x_cond.to(device), x_target.to(device)
                    t_dummy = torch.zeros(x_cond.size(0), device=device)
                    
                    pred = model(x_cond, t_dummy)
                    loss = F.l1_loss(pred, x_target)
                    val_loss += loss.item()
                    pbar_val.set_postfix({"L1 Loss": f"{loss.item():.4f}"})
                    
                    # Generamos la grilla visual SOLO con el primer batch de la validación
                    if not saved_grid:
                        n = min(4, x_cond.size(0)) # Tomamos hasta 4 imágenes de ejemplo
                        
                        # Des-normalizamos de [-1, 1] a [0, 1] para guardar la imagen
                        cond_vis = (x_cond[:n] * 0.5) + 0.5
                        pred_vis = (pred[:n] * 0.5) + 0.5
                        target_vis = (x_target[:n] * 0.5) + 0.5
                        
                        # Concatenamos verticalmente: Fila 1 (Inputs), Fila 2 (Predicciones), Fila 3 (Targets Reales)
                        grid = torch.cat([cond_vis, pred_vis, target_vis], dim=0)
                        
                        # Guardamos sobreescribiendo siempre el mismo archivo para no llenar el disco
                        sample_path = os.path.join(sample_dir, "latest_validation_sample.png")
                        save_image(grid, sample_path, nrow=n)
                        saved_grid = True
                        
            avg_val_loss = val_loss / len(val_loader)
            print(f"   -> Resumen Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
            
            # Guardamos el mejor modelo según validación
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(save_dir, "unet_baseline_best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"   [!] Nuevo mejor modelo guardado en {best_path}")

        # Checkpoints regulares cada 20 épocas (sobreescribiendo para ahorrar disco)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "unet_baseline_latest.pt"))

if __name__ == "__main__":
    main()
