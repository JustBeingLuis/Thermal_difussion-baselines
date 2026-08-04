import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF
import argparse
import json
import math
from tqdm import tqdm

from denoiser import Denoiser

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Palette Full Image Evaluation')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_palette/palette_latest.pt')
    parser.add_argument('--config', type=str, default='checkpoints_palette/config.json')
    parser.add_argument('--scenes_dir', type=str, default='Scenes')
    parser.add_argument('--cond_folder', type=str, default='120')
    parser.add_argument('--target_folder', type=str, default='005')
    parser.add_argument('--out_dir', type=str, default='palette_full_results')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128) # Solapamiento del 50%
    parser.add_argument('--batch_size', type=int, default=16) # Parches por batch para no saturar VRAM
    args_eval = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluando Modelo Palette en Imágenes Completas en {device} ===")
    os.makedirs(args_eval.out_dir, exist_ok=True)

    # 1. Cargar configuración original
    with open(args_eval.config, 'r') as f:
        config_dict = json.load(f)
        
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
            
    args = AttrDict(config_dict)
    
    # 2. Inicializar Modelo
    args.in_channels = 6
    model = Denoiser(args, model_name=args.model_name).to(device)
    model.eval()

    # 3. Cargar Pesos (Prioridad al EMA)
    checkpoint = torch.load(args_eval.checkpoint, map_location=device)
    if 'ema_params1' in checkpoint:
        print("[*] Cargando pesos EMA 1 (Alta Calidad) ...")
        # Load EMA params into the model's active parameters
        for targ, ema_p in zip(model.parameters(), checkpoint['ema_params1']):
            targ.data.copy_(ema_p)
    else:
        print("[!] No se encontraron pesos EMA. Cargando pesos activos del optimizador...")
        model.load_state_dict(checkpoint['model'])

    # 4. Encontrar Escenas de Validación
    # Seleccionaremos algunas escenas para evaluar (por ejemplo, las últimas 5)
    all_scenes = sorted(os.listdir(args_eval.scenes_dir))
    val_scenes = all_scenes[-5:] # Tomar las últimas 5 escenas como validación
    print(f"[*] Evaluando {len(val_scenes)} escenas: {val_scenes}")

    base_size = (800, 1080) # (Ancho, Alto)

    for scene_name in val_scenes:
        scene_path = os.path.join(args_eval.scenes_dir, scene_name)
        if not os.path.isdir(scene_path):
            continue

        cond_path = os.path.join(scene_path, args_eval.cond_folder, "TH.png")
        target_path = os.path.join(scene_path, args_eval.target_folder, "TH.png")
        if not os.path.exists(cond_path): cond_path = cond_path.replace(".png", ".jpg")
        if not os.path.exists(target_path): target_path = target_path.replace(".png", ".jpg")
        
        if not (os.path.exists(cond_path) and os.path.exists(target_path)):
            continue

        print(f"  -> Procesando {scene_name} ...")
        img_cond = Image.open(cond_path).convert("RGB")
        img_target = Image.open(target_path).convert("RGB")
        
        if img_cond.size != base_size: img_cond = img_cond.resize(base_size, Image.Resampling.BILINEAR)
        if img_target.size != base_size: img_target = img_target.resize(base_size, Image.Resampling.BILINEAR)

        # Convertir a tensor y normalizar a [-1, 1]
        t_cond = (TF.to_tensor(img_cond) - 0.5) / 0.5 # [3, H, W]
        t_target = (TF.to_tensor(img_target) - 0.5) / 0.5 # [3, H, W]
        
        # Buffers para reconstrucción de ventana deslizante
        C, H, W = t_cond.shape
        output_img = torch.zeros((1, 3, H, W), device=device)
        weight_map = torch.zeros((1, 1, H, W), device=device)
        
        # Crear ventana de Hann 2D para blending suave (Difuminado en los bordes)
        window_1d = torch.hann_window(args_eval.patch_size, device=device)
        window_2d = (window_1d.unsqueeze(1) * window_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        
        # Extraer parches
        patches = []
        coords = []
        
        patch_size = args_eval.patch_size
        stride = args_eval.stride
        
        for y in range(0, H - patch_size + stride, stride):
            for x in range(0, W - patch_size + stride, stride):
                # Ajustar bordes si el parche se sale
                y_start = min(y, H - patch_size)
                x_start = min(x, W - patch_size)
                y_end = y_start + patch_size
                x_end = x_start + patch_size
                
                patch = t_cond[:, y_start:y_end, x_start:x_end]
                patches.append(patch)
                coords.append((y_start, y_end, x_start, x_end))
                
        patches = torch.stack(patches) # [N, 3, 256, 256]
        
        # Procesar parches en batches para no saturar memoria
        pred_patches = []
        for i in tqdm(range(0, len(patches), args_eval.batch_size), desc="Inferiendo parches", leave=False):
            batch_cond = patches[i:i+args_eval.batch_size].to(device)
            labels = torch.zeros(batch_cond.size(0), dtype=torch.long, device=device)
            
            # Generar parche
            batch_pred = model.generate(labels, cond=batch_cond, rgb=True)
            pred_patches.append(batch_pred.cpu())
            
        pred_patches = torch.cat(pred_patches, dim=0) # [N, 3, 256, 256]
        
        # Reconstruir imagen completa (Blending con ventana de Hann)
        for i, (y_start, y_end, x_start, x_end) in enumerate(coords):
            output_img[:, :, y_start:y_end, x_start:x_end] += pred_patches[i].to(device) * window_2d
            weight_map[:, :, y_start:y_end, x_start:x_end] += window_2d
            
        # Evitar división por cero en esquinas remotas (muy poco probable pero seguro)
        weight_map = torch.clamp(weight_map, min=1e-5)
        # Promediar
        final_pred = output_img / weight_map # [-1, 1]
        
        # Desnormalizar a [0, 1]
        vis_cond = (t_cond.unsqueeze(0).to(device) * 0.5 + 0.5).clamp(0, 1)
        vis_pred = (final_pred * 0.5 + 0.5).clamp(0, 1)
        vis_target = (t_target.unsqueeze(0).to(device) * 0.5 + 0.5).clamp(0, 1)
        
        # Armar Grid Completo: [Condición | Generación | Target]
        grid = torch.cat([vis_cond, vis_pred, vis_target], dim=3) # Concatenar a lo ancho
        
        save_path = os.path.join(args_eval.out_dir, f"{scene_name}_full.png")
        save_image(grid, save_path)
        print(f"  -> Guardado en {save_path}")

    print("=== Evaluación Completada ===")

if __name__ == "__main__":
    main()
