import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF
import argparse
import json
from tqdm import tqdm

from denoiser import Denoiser

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Palette One-Shot Full Image Evaluation')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_palette/palette_latest.pt')
    parser.add_argument('--config', type=str, default='checkpoints_palette/config.json')
    parser.add_argument('--scenes_dir', type=str, default='Scenes')
    parser.add_argument('--cond_folder', type=str, default='120')
    parser.add_argument('--target_folder', type=str, default='005')
    parser.add_argument('--out_dir', type=str, default='palette_oneshot_results')
    args_eval = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluando Modelo Palette (One-Shot) en {device} ===")
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
        for targ, ema_p in zip(model.parameters(), checkpoint['ema_params1']):
            targ.data.copy_(ema_p)
    else:
        print("[!] No se encontraron pesos EMA. Cargando pesos activos...")
        model.load_state_dict(checkpoint['model'])

    # 4. Encontrar Escenas de Validación
    all_scenes = sorted(os.listdir(args_eval.scenes_dir))
    val_scenes = all_scenes[-5:] # Tomar las últimas 5 escenas
    print(f"[*] Evaluando {len(val_scenes)} escenas: {val_scenes}")

    base_size = (800, 1080) # (Ancho, Alto). Múltiplo de 8 para DRUNet.

    for scene_name in val_scenes:
        scene_path = os.path.join(args_eval.scenes_dir, scene_name)
        if not os.path.isdir(scene_path): continue

        cond_path = os.path.join(scene_path, args_eval.cond_folder, "TH.png")
        target_path = os.path.join(scene_path, args_eval.target_folder, "TH.png")
        if not os.path.exists(cond_path): cond_path = cond_path.replace(".png", ".jpg")
        if not os.path.exists(target_path): target_path = target_path.replace(".png", ".jpg")
        if not (os.path.exists(cond_path) and os.path.exists(target_path)): continue

        print(f"  -> Procesando {scene_name} en One-Shot ...")
        img_cond = Image.open(cond_path).convert("RGB")
        img_target = Image.open(target_path).convert("RGB")
        
        # Redimensionar si no es 800x1080
        if img_cond.size != base_size: img_cond = img_cond.resize(base_size, Image.Resampling.BILINEAR)
        if img_target.size != base_size: img_target = img_target.resize(base_size, Image.Resampling.BILINEAR)

        # Convertir a tensor [1, 3, H, W] y normalizar a [-1, 1]
        t_cond = ((TF.to_tensor(img_cond) - 0.5) / 0.5).unsqueeze(0).to(device)
        t_target = ((TF.to_tensor(img_target) - 0.5) / 0.5).unsqueeze(0).to(device)
        
        # Etiqueta dummy
        labels = torch.zeros(1, dtype=torch.long, device=device)
        
        # Generar toda la imagen de una vez
        # La función model.generate() ahora detecta automáticamente la forma de t_cond
        print("     [Iterando Solver ODE...]")
        pred = model.generate(labels, cond=t_cond, rgb=True)
        
        # Desnormalizar a [0, 1]
        vis_cond = (t_cond * 0.5 + 0.5).clamp(0, 1)
        vis_pred = (pred * 0.5 + 0.5).clamp(0, 1)
        vis_target = (t_target * 0.5 + 0.5).clamp(0, 1)
        
        # Armar Grid Completo: [Condición | Generación | Target]
        grid = torch.cat([vis_cond, vis_pred, vis_target], dim=3) # Concatenar a lo ancho
        
        save_path = os.path.join(args_eval.out_dir, f"{scene_name}_oneshot.png")
        save_image(grid, save_path)
        print(f"  -> Guardado en {save_path}")

    print("=== Evaluación One-Shot Completada ===")

if __name__ == "__main__":
    main()
