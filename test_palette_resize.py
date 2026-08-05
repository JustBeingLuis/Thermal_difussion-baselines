import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF
import argparse
import json

from denoiser import Denoiser

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Palette Resize Evaluation')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_palette/palette_latest.pt')
    parser.add_argument('--config', type=str, default='checkpoints_palette/config.json')
    parser.add_argument('--scenes_dir', type=str, default='Scenes')
    parser.add_argument('--cond_folder', type=str, default='120')
    parser.add_argument('--target_folder', type=str, default='005')
    parser.add_argument('--out_dir', type=str, default='palette_resize_results')
    args_eval = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluando Modelo Palette (Resize 256x256) en {device} ===")
    os.makedirs(args_eval.out_dir, exist_ok=True)

    with open(args_eval.config, 'r') as f:
        config_dict = json.load(f)
        
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
            
    args = AttrDict(config_dict)
    
    args.in_channels = 6
    model = Denoiser(args, model_name=args.model_name).to(device)
    model.eval()

    checkpoint = torch.load(args_eval.checkpoint, map_location=device)
    if 'ema_params1' in checkpoint:
        print("[*] Cargando pesos EMA 1 (Alta Calidad) ...")
        for targ, ema_p in zip(model.parameters(), checkpoint['ema_params1']):
            targ.data.copy_(ema_p)
    else:
        model.load_state_dict(checkpoint['model'])

    all_scenes = sorted(os.listdir(args_eval.scenes_dir))
    val_scenes = all_scenes[-5:]
    print(f"[*] Evaluando {len(val_scenes)} escenas: {val_scenes}")

    base_size = (800, 1080)
    net_size = (256, 256)

    for scene_name in val_scenes:
        scene_path = os.path.join(args_eval.scenes_dir, scene_name)
        if not os.path.isdir(scene_path): continue

        cond_path = os.path.join(scene_path, args_eval.cond_folder, "TH.png")
        target_path = os.path.join(scene_path, args_eval.target_folder, "TH.png")
        if not os.path.exists(cond_path): cond_path = cond_path.replace(".png", ".jpg")
        if not os.path.exists(target_path): target_path = target_path.replace(".png", ".jpg")
        if not (os.path.exists(cond_path) and os.path.exists(target_path)): continue

        print(f"  -> Procesando {scene_name} en Resize Mode ...")
        img_cond = Image.open(cond_path).convert("RGB")
        img_target = Image.open(target_path).convert("RGB")
        
        # 1. Redimensionar ambas imágenes a 256x256
        img_cond_256 = img_cond.resize(net_size, Image.Resampling.BILINEAR)
        img_target_800 = img_target.resize(base_size, Image.Resampling.BILINEAR) # Solo para comparar al final
        img_cond_800 = img_cond.resize(base_size, Image.Resampling.BILINEAR)

        # 2. Convertir a tensor [1, 3, 256, 256] y normalizar a [-1, 1]
        t_cond = ((TF.to_tensor(img_cond_256) - 0.5) / 0.5).unsqueeze(0).to(device)
        
        # 3. Generar en 256x256
        labels = torch.zeros(1, dtype=torch.long, device=device)
        pred_256 = model.generate(labels, cond=t_cond, rgb=True) # [1, 3, 256, 256]
        
        # 4. Redimensionar predicción hacia arriba a 800x1080
        # PyTorch interpolate usa (Height, Width), mientras que base_size es (Width, Height) de PIL
        pred_800 = torch.nn.functional.interpolate(pred_256, size=(base_size[1], base_size[0]), mode='bicubic', align_corners=False)
        
        # 5. Desnormalizar y preparar para guardar
        t_target_800 = ((TF.to_tensor(img_target_800) - 0.5) / 0.5).unsqueeze(0).to(device)
        t_cond_800 = ((TF.to_tensor(img_cond_800) - 0.5) / 0.5).unsqueeze(0).to(device)
        
        vis_cond = (t_cond_800 * 0.5 + 0.5).clamp(0, 1)
        vis_pred = (pred_800 * 0.5 + 0.5).clamp(0, 1)
        vis_target = (t_target_800 * 0.5 + 0.5).clamp(0, 1)
        
        # Grid: [Condición | Generación Upscaled | Target]
        grid = torch.cat([vis_cond, vis_pred, vis_target], dim=3)
        
        save_path = os.path.join(args_eval.out_dir, f"{scene_name}_resize.png")
        save_image(grid, save_path)
        print(f"  -> Guardado en {save_path}")

if __name__ == "__main__":
    main()
