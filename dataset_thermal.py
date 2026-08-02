import os
from pathlib import Path
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

class ThermalPairedDataset(Dataset):
    """
    Dataset para cargar pares de imágenes térmicas (Condición -> Target).
    Ejemplo: 120s (disipada) -> 005s (reciente).
    
    Aplica Data Augmentation (Random Crop y Random Flip) de manera SINCRONIZADA,
    asegurando que el recorte se haga en la misma ubicación exacta para ambas imágenes.
    """
    
    def __init__(self, scenes_dir, cond_folder="120", target_folder="005", patch_size=256, is_train=True):
        """
        Args:
            scenes_dir (str): Ruta a la carpeta principal 'Scenes'.
            cond_folder (str): Nombre de la carpeta de entrada (ej. "120", "060").
            target_folder (str): Nombre de la carpeta objetivo (ej. "005", "GT").
            patch_size (int): Tamaño del parche cuadrado a extraer (ej. 256).
            is_train (bool): Si es True, aplica recortes aleatorios (Data Augmentation). 
                             Si es False, hace un recorte central (Center Crop) para evaluación.
        """
        super().__init__()
        self.scenes_dir = Path(scenes_dir)
        self.cond_folder = cond_folder
        self.target_folder = target_folder
        self.patch_size = patch_size
        self.is_train = is_train
        
        self.samples = []
        
        # Escaneamos todas las escenas disponibles buscando los pares
        for scene_name in sorted(os.listdir(self.scenes_dir)):
            scene_path = self.scenes_dir / scene_name
            if not scene_path.is_dir():
                continue
                
            cond_path = scene_path / self.cond_folder / "TH.png"
            target_path = scene_path / self.target_folder / "TH.png"
            
            # Solo agregamos a la lista si ambas imágenes existen
            if cond_path.exists() and target_path.exists():
                self.samples.append((cond_path, target_path))
                
        print(f"[Dataset] Encontrados {len(self.samples)} pares de imágenes para {cond_folder}s -> {target_folder}s.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond_path, target_path = self.samples[idx]
        
        # 1. Carga de imágenes
        # Convertimos a RGB para eliminar el canal Alpha (Transparencia) de los PNG de FLIR
        img_cond = Image.open(cond_path).convert("RGB")
        img_target = Image.open(target_path).convert("RGB")
        
        # 2. Corrección de discrepancias geométricas (El problema 808x1080 vs 801x1080)
        # Forzamos ambas imágenes a una resolución base común antes de recortar parches.
        # Usamos (800, 1080) como estándar seguro para estas imágenes térmicas.
        base_size = (800, 1080) # (Ancho, Alto) en PIL
        if img_cond.size != base_size:
            img_cond = img_cond.resize(base_size, Image.Resampling.BILINEAR)
        if img_target.size != base_size:
            img_target = img_target.resize(base_size, Image.Resampling.BILINEAR)
            
        # 3. Extracción de Parches Sincronizada (Data Augmentation)
        if self.is_train:
            # Obtenemos coordenadas aleatorias (i, j, h, w) para el recorte
            i, j, h, w = transforms.RandomCrop.get_params(img_cond, output_size=(self.patch_size, self.patch_size))
            img_cond = TF.crop(img_cond, i, j, h, w)
            img_target = TF.crop(img_target, i, j, h, w)
            
            # Flip horizontal aleatorio (50% de probabilidad)
            if random.random() > 0.5:
                img_cond = TF.hflip(img_cond)
                img_target = TF.hflip(img_target)
        else:
            # Si es validación/test, siempre tomamos el centro para consistencia
            img_cond = TF.center_crop(img_cond, output_size=(self.patch_size, self.patch_size))
            img_target = TF.center_crop(img_target, output_size=(self.patch_size, self.patch_size))

        # 4. Conversión a Tensores PyTorch y Normalización a [-1, 1]
        # ToTensor() convierte de PIL (0-255) a tensor flotante (0.0 a 1.0)
        img_cond = TF.to_tensor(img_cond)
        img_target = TF.to_tensor(img_target)
        
        # Normalizamos restando 0.5 y dividiendo por 0.5 para que queden en [-1, 1]
        # Esto es estándar en modelos de generación.
        img_cond = (img_cond - 0.5) / 0.5
        img_target = (img_target - 0.5) / 0.5
        
        return img_cond, img_target

# --- Bloque de prueba (solo se ejecuta si corres este archivo directamente) ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Creamos un dataset de prueba apuntando a tu carpeta Scenes
    dataset = ThermalPairedDataset(scenes_dir="./Scenes", cond_folder="120", target_folder="005", patch_size=256, is_train=True)
    
    if len(dataset) > 0:
        cond, target = dataset[0]
        print(f"Tensor Condición shape: {cond.shape}, Rango: [{cond.min():.2f}, {cond.max():.2f}]")
        print(f"Tensor Target shape: {target.shape}, Rango: [{target.min():.2f}, {target.max():.2f}]")
        
        # Des-normalizamos de [-1, 1] a [0, 1] para visualizarlas
        cond_vis = (cond * 0.5) + 0.5
        target_vis = (target * 0.5) + 0.5
        
        # Visualizamos el primer parche aleatorio extraído
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(cond_vis.permute(1, 2, 0).numpy())
        axes[0].set_title("Input 120s (Condición)")
        axes[1].imshow(target_vis.permute(1, 2, 0).numpy())
        axes[1].set_title("Target 005s (Ground Truth)")
        plt.show()
