import os
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF
from model_drunet import DRUNet

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluación del Mejor Modelo Baseline en: {device} ===")

    # 1. Configuración de rutas
    model_path = "checkpoints_baseline/unet_baseline_best.pt"
    scenes_dir = "Scenes"
    output_dir = "eval_results_baseline"
    os.makedirs(output_dir, exist_ok=True)

    # Verificar que el modelo exista
    if not os.path.exists(model_path):
        print(f"[Error] No se encontró el modelo en {model_path}. Asegúrate de haber entrenado primero.")
        return

    # 2. Cargar el modelo
    model = DRUNet(in_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("[*] Modelo unet_baseline_best.pt cargado con éxito.")

    # 3. Seleccionar algunas escenas de prueba (puedes cambiar los nombres aquí)
    # Por defecto, tomamos las últimas escenas asumiendo que cayeron en validación
    all_scenes = sorted(os.listdir(scenes_dir))
    test_scenes = all_scenes[-5:] # Tomamos las 5 últimas escenas
    print(f"[*] Evaluando escenas: {test_scenes}")

    base_size = (800, 1080)

    with torch.no_grad():
        for scene_name in test_scenes:
            cond_dir = os.path.join(scenes_dir, scene_name, "120")
            target_dir = os.path.join(scenes_dir, scene_name, "005")
            
            cond_path = os.path.join(cond_dir, "TH.png")
            if not os.path.exists(cond_path):
                cond_path = os.path.join(cond_dir, "TH.jpg")
                
            target_path = os.path.join(target_dir, "TH.png")
            if not os.path.exists(target_path):
                target_path = os.path.join(target_dir, "TH.jpg")
            
            if not os.path.exists(cond_path) or not os.path.exists(target_path):
                continue
                
            # Cargar imágenes completas
            img_cond = Image.open(cond_path).convert("RGB")
            img_target = Image.open(target_path).convert("RGB")
            
            # Redimensionar al tamaño estándar seguro
            img_cond = img_cond.resize(base_size, Image.Resampling.BILINEAR)
            img_target = img_target.resize(base_size, Image.Resampling.BILINEAR)
            
            # Convertir a tensor y normalizar a [-1, 1]
            t_cond = TF.to_tensor(img_cond).unsqueeze(0).to(device)
            t_cond = (t_cond - 0.5) / 0.5
            
            t_target = TF.to_tensor(img_target).unsqueeze(0).to(device)
            t_target = (t_target - 0.5) / 0.5
            
            # Generar predicción de la imagen COMPLETA (sin recortes)
            # Pasamos t=0 al igual que en el entrenamiento
            t_dummy = torch.zeros(1, device=device)
            pred = model(t_cond, t_dummy)
            
            # Desnormalizar a [0, 1] para guardar
            vis_cond = (t_cond * 0.5) + 0.5
            vis_pred = (pred * 0.5) + 0.5
            vis_target = (t_target * 0.5) + 0.5
            
            # Crear una grilla horizontal: [Input (120s) | Predicción | Target (5s)]
            grid = torch.cat([vis_cond, vis_pred, vis_target], dim=3) # Concatenar a lo ancho (W)
            
            # Guardar la imagen de alta resolución
            out_path = os.path.join(output_dir, f"{scene_name}_full_comparison.png")
            save_image(grid, out_path)
            print(f"  -> Guardado: {out_path}")

    print(f"\n[+] Evaluación finalizada. Revisa la carpeta '{output_dir}'.")

if __name__ == "__main__":
    evaluate_model()
