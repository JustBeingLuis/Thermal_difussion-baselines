import torch
import torch.nn as nn

class PatchGAN(nn.Module):
    """
    Discriminador PatchGAN clásico de Pix2Pix.
    No predice un solo valor (Real/Fake) para toda la imagen, sino una matriz N x N,
    donde cada píxel clasifica si ese "parche" local es real o falso.
    Esto fomenta que el generador respete las altas frecuencias (detalles y bordes).
    """
    def __init__(self, in_channels=6, ndf=64):
        super(PatchGAN, self).__init__()
        
        # PatchGAN está compuesto por bloques: Conv -> BatchNorm -> LeakyReLU
        # Excepciones: 
        # - La primera capa no usa BatchNorm
        # - La última capa tiene stride 1 y no usa activación (devuelve logits puros)
        
        def conv_block(in_c, out_c, normalize=True, stride=2):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *conv_block(in_channels, ndf, normalize=False),        # (ndf, H/2, W/2)
            *conv_block(ndf, ndf * 2),                             # (ndf*2, H/4, W/4)
            *conv_block(ndf * 2, ndf * 4),                         # (ndf*4, H/8, W/8)
            *conv_block(ndf * 4, ndf * 8, stride=1),               # (ndf*8, H/8, W/8)
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1) # (1, H/8, W/8) -> Logits
        )

    def forward(self, img_cond, img_target):
        # Concatena la condición (120s) y el target/predicción (5s) en los canales
        # Esto le da al discriminador el contexto de QUÉ debería estar evaluando
        img_input = torch.cat((img_cond, img_target), dim=1)
        return self.model(img_input)
