import torch 
from typing import List, Optional
import math
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # Expect t shape (batch,) or (batch,1)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb  # shape (batch, dim)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.SiLU()
        self.res_conv = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        if time_emb_dim > 0:
            self.time_proj = nn.Linear(time_emb_dim, out_ch)
        else:
            self.time_proj = None

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor]):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h)

        h = self.conv2(h)
        if t_emb is not None:
            t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t
        h = self.gn2(h)
        h = self.act(h)

        return h + self.res_conv(x)

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class DRUNet(nn.Module):
    """
    Denoising Residual U-Net with time embedding and label conditioning.
    - Fixed to 4 levels.
    - channel_mults must be length 4 (multipliers of base_channels).
    Forward signature: model(x, t, y) where
      x: (B, C, H, W)
      t: (B,) or (B,1) or scalar timestep (float/int)
      y: (B,) class labels (optional, can be None)
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Optional[List[int]] = None,
        time_emb_dim: int = 256,
        num_classes: Optional[int] = None,
        out_channels: Optional[int] = None
    ):
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 4, 8]
        if len(channel_mults) != 4:
            raise ValueError("channel_mults must have length 4 (fixed 4 levels)")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes

        # time embedding modules
        self.time_sinu = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # label embedding (if num_classes provided)
        # +1 for unconditional token (used in classifier-free guidance)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        # compute channels per level
        self.chs = [base_channels * m for m in channel_mults]  # length 4

        # input conv
        self.init_conv = nn.Conv2d(in_channels, self.chs[0], kernel_size=3, padding=1)

        # down blocks (each level has two resblocks then downsample except bottom)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        in_ch = self.chs[0]
        for i, ch in enumerate(self.chs):
            blocks = nn.ModuleList()
            blocks.append(ResBlock(in_ch, ch, time_emb_dim))
            blocks.append(ResBlock(ch, ch, time_emb_dim))
            self.down_blocks.append(blocks)
            in_ch = ch
            if i != len(self.chs) - 1:
                self.down_samples.append(Downsample(in_ch))

        # middle blocks
        mid_ch = self.chs[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_emb_dim)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim)

        # up blocks
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(len(self.chs) - 1, -1, -1):
            ch = self.chs[i]
            if i != len(self.chs) - 1:
                # upsample from previous level channels to current level channels
                self.up_samples.append(Upsample(prev_ch, ch))
                # after concatenation, channel becomes ch + ch (skip)
                blocks = nn.ModuleList([ResBlock(ch * 2, ch, time_emb_dim), ResBlock(ch, ch, time_emb_dim)])
                self.up_blocks.append(blocks)
            prev_ch = ch

        # final conv to map to out_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, self.chs[0]), num_channels=self.chs[0]),
            nn.SiLU(),
            nn.Conv2d(self.chs[0], self.out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        x: (B, C, H, W)
        t: (B,) or (B,1) or scalar
        y: (B,) class labels (optional)
        returns: (B, C, H, W) predicted noise / denoised output depending on training
        """
        # time embedding
        t_emb = self.time_sinu(t)
        t_emb = self.time_mlp(t_emb)

        # add label embedding if provided
        if y is not None and self.num_classes is not None:
            y_emb = self.label_emb(y)
            t_emb = t_emb + y_emb

        # input conv
        h = self.init_conv(x)

        # down pass
        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if i < len(self.down_samples):
                h = self.down_samples[i](h)

        # middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # up pass
        # Note: up_samples and up_blocks are stored for levels 3->0 (excluding topmost no upsample)
        up_idx = 0
        for i in range(len(self.chs) - 1, -1, -1):
            if i == len(self.chs) - 1:
                # top of decoder, just use skip at this level (no upsample)
                # (we're already at this resolution)
                pass
            else:
                h = self.up_samples[up_idx](h)
                # concatenate skip connection
                skip = skips[i]
                h = torch.cat([h, skip], dim=1)
                blocks = self.up_blocks[up_idx]
                for block in blocks:
                    h = block(h, t_emb)
                up_idx += 1

        out = self.final_conv(h)
        return out