import torch
import torch.nn as nn
from model_jit import JiT_models
from model_drunet import DRUNet
from model_unet import UNet


class Denoiser(nn.Module):
    def __init__(
        self,
        args,
        model_name='DRUnet', # 'DRUnet', 'JiT-B/16',
    ):
        super().__init__()

        # Get in_channels from args, default to 1 for backward compatibility
        in_channels = getattr(args, 'in_channels', 1)

        if model_name == 'DRUnet':
        
            self.net = DRUNet(
                in_channels=in_channels,
                out_channels=3 if in_channels == 6 else in_channels,
                base_channels=args.base_channels,
                time_emb_dim=32,
                num_classes=args.class_num
            )
        
        elif model_name.startswith('JiT'):
            self.net = JiT_models[model_name](
                input_size=args.img_size,
                in_channels=in_channels,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
            )

        elif model_name.startswith('Unet'):

            self.net = UNet(
                input_channels=in_channels,
                input_height=args.img_size,
                ch=32,
                ch_mult=(1, 2, 4, 8),
                num_res_blocks=6,
                attn_resolutions=(16, 8),
                resamp_with_conv=True,
            )   
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.in_channels = in_channels

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        self.loss = args.loss # 'sup', 'noisy', 'noise2noise', 'gr2r'
        self.base_sigma  = 0.2 # standard deviation of noise added to input during training when noisy_input is True
        self.pred = getattr(args, 'pred', 'eps') # 'x', 'v', or 'eps'
        self.w    = getattr(args, 'w', 'none')   # 'x', 'v', or 'none'

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)
        # t = torch.rand(n, device=device)
        # return t

    def c_pred(self, z, pred, t):
        # Deprecated: The loss directly targets the raw output for x, eps, and v.
        return pred

    def forward(self, x, labels, cond=None):



        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale


        if self.loss == "sup":
            y1 = x
            y2 = x

        if self.loss == "noisy":
            y1 = x + torch.randn_like(x) * self.base_sigma
            y2 = y1

        if self.loss == "noise2noise":
            y1 = x + torch.randn_like(x) * self.base_sigma
            y2 = x + torch.randn_like(x) * self.base_sigma

        if self.loss == "gr2r":
            tau = 0.5

            y = x + torch.randn_like(x) * self.base_sigma

            w = torch.randn_like(x) * self.base_sigma 
            y1 = y + w * tau * t
            y2 = y - w * (1 / tau) * t


        z  = t * y1 + (1 - t) * e

        if cond is not None:
            net_input = torch.cat([cond, z], dim=1)
        else:
            net_input = z

        pred = self.net(net_input, t.flatten(), labels_dropped)
        pred = self.c_pred(z, pred, t)

        if self.pred == 'eps':
            loss = (e - pred) ** 2
        elif self.pred == 'v':
            v_target = y1 - e
            loss = (v_target - pred) ** 2
        else:
            loss = (y2 - pred) ** 2

        if self.w == 'v':
            w = (1 - t).clamp_min(self.t_eps)
            w = 1 / (w ** 2)
        elif self.w == 'x':
            w = 1.0
        else:
            w = 1.0


        loss = loss.mean(dim=(1, 2, 3)) * w
        loss = loss.mean()

        # self-supervised denoising loss
        # x_hat = self.net(y1, torch.ones_like(t).flatten(), labels_dropped)
        # loss2 = (y2 - x_hat) ** 2
        # loss2 = loss2.mean(dim=(1, 2, 3)).mean()

        return loss # + loss2 * 0.01

    @torch.no_grad()
    def generate(self, labels, cond=None, rgb=None):
        # Use model's in_channels if rgb not specified
        if rgb is None:
            n_channels = 3 if cond is not None else self.in_channels
        else:
            n_channels = 3 if rgb else 1

        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, n_channels, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels, cond)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, cond)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, cond=None):
        
        if cond is not None:
            net_input = torch.cat([cond, z], dim=1)
        else:
            net_input = z

        if self.cfg_scale == 1.0:
            if self.pred == 'x':
                x_cond = self.net(net_input, t.flatten(), labels)
                return (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
            elif self.pred == 'v':
                return self.net(net_input, t.flatten(), labels)
            elif self.pred == 'eps':
                eps_cond = self.net(net_input, t.flatten(), labels)
                return (z - eps_cond) / t.clamp_min(1e-5)

        # CFG active (cfg_scale != 1.0)
        # unconditional
        if self.pred == 'x':
            x_uncond = self.net(net_input, t.flatten(), torch.full_like(labels, self.num_classes))
            v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        elif self.pred == 'v':
            v_uncond = self.net(net_input, t.flatten(), torch.full_like(labels, self.num_classes))
        elif self.pred == 'eps':
            eps_uncond = self.net(net_input, t.flatten(), torch.full_like(labels, self.num_classes))
            v_uncond = (z - eps_uncond) / t.clamp_min(1e-5)

        # conditional
        if self.pred == 'x':
            x_cond = self.net(net_input, t.flatten(), labels)
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        elif self.pred == 'v':
            v_cond = self.net(net_input, t.flatten(), labels)
        elif self.pred == 'eps':
            eps_cond = self.net(net_input, t.flatten(), labels)
            v_cond = (z - eps_cond) / t.clamp_min(1e-5)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, cond=None):
        v_pred = self._forward_sample(z, t, labels, cond)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, cond=None):
        v_pred_t = self._forward_sample(z, t, labels, cond)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, cond)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
