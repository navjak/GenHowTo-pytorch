import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class GenHowToModel(nn.Module):
    def __init__(self):
        super().__init__()

        model_id = "Manojb/stable-diffusion-2-1-base"

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae") 
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet") # main UNet -> copy this to get Controlnet enc
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.config.prediction_type = "epsilon"

        # freeze VAE and CLIP
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False) # Freeze full unet initially

        # unfreeze decoder only
        for p in self.unet.up_blocks.parameters(): p.requires_grad = True
        for p in self.unet.conv_norm_out.parameters(): p.requires_grad = True
        for p in self.unet.conv_out.parameters(): p.requires_grad = True

        # Copy unet encoder for controlnet [TRAINABLE]
        self.controlnet_enc = nn.ModuleDict({
            'time_proj': copy.deepcopy(self.unet.time_proj),
            'time_embedding': copy.deepcopy(self.unet.time_embedding),
            'conv_in': copy.deepcopy(self.unet.conv_in),
            'down_blocks': copy.deepcopy(self.unet.down_blocks),
            'mid_block': copy.deepcopy(self.unet.mid_block)
        })
        self.controlnet_enc.requires_grad_(True) # Controlnet enc [TRAINABLE]

        # Zero convs
        self.zero_convs = self._make_zero_convs()


    # zero convs helper
    def _make_zero_convs(self):
        embed_dim = self.text_encoder.config.hidden_size
        seq_len = self.tokenizer.model_max_length

        # dummy pass for channel sizes
        B, C, H, W = 1, 4, 64, 64
        t = torch.tensor([1])
        emb = torch.randn(B, seq_len, embed_dim)

        with torch.no_grad():
            dummy_res_samples, dummy_mid = self._forward_control_encoder(torch.randn(B, C, H, W), t, emb)

        all_res = list(dummy_res_samples) + [dummy_mid]

        convs = nn.ModuleList()
        for res in all_res:
             channels = res.shape[1]
             # zero-init 1x1 conv
             z_conv = nn.Conv2d(channels, channels, kernel_size=1)
             z_conv.weight.data.zero_()
             z_conv.bias.data.zero_()
             convs.append(z_conv)
        return convs


    # helper for controlnet forward pass
    # replicate unet encoder pass
    def _forward_control_encoder(self, x, t, emb):

        # time embedding
        t_emb = self.controlnet_enc['time_proj'](t)
        t_emb = self.controlnet_enc['time_embedding'](t_emb)
        t_emb = t_emb.to(dtype=x.dtype)

        # pre-process
        x = self.controlnet_enc['conv_in'](x)
        res_samples = [x]

        # Down blocks
        for block in self.controlnet_enc['down_blocks']:
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                x, res = block(x, t_emb, encoder_hidden_states=emb)
            else:
                x, res = block(x, t_emb)
            res_samples.extend(res)

        # Mid block
        mid_block = self.controlnet_enc['mid_block']
        if hasattr(mid_block, "has_cross_attention") and mid_block.has_cross_attention:
            x = mid_block(x, t_emb, encoder_hidden_states=emb)
        else:
            x = mid_block(x, t_emb)

        return res_samples, x


    def predict_noise(self, z_src, z_noisy, txt, t):
        # input z_src, z_noisy - [B, 4, 64, 64]
        
        # Text encoding
        tokens = self.tokenizer(
            txt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(z_src.device)
        
        emb_txt = self.text_encoder(tokens)[0] # [B, 77, 1024]
            
        # Controlnet path [TRAINABLE]
        z_cond = z_noisy + z_src # [B, 4, 64, 64]
        
        ctrl_down_res, ctrl_mid_res = self._forward_control_encoder(z_cond, t, emb_txt)
        
        # Apply Zero Convs
        ctrl_down_fused = []
        num_down = len(ctrl_down_res)
        
        for i in range(num_down):
            ctrl_down_fused.append(self.zero_convs[i](ctrl_down_res[i]))
            
        ctrl_mid_fused = self.zero_convs[-1](ctrl_mid_res)
        
        # unet forward
        pred = self.unet(
            z_noisy, t,
            encoder_hidden_states=emb_txt,
            down_block_additional_residuals=ctrl_down_fused,
            mid_block_additional_residual=ctrl_mid_fused).sample # [B, 4, 64, 64]
        
        return pred

    def forward(self, src, tgt, txt, t):
        # input src,tgt - [B, 3, 512, 512]

        # Text encoding and CFG dropout
        if self.training and random.random() < 0.1:
            txt = [""] * len(src) # 10% dropout for cfg

        # VAE Encoding [FROZEN]
        with torch.no_grad():
            z_src = self.vae.encode(src).latent_dist.sample() * self.vae.config.scaling_factor # [B, 4, 64, 64]
            z_tgt = self.vae.encode(tgt).latent_dist.sample() * self.vae.config.scaling_factor

        # add noise using DDPM
        noise = torch.randn(z_tgt.shape, device=src.device, dtype=src.dtype)
        z_noisy = self.scheduler.add_noise(z_tgt, noise, t)

        pred = self.predict_noise(z_src, z_noisy, txt, t)

        # MSE loss
        loss = F.mse_loss(pred, noise)

        return loss, pred

