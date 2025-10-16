# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Enhanced StyleGAN4 Generator and Discriminator architectures with high-resolution support and efficiency improvements."""

import numpy as np
import scipy.signal
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = True, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class EfficientSelfAttention(nn.Module):
    """Memory-efficient self-attention mechanism for high-resolution images."""
    def __init__(self, in_channels, reduction=8, window_size=64):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.window_size = window_size
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Use windowed attention for memory efficiency
        if H > self.window_size or W > self.window_size:
            return self._windowed_attention(x)
        else:
            return self._full_attention(x)
    
    def _full_attention(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate Q, K, V
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x
    
    def _windowed_attention(self, x):
        batch_size, C, H, W = x.size()
        window_size = self.window_size
        
        # Pad if necessary
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        
        # Split into windows
        x_windows = x.view(batch_size, C, H // window_size, window_size, 
                          W // window_size, window_size)
        x_windows = x_windows.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_windows = x_windows.view(-1, C, window_size, window_size)
        
        # Apply attention to each window
        query = self.query(x_windows).view(-1, C // self.reduction, window_size * window_size).permute(0, 2, 1)
        key = self.key(x_windows).view(-1, C // self.reduction, window_size * window_size)
        value = self.value(x_windows).view(-1, C, window_size * window_size)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(-1, C, window_size, window_size)
        
        # Reconstruct
        out = out.view(batch_size, H // window_size, W // window_size, C, window_size, window_size)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(batch_size, C, H, W)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H-pad_h, :W-pad_w]
        
        return self.gamma * out + x

#----------------------------------------------------------------------------

@persistence.persistent_class
class MemoryEfficientResidualBlock(nn.Module):
    """Memory-efficient residual block with gradient checkpointing."""
    def __init__(self, in_channels, out_channels, w_dim, activation='lrelu', 
                 use_attention=False, use_checkpointing=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.activation = activation
        self.use_attention = use_attention
        self.use_checkpointing = use_checkpointing
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.style1 = nn.Linear(w_dim, out_channels)
        self.style2 = nn.Linear(w_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = EfficientSelfAttention(out_channels) if use_attention else nn.Identity()
        
    def _forward_impl(self, x, w):
        residual = self.shortcut(x)
        s1 = self.style1(w).unsqueeze(-1).unsqueeze(-1)
        s2 = self.style2(w).unsqueeze(-1).unsqueeze(-1)
        
        out = self.conv1(x)
        out = bias_act.bias_act(out, act=self.activation)
        out = out * (1 + s1)
        
        out = self.conv2(out)
        out = bias_act.bias_act(out, act=self.activation)
        out = out * (1 + s2)
        
        out = self.attention(out)
        return out + residual
    
    def forward(self, x, w):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, w)
        else:
            return self._forward_impl(x, w)

#----------------------------------------------------------------------------

@persistence.persistent_class
class MultiScaleSynthesisNetwork(torch.nn.Module):
    """Multi-scale synthesis network that can generate images at multiple resolutions simultaneously."""
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        img_resolution,                 # Output image resolution.
        img_channels,                   # Number of color channels.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
        num_critical        = 2,        # Number of critically sampled layers at the end.
        first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
        first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
        last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
        margin_size         = 10,       # Number of additional pixels outside the image.
        output_scale        = 0.25,     # Scale factor for the output image.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        use_attention       = True,     # Use self-attention in higher resolution layers.
        use_residual        = True,     # Use residual connections.
        use_checkpointing   = True,     # Use gradient checkpointing for memory efficiency.
        target_scales       = None,     # List of target scales for multi-scale generation.
        **layer_kwargs,                 # Arguments for SynthesisLayer.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.channel_base = channel_base
        self.channel_max = channel_max
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.first_cutoff = first_cutoff
        self.first_stopband = first_stopband
        self.last_stopband_rel = last_stopband_rel
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_checkpointing = use_checkpointing
        self.target_scales = target_scales or [1, 2, 4, 8]  # Default scales: full, 1/2, 1/4, 1/8

        # Geometric progression of layer cutoffs and channels.
        cutoffs = [self.first_cutoff]
        stopbands = [self.first_stopband]
        for i in range(self.num_layers - 1):
            cutoffs.append(cutoffs[-1] / 2)
            stopbands.append(stopbands[-1] / 2)

        # Compute remaining layer parameters.
        sampling_rates = np.array(cutoffs) * 2
        half_widths = np.maximum(stopbands, sampling_rates / 2)
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((self.channel_base / 2) / cutoffs, self.channel_max))
        channels[-1] = self.img_channels

        # Construct layers.
        self.input = SynthesisInput(w_dim=w_dim, channels=int(channels[0]), size=int(sizes[0]),
            sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            layer = SynthesisLayer(
                w_dim=w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=int(channels[prev]), out_channels=int(channels[idx]),
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_sampling_rate=sampling_rates[prev], out_sampling_rate=sampling_rates[idx],
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx],
                use_attention=self.use_attention and idx > self.num_layers // 2,
                use_residual=self.use_residual and idx > 0,
                **layer_kwargs)
            name = f'L{idx}_{layer.out_size[0]}x{layer.out_size[1]}'
            setattr(self, name, layer)
            self.layer_names.append(name)

        # Multi-scale output layers
        self.multi_scale_outputs = nn.ModuleDict()
        for scale in self.target_scales:
            if scale != 1:  # Skip full resolution as it's already handled
                scale_res = img_resolution // scale
                self.multi_scale_outputs[f'scale_{scale}'] = nn.Sequential(
                    nn.AdaptiveAvgPool2d((scale_res, scale_res)),
                    nn.Conv2d(img_channels, img_channels, 1)
                )

    def forward(self, ws, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_layers + 1, self.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)

        # Execute layers with checkpointing for memory efficiency
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.input, ws[0])
            for name, w in zip(self.layer_names, ws[1:]):
                layer = getattr(self, name)
                x = torch.utils.checkpoint.checkpoint(layer, x, w, **layer_kwargs)
        else:
            x = self.input(ws[0])
            for name, w in zip(self.layer_names, ws[1:]):
                x = getattr(self, name)(x, w, **layer_kwargs)

        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        
        # Generate multi-scale outputs
        outputs = {'full': x}
        for scale_name, scale_layer in self.multi_scale_outputs.items():
            outputs[scale_name] = scale_layer(x)
        
        return outputs

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, num_layers={self.num_layers:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'channel_base={self.channel_base}, channel_max={self.channel_max},',
            f'num_critical={self.num_critical:d}, margin_size={self.margin_size:d},',
            f'output_scale={self.output_scale:g}, num_fp16_res={self.num_fp16_res:d},',
            f'use_checkpointing={self.use_checkpointing}, target_scales={self.target_scales}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class HighResolutionGenerator(torch.nn.Module):
    """Enhanced generator with high-resolution support and memory optimizations."""
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        use_multi_scale     = True, # Enable multi-scale generation.
        use_checkpointing   = True, # Enable gradient checkpointing.
        target_scales       = None, # Target scales for multi-scale generation.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.use_multi_scale = use_multi_scale
        self.use_checkpointing = use_checkpointing
        self.target_scales = target_scales

        # Construct mapping network
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=14, **mapping_kwargs)

        # Construct synthesis network
        if use_multi_scale:
            self.synthesis = MultiScaleSynthesisNetwork(
                w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                use_checkpointing=use_checkpointing, target_scales=target_scales, **synthesis_kwargs)
        else:
            self.synthesis = SynthesisNetwork(
                w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                use_checkpointing=use_checkpointing, **synthesis_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)

#----------------------------------------------------------------------------

@persistence.persistent_class
class MemoryEfficientDiscriminator(nn.Module):
    """Memory-efficient discriminator with gradient checkpointing."""
    def __init__(self, c_dim, img_resolution, img_channels, channel_base=32768, channel_max=512, 
                 num_scales=3, use_attention=True, use_residual=True, use_checkpointing=True):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.channel_base = channel_base
        self.channel_max = channel_max
        self.num_scales = num_scales
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_checkpointing = use_checkpointing
        
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            scale = 2 ** i
            res = img_resolution // scale
            disc = MemoryEfficientDiscriminatorBlock(
                in_channels=img_channels if i == 0 else 0, 
                tmp_channels=int(channel_base // scale),
                out_channels=int(channel_base // scale),
                resolution=res,
                img_channels=img_channels,
                first_layer_idx=i * 2,
                use_attention=use_attention and i < num_scales // 2,
                use_residual=use_residual,
                use_checkpointing=use_checkpointing
            )
            self.discriminators.append(disc)
        
        self.final_conv = nn.Conv2d(channel_base // (2 ** (num_scales - 1)), 1, 1)
        self.final_fc = nn.Linear(channel_base // (2 ** (num_scales - 1)), 1)
        
    def forward(self, img, c=None, update_emas=False):
        scales = []
        x = img
        for i in range(self.num_scales):
            if i > 0:
                x = F.avg_pool2d(x, 2)
            scales.append(x)
        
        features = []
        for i, (disc, scale_img) in enumerate(zip(self.discriminators, scales)):
            if self.use_checkpointing and self.training:
                feat = torch.utils.checkpoint.checkpoint(disc, scale_img, c, update_emas)
            else:
                feat = disc(scale_img, c, update_emas=update_emas)
            features.append(feat)
        
        combined = torch.cat(features, dim=1)
        out = self.final_conv(combined)
        out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        out = self.final_fc(out)
        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class MemoryEfficientDiscriminatorBlock(nn.Module):
    """Memory-efficient discriminator block with gradient checkpointing."""
    def __init__(self, in_channels, tmp_channels, out_channels, resolution, img_channels, 
                 first_layer_idx, use_attention=False, use_residual=False, use_checkpointing=True):
        super().__init__()
        self.in_channels = in_channels
        self.tmp_channels = tmp_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_checkpointing = use_checkpointing
        
        self.conv1 = nn.Conv2d(in_channels, tmp_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(tmp_channels, out_channels, 3, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.attention = EfficientSelfAttention(out_channels) if use_attention else nn.Identity()
        self.residual = nn.Identity() if use_residual and in_channels == out_channels else None
        self.activation = nn.LeakyReLU(0.2)
        
    def _forward_impl(self, x, c=None, update_emas=False):
        residual = self.residual(x) if self.residual is not None else None
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.attention(x)
        x = self.downsample(x)
        if residual is not None:
            residual = self.downsample(residual)
            x = x + residual
        return x
        
    def forward(self, x, c=None, update_emas=False):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, c, update_emas)
        else:
            return self._forward_impl(x, c, update_emas)

#----------------------------------------------------------------------------

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    """Normalize 2nd moment of x."""
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

# Import the original classes for compatibility
from .networks_stylegan4 import (
    FullyConnectedLayer,
    MappingNetwork,
    SynthesisInput,
    SynthesisLayer,
    SynthesisNetwork,
    Generator,
    SelfAttention,
    ResidualBlock,
    MultiScaleDiscriminator,
    DiscriminatorBlock
) 