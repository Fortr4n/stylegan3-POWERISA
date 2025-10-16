# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""StyleGAN4 Generator and Discriminator architectures with major improvements."""

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
class SelfAttention(nn.Module):
    """Self-attention mechanism for StyleGAN4."""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
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

#----------------------------------------------------------------------------

@persistence.persistent_class
class ResidualBlock(nn.Module):
    """Residual block with improved design for StyleGAN4."""
    def __init__(self, in_channels, out_channels, w_dim, activation='lrelu', use_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.activation = activation
        self.use_attention = use_attention
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Style modulation
        self.style1 = nn.Linear(w_dim, out_channels)
        self.style2 = nn.Linear(w_dim, out_channels)
        
        # Shortcut path
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
        # Attention
        if use_attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = nn.Identity()
            
    def forward(self, x, w):
        residual = self.shortcut(x)
        
        # Style modulation
        s1 = self.style1(w).unsqueeze(-1).unsqueeze(-1)
        s2 = self.style2(w).unsqueeze(-1).unsqueeze(-1)
        
        # Main path
        out = self.conv1(x)
        out = bias_act.bias_act(out, act=self.activation)
        out = out * (1 + s1)
        
        out = self.conv2(out)
        out = bias_act.bias_act(out, act=self.activation)
        out = out * (1 + s2)
        
        # Attention
        out = self.attention(out)
        
        return out + residual

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output.
        num_layers      = 8,        # Number of mapping layers.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # Construct layers.
        self.embed = None
        if self.c_dim > 0:
            self.embed = FullyConnectedLayer(self.c_dim, self.w_dim)
        for idx in range(self.num_layers):
            in_features = self.w_dim if idx == 0 and self.embed is None else self.w_dim
            layer = FullyConnectedLayer(in_features, self.w_dim, activation='lrelu', lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)
        self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            misc.assert_shape(z, [None, self.z_dim])
            x = normalize_2nd_moment(z.to(torch.float32))
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if truncation_psi != 1:
            assert self.w_avg_beta > 0
            if truncation_cutoff is None:
                truncation_cutoff = self.num_ws
            layer_idx = torch.arange(self.num_ws)[np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32, device=layer_idx.device)
            coefs = torch.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            x = x.unsqueeze(-1).coalesce() * coefs + self.w_avg.unsqueeze(0).unsqueeze(-1) * (1 - coefs)

        return x.unsqueeze(1).expand(-1, self.num_ws, -1)

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisInput(torch.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = size
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().sqrt()
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,1])
        self.register_buffer('transform', torch.eye(3, dtype=torch.float32))
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w):
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0).repeat([w.shape[0], 1, 1])
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].square().sum(dim=1, keepdim=True).sqrt()
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        m_r[:, 0, 0] = t[:, 0]  # r_c
        m_r[:, 0, 1] = -t[:, 1] # -r_s
        m_r[:, 1, 0] = t[:, 1]  # r_s
        m_r[:, 1, 1] = t[:, 0]  # r_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        m_t[:, 0, 2] = t[:, 2]  # t_x
        m_t[:, 1, 2] = t[:, 3]  # t_y
        transforms = m_r @ m_t @ transforms

        # Transform frequencies.
        freqs = freqs @ transforms[:, :2, :2].transpose(1, 2)
        freqs = freqs * transforms[:, :2, 2:].unsqueeze(1)
        phases = phases + (freqs * transforms[:, 2:3, :2]).sum(dim=2)

        # DCT-II transformation.
        x = (freqs.unsqueeze(2) * self.size).squeeze(2)
        x = x + phases.unsqueeze(2)
        x = torch.cos(x * math.pi)
        x = x * freqs.square().sum(dim=2, keepdim=True).rsqrt()

        # Ensure correct shape.
        misc.assert_shape(x, [w.shape[0], self.channels, self.size, self.size])
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={self.size:d},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb,                       # Is this the final ToRGB layer?
        is_critically_sampled,          # Does this layer use critical sampling?
        use_fp16,                       # Does this layer use FP16?

        # Input & output specifications.
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
        magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
        use_attention       = False,    # Use self-attention?
        use_residual        = False,    # Use residual connections?
    ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.use_radial_filters = use_radial_filters
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta
        self.use_attention = use_attention
        self.use_residual = use_residual

        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.out_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.out_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width, fs=self.out_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.in_sampling_rate / self.out_sampling_rate))
        assert self.in_sampling_rate / self.down_factor == self.out_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width, fs=self.in_sampling_rate))

        # Compute padding.
        pad = (self.conv_kernel - 1) // 2
        self.padding = [pad, pad, pad, pad]

        # Setup attention and residual connections.
        if self.use_attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = nn.Identity()
            
        if self.use_residual and in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = None

    def forward(self, x, w, noise_mode='random', force_fp32=False, update_emas=False):
        # Track input magnitude.
        if update_emas:
            with torch.autograd.no_grad():
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels)
            styles = styles * weight_gain

        # Execute modulated conv2d.
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
            padding=self.padding, demodulate=not self.is_torgb, input_gain=input_gain)

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r + (r == 0))
        f = f / f.sum()
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
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

    def forward(self, ws, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_layers + 1, self.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)

        # Execute layers.
        x = self.input(ws[0])
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, num_layers={self.num_layers:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'channel_base={self.channel_base}, channel_max={self.channel_max},',
            f'num_critical={self.num_critical:d}, margin_size={self.margin_size:d},',
            f'output_scale={self.output_scale:g}, num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for StyleGAN4."""
    def __init__(self, c_dim, img_resolution, img_channels, channel_base=32768, channel_max=512, 
                 num_scales=3, use_attention=True, use_residual=True):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.channel_base = channel_base
        self.channel_max = channel_max
        self.num_scales = num_scales
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Create discriminators for different scales
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            scale = 2 ** i
            res = img_resolution // scale
            disc = DiscriminatorBlock(
                in_channels=img_channels if i == 0 else 0,
                tmp_channels=int(channel_base // scale),
                out_channels=int(channel_base // scale),
                resolution=res,
                img_channels=img_channels,
                first_layer_idx=i * 2,
                use_attention=use_attention and i < num_scales // 2,
                use_residual=use_residual
            )
            self.discriminators.append(disc)
            
        # Final classification layers
        self.final_conv = nn.Conv2d(channel_base // (2 ** (num_scales - 1)), 1, 1)
        self.final_fc = nn.Linear(channel_base // (2 ** (num_scales - 1)), 1)
        
    def forward(self, img, c=None, update_emas=False):
        # Downsample image for different scales
        scales = []
        x = img
        for i in range(self.num_scales):
            if i > 0:
                x = F.avg_pool2d(x, 2)
            scales.append(x)
            
        # Process each scale
        features = []
        for i, (disc, scale_img) in enumerate(zip(self.discriminators, scales)):
            feat = disc(scale_img, c, update_emas=update_emas)
            features.append(feat)
            
        # Combine features from all scales
        combined = torch.cat(features, dim=1)
        
        # Final classification
        out = self.final_conv(combined)
        out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        out = self.final_fc(out)
        
        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(nn.Module):
    """Enhanced discriminator block for StyleGAN4."""
    def __init__(self, in_channels, tmp_channels, out_channels, resolution, img_channels, 
                 first_layer_idx, use_attention=False, use_residual=False):
        super().__init__()
        self.in_channels = in_channels
        self.tmp_channels = tmp_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Main layers
        self.conv1 = nn.Conv2d(in_channels, tmp_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(tmp_channels, out_channels, 3, padding=1)
        
        # Downsampling
        self.downsample = nn.AvgPool2d(2)
        
        # Attention
        if use_attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = nn.Identity()
            
        # Residual connection
        if use_residual and in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = None
            
        # Activation
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, c=None, update_emas=False):
        residual = self.residual(x) if self.residual is not None else None
        
        # Main path
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        
        # Attention
        x = self.attention(x)
        
        # Downsample
        x = self.downsample(x)
        
        # Residual connection
        if residual is not None:
            residual = self.downsample(residual)
            x = x + residual
            
        return x

#----------------------------------------------------------------------------

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt() 