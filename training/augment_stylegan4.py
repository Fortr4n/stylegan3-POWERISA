# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Enhanced data augmentation pipeline for StyleGAN4."""

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------

class StyleGAN4AugmentPipe(nn.Module):
    """Enhanced augmentation pipeline for StyleGAN4 with advanced techniques."""
    
    def __init__(self, xflip=0, rotate90=0, rotate180=0, rotate270=0, xint=0, 
                 scale=0, rotate=0, aniso=0, xfrac=0, brightness=0, contrast=0, 
                 lumaflip=0, hue=0, saturation=0, imgfilter=0, noise=0, cutout=0,
                 use_advanced_aug=True, use_style_mixing=True):
        super().__init__()
        self.xflip = float(xflip)
        self.rotate90 = float(rotate90)
        self.rotate180 = float(rotate180)
        self.rotate270 = float(rotate270)
        self.xint = float(xint)
        self.scale = float(scale)
        self.rotate = float(rotate)
        self.aniso = float(aniso)
        self.xfrac = float(xfrac)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.lumaflip = float(lumaflip)
        self.hue = float(hue)
        self.saturation = float(saturation)
        self.imgfilter = float(imgfilter)
        self.noise = float(noise)
        self.cutout = float(cutout)
        self.use_advanced_aug = use_advanced_aug
        self.use_style_mixing = use_style_mixing
        
        # Advanced augmentation parameters
        self.color_jitter_prob = 0.8
        self.geometric_aug_prob = 0.5
        self.noise_aug_prob = 0.3
        self.cutout_prob = 0.2
        
        # Initialize filters
        self.register_buffer('Hz_geom', upfirdn2d.setup_filter([1, 2, 1], device='cpu'))
        self.register_buffer('Hz_color', upfirdn2d.setup_filter([1, 2, 1], device='cpu'))
        
    def forward(self, images, debug=False):
        if not self.training:
            return images
            
        batch_size, num_channels, height, width = images.shape
        device = images.device
        dtype = images.dtype
        
        # Apply basic augmentations
        if self.xflip > 0:
            images = self._apply_xflip(images)
            
        if self.rotate90 > 0:
            images = self._apply_rotate90(images)
            
        if self.rotate180 > 0:
            images = self._apply_rotate180(images)
            
        if self.rotate270 > 0:
            images = self._apply_rotate270(images)
            
        # Apply advanced augmentations
        if self.use_advanced_aug:
            images = self._apply_advanced_augmentations(images)
            
        return images
    
    def _apply_xflip(self, images):
        """Apply horizontal flip with probability."""
        if torch.rand(1, device=images.device) < self.xflip:
            images = torch.flip(images, dims=[3])
        return images
    
    def _apply_rotate90(self, images):
        """Apply 90-degree rotation with probability."""
        if torch.rand(1, device=images.device) < self.rotate90:
            images = torch.rot90(images, k=1, dims=[2, 3])
        return images
    
    def _apply_rotate180(self, images):
        """Apply 180-degree rotation with probability."""
        if torch.rand(1, device=images.device) < self.rotate180:
            images = torch.rot90(images, k=2, dims=[2, 3])
        return images
    
    def _apply_rotate270(self, images):
        """Apply 270-degree rotation with probability."""
        if torch.rand(1, device=images.device) < self.rotate270:
            images = torch.rot90(images, k=3, dims=[2, 3])
        return images
    
    def _apply_advanced_augmentations(self, images):
        """Apply advanced augmentation techniques."""
        batch_size = images.shape[0]
        
        # Color jittering
        if torch.rand(1) < self.color_jitter_prob:
            images = self._color_jitter(images)
            
        # Geometric augmentations
        if torch.rand(1) < self.geometric_aug_prob:
            images = self._geometric_augment(images)
            
        # Noise augmentation
        if torch.rand(1) < self.noise_aug_prob:
            images = self._noise_augment(images)
            
        # Cutout augmentation
        if torch.rand(1) < self.cutout_prob:
            images = self._cutout_augment(images)
            
        return images
    
    def _color_jitter(self, images):
        """Apply color jittering augmentation."""
        batch_size = images.shape[0]
        
        # Brightness
        if self.brightness > 0:
            brightness_factor = torch.rand(batch_size, 1, 1, 1, device=images.device) * 2 - 1
            brightness_factor *= self.brightness
            images = images + brightness_factor
            images = torch.clamp(images, -1, 1)
            
        # Contrast
        if self.contrast > 0:
            contrast_factor = torch.rand(batch_size, 1, 1, 1, device=images.device) * 2 - 1
            contrast_factor = 1 + contrast_factor * self.contrast
            mean = images.mean(dim=[2, 3], keepdim=True)
            images = (images - mean) * contrast_factor + mean
            images = torch.clamp(images, -1, 1)
            
        # Hue and saturation (for RGB images)
        if images.shape[1] == 3 and (self.hue > 0 or self.saturation > 0):
            images = self._hsv_augment(images)
            
        return images
    
    def _hsv_augment(self, images):
        """Apply HSV augmentation."""
        batch_size = images.shape[0]
        
        # Convert to HSV
        images_hsv = self._rgb_to_hsv(images)
        
        # Hue augmentation
        if self.hue > 0:
            hue_shift = torch.rand(batch_size, 1, 1, 1, device=images.device) * 2 - 1
            hue_shift *= self.hue
            images_hsv[:, 0] = images_hsv[:, 0] + hue_shift
            images_hsv[:, 0] = torch.clamp(images_hsv[:, 0], 0, 1)
            
        # Saturation augmentation
        if self.saturation > 0:
            sat_factor = torch.rand(batch_size, 1, 1, 1, device=images.device) * 2 - 1
            sat_factor = 1 + sat_factor * self.saturation
            images_hsv[:, 1] = images_hsv[:, 1] * sat_factor
            images_hsv[:, 1] = torch.clamp(images_hsv[:, 1], 0, 1)
            
        # Convert back to RGB
        images = self._hsv_to_rgb(images_hsv)
        return images
    
    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV."""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        max_rgb, _ = torch.max(rgb, dim=1)
        min_rgb, _ = torch.min(rgb, dim=1)
        diff = max_rgb - min_rgb
        
        # Hue
        h = torch.zeros_like(max_rgb)
        h[max_rgb == r] = (60 * ((g[max_rgb == r] - b[max_rgb == r]) / diff[max_rgb == r]) % 360) / 360
        h[max_rgb == g] = (60 * ((b[max_rgb == g] - r[max_rgb == g]) / diff[max_rgb == g] + 2) % 360) / 360
        h[max_rgb == b] = (60 * ((r[max_rgb == b] - g[max_rgb == b]) / diff[max_rgb == b] + 4) % 360) / 360
        
        # Saturation
        s = torch.zeros_like(max_rgb)
        s[max_rgb != 0] = diff[max_rgb != 0] / max_rgb[max_rgb != 0]
        
        # Value
        v = max_rgb
        
        return torch.stack([h, s, v], dim=1)
    
    def _hsv_to_rgb(self, hsv):
        """Convert HSV to RGB."""
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        
        h = h * 360
        c = v * s
        x = c * (1 - torch.abs((h / 60) % 2 - 1))
        m = v - c
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        mask = (h >= 0) & (h < 60)
        r[mask] = c[mask]
        g[mask] = x[mask]
        
        mask = (h >= 60) & (h < 120)
        r[mask] = x[mask]
        g[mask] = c[mask]
        
        mask = (h >= 120) & (h < 180)
        g[mask] = c[mask]
        b[mask] = x[mask]
        
        mask = (h >= 180) & (h < 240)
        g[mask] = x[mask]
        b[mask] = c[mask]
        
        mask = (h >= 240) & (h < 300)
        r[mask] = x[mask]
        b[mask] = c[mask]
        
        mask = (h >= 300) & (h < 360)
        r[mask] = c[mask]
        b[mask] = x[mask]
        
        r = r + m
        g = g + m
        b = b + m
        
        return torch.stack([r, g, b], dim=1)
    
    def _geometric_augment(self, images):
        """Apply geometric augmentations."""
        batch_size = images.shape[0]
        
        # Random scaling
        if self.scale > 0:
            scale_factor = torch.rand(batch_size, 1, 1, 1, device=images.device) * 2 - 1
            scale_factor = 1 + scale_factor * self.scale
            images = F.interpolate(images, scale_factor=scale_factor.mean().item(), mode='bilinear', align_corners=False)
            
        # Random rotation
        if self.rotate > 0:
            angle = torch.rand(batch_size, device=images.device) * 2 - 1
            angle *= self.rotate * np.pi / 180
            images = self._rotate_images(images, angle)
            
        return images
    
    def _rotate_images(self, images, angles):
        """Apply rotation to images."""
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        
        # Create rotation matrices
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, height, device=images.device),
                                       torch.linspace(-1, 1, width, device=images.device))
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Apply rotation
        for i in range(batch_size):
            rotation_matrix = torch.tensor([[cos_a[i], -sin_a[i]],
                                         [sin_a[i], cos_a[i]]], device=images.device)
            grid[i] = torch.matmul(rotation_matrix, grid[i].view(2, -1)).view(2, height, width)
            
        # Sample using grid
        images = F.grid_sample(images, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)
        return images
    
    def _noise_augment(self, images):
        """Apply noise augmentation."""
        if self.noise > 0:
            noise = torch.randn_like(images) * self.noise
            images = images + noise
            images = torch.clamp(images, -1, 1)
        return images
    
    def _cutout_augment(self, images):
        """Apply cutout augmentation."""
        if self.cutout > 0:
            batch_size, channels, height, width = images.shape
            cutout_size = int(min(height, width) * self.cutout)
            
            for i in range(batch_size):
                if torch.rand(1) < 0.5:
                    # Random cutout
                    x = torch.randint(0, width - cutout_size + 1, (1,))
                    y = torch.randint(0, height - cutout_size + 1, (1,))
                    images[i, :, y:y+cutout_size, x:x+cutout_size] = 0
                    
        return images

#----------------------------------------------------------------------------

class AdaptiveAugmentPipe(StyleGAN4AugmentPipe):
    """Adaptive augmentation pipeline that adjusts strength based on training progress."""
    
    def __init__(self, xflip=0, rotate90=0, rotate180=0, rotate270=0, xint=0, 
                 scale=0, rotate=0, aniso=0, xfrac=0, brightness=0, contrast=0, 
                 lumaflip=0, hue=0, saturation=0, imgfilter=0, noise=0, cutout=0,
                 use_advanced_aug=True, use_style_mixing=True, target_p=0.6):
        super().__init__(xflip, rotate90, rotate180, rotate270, xint, scale, rotate, 
                        aniso, xfrac, brightness, contrast, lumaflip, hue, saturation, 
                        imgfilter, noise, cutout, use_advanced_aug, use_style_mixing)
        
        self.target_p = target_p
        self.current_p = 0.0
        self.ada_speed = 500  # Number of images to process before adjusting p
        
    def update_p(self, real_pred, fake_pred):
        """Update augmentation probability based on discriminator predictions."""
        if not self.training:
            return
            
        # Compute prediction accuracy
        real_accuracy = (real_pred > 0).float().mean()
        fake_accuracy = (fake_pred < 0).float().mean()
        accuracy = (real_accuracy + fake_accuracy) / 2
        
        # Update p
        if accuracy > self.target_p:
            self.current_p = min(self.current_p + 1e-6, 1.0)
        else:
            self.current_p = max(self.current_p - 1e-6, 0.0)
            
        # Adjust augmentation probabilities based on current_p
        self.color_jitter_prob = self.current_p * 0.8
        self.geometric_aug_prob = self.current_p * 0.5
        self.noise_aug_prob = self.current_p * 0.3
        self.cutout_prob = self.current_p * 0.2

#----------------------------------------------------------------------------

class ProgressiveAugmentPipe(StyleGAN4AugmentPipe):
    """Progressive augmentation pipeline that gradually increases complexity."""
    
    def __init__(self, xflip=0, rotate90=0, rotate180=0, rotate270=0, xint=0, 
                 scale=0, rotate=0, aniso=0, xfrac=0, brightness=0, contrast=0, 
                 lumaflip=0, hue=0, saturation=0, imgfilter=0, noise=0, cutout=0,
                 use_advanced_aug=True, use_style_mixing=True):
        super().__init__(xflip, rotate90, rotate180, rotate270, xint, scale, rotate, 
                        aniso, xfrac, brightness, contrast, lumaflip, hue, saturation, 
                        imgfilter, noise, cutout, use_advanced_aug, use_style_mixing)
        
        # Progressive stages
        self.stages = [
            {'kimg': 1000, 'color_jitter': 0.0, 'geometric': 0.0, 'noise': 0.0, 'cutout': 0.0},
            {'kimg': 5000, 'color_jitter': 0.4, 'geometric': 0.25, 'noise': 0.15, 'cutout': 0.1},
            {'kimg': 10000, 'color_jitter': 0.8, 'geometric': 0.5, 'noise': 0.3, 'cutout': 0.2},
            {'kimg': float('inf'), 'color_jitter': 0.8, 'geometric': 0.5, 'noise': 0.3, 'cutout': 0.2}
        ]
        
    def get_current_stage(self, cur_nimg):
        """Get current training stage based on number of images processed."""
        cur_kimg = cur_nimg / 1000
        for stage in self.stages:
            if cur_kimg <= stage['kimg']:
                return stage
        return self.stages[-1]
    
    def forward(self, images, cur_nimg=0, debug=False):
        if not self.training:
            return images
            
        # Get current stage
        stage = self.get_current_stage(cur_nimg)
        
        # Update probabilities based on current stage
        self.color_jitter_prob = stage['color_jitter']
        self.geometric_aug_prob = stage['geometric']
        self.noise_aug_prob = stage['noise']
        self.cutout_prob = stage['cutout']
        
        return super().forward(images, debug) 