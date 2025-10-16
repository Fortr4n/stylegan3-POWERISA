# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Enhanced loss functions for StyleGAN4 with contrastive learning and improved stability."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """Contrastive learning loss for StyleGAN4."""
    def __init__(self, temperature=0.07, queue_size=8192):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.register_buffer("queue", torch.randn(512, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def forward(self, features, labels=None):
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute logits
        logits = torch.mm(features, self.queue.clone().detach().t())
        
        # Labels are the diagonal (positive pairs)
        labels = torch.arange(batch_size, device=features.device)
        
        # Compute loss
        loss = F.cross_entropy(logits / self.temperature, labels)
        
        # Update queue
        with torch.no_grad():
            ptr = int(self.queue_ptr)
            self.queue[:, ptr:ptr + batch_size] = features.t()
            ptr = (ptr + batch_size) % self.queue_size
            self.queue_ptr[0] = ptr
            
        return loss

#----------------------------------------------------------------------------

class StyleGAN4Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, 
                 pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, 
                 blur_init_sigma=0, blur_fade_kimg=0, contrastive_weight=0.1, 
                 feature_matching_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.contrastive_weight = contrastive_weight
        self.feature_matching_weight = feature_matching_weight
        self.perceptual_weight  = perceptual_weight
        
        # Initialize contrastive loss
        self.contrastive_loss = ContrastiveLoss()
        
        # Feature extractor for perceptual loss
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self):
        """Build a simple feature extractor for perceptual loss."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        return model.to(self.device)

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def compute_perceptual_loss(self, real_img, fake_img):
        """Compute perceptual loss between real and fake images."""
        real_features = self.feature_extractor(real_img)
        fake_features = self.feature_extractor(fake_img)
        return F.mse_loss(real_features, fake_features)

    def compute_feature_matching_loss(self, real_img, fake_img):
        """Compute feature matching loss between real and fake images."""
        # Extract features from discriminator
        if hasattr(self.D, 'get_features'):
            real_features = self.D.get_features(real_img)
            fake_features = self.D.get_features(fake_img)
            
            # Compute L1 loss between features
            loss = 0
            for real_feat, fake_feat in zip(real_features, fake_features):
                loss += F.l1_loss(real_feat, fake_feat)
            return loss
        return torch.tensor(0.0, device=self.device)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                
                # Add contrastive loss
                if self.contrastive_weight > 0:
                    contrastive_loss = self.contrastive_loss(gen_img.view(gen_img.size(0), -1))
                    loss_Gmain = loss_Gmain + self.contrastive_weight * contrastive_loss
                    training_stats.report('Loss/G/contrastive', contrastive_loss)
                
                # Add perceptual loss
                if self.perceptual_weight > 0:
                    perceptual_loss = self.compute_perceptual_loss(real_img, gen_img)
                    loss_Gmain = loss_Gmain + self.perceptual_weight * perceptual_loss
                    training_stats.report('Loss/G/perceptual', perceptual_loss)
                
                # Add feature matching loss
                if self.feature_matching_weight > 0:
                    feature_loss = self.compute_feature_matching_loss(real_img, gen_img)
                    loss_Gmain = loss_Gmain + self.feature_matching_weight * feature_loss
                    training_stats.report('Loss/G/feature_matching', feature_loss)
                
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, update_emas=True)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dreal = loss_Dreal + loss_Dgen
            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.mean().mul(gain).backward()

        # Dreg: Apply R1 regularization.
        if phase in ['Dreg', 'Dboth']:
            if self.r1_gamma != 0:
                with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().reshape(r1_grads.shape[0], -1).sum(1).mean()
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                training_stats.report('Loss/r1_penalty', r1_penalty)
                training_stats.report('Loss/D/reg', loss_Dr1)
            else:
                loss_Dr1 = 0
            with torch.autograd.profiler.record_function('Dr1_backward'):
                loss_Dr1.mean().mul(gain).backward()

#----------------------------------------------------------------------------

class AdaptiveLoss(StyleGAN4Loss):
    """Adaptive loss that adjusts weights based on training progress."""
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, 
                 pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, 
                 blur_init_sigma=0, blur_fade_kimg=0, contrastive_weight=0.1, 
                 feature_matching_weight=1.0, perceptual_weight=0.1):
        super().__init__(device, G, D, augment_pipe, r1_gamma, style_mixing_prob, 
                        pl_weight, pl_batch_shrink, pl_decay, pl_no_weight_grad, 
                        blur_init_sigma, blur_fade_kimg, contrastive_weight, 
                        feature_matching_weight, perceptual_weight)
        
        # Adaptive weights
        self.adaptive_weights = {
            'contrastive': contrastive_weight,
            'perceptual': perceptual_weight,
            'feature_matching': feature_matching_weight
        }
        
        # Moving averages for loss components
        self.loss_history = {
            'contrastive': [],
            'perceptual': [],
            'feature_matching': []
        }
        
    def update_adaptive_weights(self, cur_nimg):
        """Update adaptive weights based on training progress."""
        if len(self.loss_history['contrastive']) < 100:
            return
            
        # Compute moving averages
        window = 50
        for key in self.loss_history:
            if len(self.loss_history[key]) >= window:
                recent_losses = self.loss_history[key][-window:]
                avg_loss = np.mean(recent_losses)
                
                # Adjust weight based on loss magnitude
                if avg_loss > 1.0:
                    self.adaptive_weights[key] *= 0.95  # Reduce weight if loss is high
                elif avg_loss < 0.1:
                    self.adaptive_weights[key] *= 1.05  # Increase weight if loss is low
                    
                # Clamp weights
                self.adaptive_weights[key] = np.clip(self.adaptive_weights[key], 0.01, 10.0)
    
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        # Update adaptive weights
        self.update_adaptive_weights(cur_nimg)
        
        # Call parent method with updated weights
        self.contrastive_weight = self.adaptive_weights['contrastive']
        self.perceptual_weight = self.adaptive_weights['perceptual']
        self.feature_matching_weight = self.adaptive_weights['feature_matching']
        
        super().accumulate_gradients(phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg)

#----------------------------------------------------------------------------

class ProgressiveLoss(StyleGAN4Loss):
    """Progressive loss that gradually increases complexity during training."""
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, 
                 pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, 
                 blur_init_sigma=0, blur_fade_kimg=0, contrastive_weight=0.1, 
                 feature_matching_weight=1.0, perceptual_weight=0.1):
        super().__init__(device, G, D, augment_pipe, r1_gamma, style_mixing_prob, 
                        pl_weight, pl_batch_shrink, pl_decay, pl_no_weight_grad, 
                        blur_init_sigma, blur_fade_kimg, contrastive_weight, 
                        feature_matching_weight, perceptual_weight)
        
        # Progressive training stages
        self.stages = [
            {'kimg': 1000, 'contrastive': 0.0, 'perceptual': 0.0, 'feature_matching': 0.0},
            {'kimg': 5000, 'contrastive': 0.05, 'perceptual': 0.05, 'feature_matching': 0.5},
            {'kimg': 10000, 'contrastive': 0.1, 'perceptual': 0.1, 'feature_matching': 1.0},
            {'kimg': float('inf'), 'contrastive': 0.1, 'perceptual': 0.1, 'feature_matching': 1.0}
        ]
        
    def get_current_stage(self, cur_nimg):
        """Get current training stage based on number of images processed."""
        cur_kimg = cur_nimg / 1000
        for stage in self.stages:
            if cur_kimg <= stage['kimg']:
                return stage
        return self.stages[-1]
    
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        # Get current stage
        stage = self.get_current_stage(cur_nimg)
        
        # Update weights based on current stage
        self.contrastive_weight = stage['contrastive']
        self.perceptual_weight = stage['perceptual']
        self.feature_matching_weight = stage['feature_matching']
        
        super().accumulate_gradients(phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg) 