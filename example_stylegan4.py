#!/usr/bin/env python3
"""
StyleGAN4 Example Script
Demonstrates the major improvements over StyleGAN3
"""

import os
import torch
import numpy as np
import dnnlib
from training.networks_stylegan4 import Generator, MultiScaleDiscriminator
from training.loss_stylegan4 import StyleGAN4Loss
from training.augment_stylegan4 import StyleGAN4AugmentPipe

def create_stylegan4_models(img_resolution=256, batch_size=4):
    """Create StyleGAN4 generator and discriminator models."""
    
    print("Creating StyleGAN4 models...")
    
    # Generator configuration
    G_kwargs = {
        'z_dim': 512,
        'c_dim': 0,
        'w_dim': 512,
        'img_resolution': img_resolution,
        'img_channels': 3,
        'use_attention': True,
        'use_residual': True,
        'channel_base': 32768,
        'channel_max': 512,
    }
    
    # Discriminator configuration
    D_kwargs = {
        'c_dim': 0,
        'img_resolution': img_resolution,
        'img_channels': 3,
        'use_attention': True,
        'use_residual': True,
        'num_scales': 3,
        'channel_base': 32768,
        'channel_max': 512,
    }
    
    # Create models
    G = Generator(**G_kwargs)
    D = MultiScaleDiscriminator(**D_kwargs)
    
    print(f"Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")
    
    return G, D

def create_enhanced_loss(G, D, device='cuda'):
    """Create enhanced loss function with all StyleGAN4 improvements."""
    
    print("Creating enhanced loss function...")
    
    # Move models to device
    G = G.to(device)
    D = D.to(device)
    
    # Create augmentation pipeline
    augment_pipe = StyleGAN4AugmentPipe(
        use_advanced_aug=True,
        use_style_mixing=True,
        brightness=0.1,
        contrast=0.1,
        hue=0.1,
        saturation=0.1,
        noise=0.05,
        cutout=0.1
    ).to(device)
    
    # Create enhanced loss
    loss = StyleGAN4Loss(
        device=device,
        G=G,
        D=D,
        augment_pipe=augment_pipe,
        r1_gamma=10,
        style_mixing_prob=0.9,
        pl_weight=2.0,
        contrastive_weight=0.1,
        perceptual_weight=0.1,
        feature_matching_weight=1.0
    )
    
    return loss, augment_pipe

def demonstrate_attention_mechanism():
    """Demonstrate the self-attention mechanism."""
    
    print("\n=== Demonstrating Self-Attention Mechanism ===")
    
    from training.networks_stylegan4 import SelfAttention
    
    # Create attention module
    attention = SelfAttention(in_channels=64, reduction=8)
    
    # Create sample input
    batch_size, channels, height, width = 2, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Apply attention
    output = attention(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention gamma parameter: {attention.gamma.item():.4f}")
    
    return output

def demonstrate_residual_connections():
    """Demonstrate residual connections."""
    
    print("\n=== Demonstrating Residual Connections ===")
    
    from training.networks_stylegan4 import ResidualBlock
    
    # Create residual block
    residual_block = ResidualBlock(
        in_channels=64,
        out_channels=128,
        w_dim=512,
        use_attention=True
    )
    
    # Create sample input
    batch_size, channels, height, width = 2, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    w = torch.randn(batch_size, 512)  # Style vector
    
    print(f"Input shape: {x.shape}")
    print(f"Style vector shape: {w.shape}")
    
    # Apply residual block
    output = residual_block(x, w)
    
    print(f"Output shape: {output.shape}")
    
    return output

def demonstrate_multi_scale_discriminator():
    """Demonstrate multi-scale discriminator."""
    
    print("\n=== Demonstrating Multi-Scale Discriminator ===")
    
    # Create discriminator
    D = MultiScaleDiscriminator(
        c_dim=0,
        img_resolution=256,
        img_channels=3,
        num_scales=3,
        use_attention=True,
        use_residual=True
    )
    
    # Create sample input
    batch_size, channels, height, width = 2, 3, 256, 256
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = D(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of discriminator scales: {len(D.discriminators)}")
    
    return output

def demonstrate_enhanced_augmentation():
    """Demonstrate enhanced augmentation pipeline."""
    
    print("\n=== Demonstrating Enhanced Augmentation ===")
    
    # Create augmentation pipeline
    augment_pipe = StyleGAN4AugmentPipe(
        use_advanced_aug=True,
        brightness=0.2,
        contrast=0.2,
        hue=0.1,
        saturation=0.1,
        noise=0.05,
        cutout=0.1,
        scale=0.1,
        rotate=10
    )
    
    # Create sample input
    batch_size, channels, height, width = 4, 3, 256, 256
    images = torch.randn(batch_size, channels, height, width)
    
    print(f"Original images shape: {images.shape}")
    print(f"Original images range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Apply augmentation
    augmented_images = augment_pipe(images)
    
    print(f"Augmented images shape: {augmented_images.shape}")
    print(f"Augmented images range: [{augmented_images.min():.3f}, {augmented_images.max():.3f}]")
    
    return augmented_images

def demonstrate_contrastive_loss():
    """Demonstrate contrastive loss."""
    
    print("\n=== Demonstrating Contrastive Loss ===")
    
    from training.loss_stylegan4 import ContrastiveLoss
    
    # Create contrastive loss
    contrastive_loss = ContrastiveLoss(temperature=0.07, queue_size=8192)
    
    # Create sample features
    batch_size, feature_dim = 8, 512
    features = torch.randn(batch_size, feature_dim)
    
    print(f"Features shape: {features.shape}")
    
    # Compute contrastive loss
    loss = contrastive_loss(features)
    
    print(f"Contrastive loss: {loss.item():.4f}")
    
    return loss

def compare_with_stylegan3():
    """Compare StyleGAN4 with StyleGAN3 architecture."""
    
    print("\n=== Comparing StyleGAN4 vs StyleGAN3 ===")
    
    # StyleGAN4 models
    G4, D4 = create_stylegan4_models(img_resolution=256)
    
    # StyleGAN3 models (import from original)
    try:
        from training.networks_stylegan3 import Generator as G3
        from training.networks_stylegan2 import Discriminator as D3
        
        G3_model = G3(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=256,
            img_channels=3
        )
        
        D3_model = D3(
            c_dim=0,
            img_resolution=256,
            img_channels=3
        )
        
        print(f"StyleGAN4 Generator parameters: {sum(p.numel() for p in G4.parameters()):,}")
        print(f"StyleGAN3 Generator parameters: {sum(p.numel() for p in G3_model.parameters()):,}")
        print(f"StyleGAN4 Discriminator parameters: {sum(p.numel() for p in D4.parameters()):,}")
        print(f"StyleGAN3 Discriminator parameters: {sum(p.numel() for p in D3_model.parameters()):,}")
        
        # Calculate parameter increase
        g4_params = sum(p.numel() for p in G4.parameters())
        g3_params = sum(p.numel() for p in G3_model.parameters())
        d4_params = sum(p.numel() for p in D4.parameters())
        d3_params = sum(p.numel() for p in D3_model.parameters())
        
        print(f"\nParameter increase:")
        print(f"Generator: {((g4_params - g3_params) / g3_params * 100):.1f}%")
        print(f"Discriminator: {((d4_params - d3_params) / d3_params * 100):.1f}%")
        
    except ImportError:
        print("StyleGAN3 models not available for comparison")

def main():
    """Main demonstration function."""
    
    print("StyleGAN4 Demonstration")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Demonstrate various components
    demonstrate_attention_mechanism()
    demonstrate_residual_connections()
    demonstrate_multi_scale_discriminator()
    demonstrate_enhanced_augmentation()
    demonstrate_contrastive_loss()
    compare_with_stylegan3()
    
    # Create full models
    print("\n=== Creating Full StyleGAN4 Models ===")
    G, D = create_stylegan4_models(img_resolution=256)
    
    # Create enhanced loss
    loss, augment_pipe = create_enhanced_loss(G, D, device)
    
    print("\nStyleGAN4 demonstration completed!")
    print("\nKey improvements demonstrated:")
    print("✅ Self-attention mechanisms")
    print("✅ Residual connections")
    print("✅ Multi-scale discriminator")
    print("✅ Enhanced augmentation")
    print("✅ Contrastive learning")
    print("✅ Improved loss functions")

if __name__ == "__main__":
    main() 