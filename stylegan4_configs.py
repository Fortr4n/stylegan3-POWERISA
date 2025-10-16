#!/usr/bin/env python3
"""
StyleGAN4 Configuration Presets
Recommended settings for different use cases and scenarios
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class StyleGAN4Config:
    """Configuration preset for StyleGAN4 training."""
    
    # Basic settings
    name: str
    description: str
    
    # Architecture settings
    use_stylegan4: bool = True
    use_attention: bool = True
    use_residual: bool = True
    use_multi_scale_d: bool = True
    
    # Loss function settings
    use_enhanced_loss: bool = True
    contrastive_weight: float = 0.1
    perceptual_weight: float = 0.1
    feature_matching_weight: float = 1.0
    
    # Training settings
    progressive_training: bool = True
    adaptive_training: bool = True
    use_advanced_aug: bool = True
    
    # Hyperparameters
    batch_size: int = 32
    gamma: float = 8.2
    learning_rate: float = 0.002
    total_kimg: int = 25000
    
    # Resolution and dataset specific
    img_resolution: int = 512
    channel_base: int = 32768
    channel_max: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for command line arguments."""
        return {
            'use_stylegan4': self.use_stylegan4,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual,
            'use_multi_scale_d': self.use_multi_scale_d,
            'use_enhanced_loss': self.use_enhanced_loss,
            'contrastive_weight': self.contrastive_weight,
            'perceptual_weight': self.perceptual_weight,
            'feature_matching_weight': self.feature_matching_weight,
            'progressive_training': self.progressive_training,
            'adaptive_training': self.adaptive_training,
            'use_advanced_aug': self.use_advanced_aug,
            'batch': self.batch_size,
            'gamma': self.gamma,
            'glr': self.learning_rate,
            'dlr': self.learning_rate,
            'kimg': self.total_kimg,
            'resolution': self.img_resolution,
            'cbase': self.channel_base,
            'cmax': self.channel_max,
        }

# Configuration presets
CONFIGS = {
    # High-quality image generation
    'stylegan4-hq': StyleGAN4Config(
        name='StyleGAN4-HQ',
        description='High-quality image generation with all features enabled',
        use_stylegan4=True,
        use_attention=True,
        use_residual=True,
        use_multi_scale_d=True,
        use_enhanced_loss=True,
        contrastive_weight=0.1,
        perceptual_weight=0.1,
        feature_matching_weight=1.0,
        progressive_training=True,
        adaptive_training=True,
        use_advanced_aug=True,
        batch_size=32,
        gamma=32.0,
        learning_rate=0.002,
        total_kimg=25000,
        img_resolution=1024,
        channel_base=32768,
        channel_max=512,
    ),
    
    # Fast training for experimentation
    'stylegan4-fast': StyleGAN4Config(
        name='StyleGAN4-Fast',
        description='Fast training for experimentation and prototyping',
        use_stylegan4=True,
        use_attention=False,  # Disable for speed
        use_residual=True,
        use_multi_scale_d=False,  # Disable for speed
        use_enhanced_loss=True,
        contrastive_weight=0.05,  # Reduced for speed
        perceptual_weight=0.05,
        feature_matching_weight=0.5,
        progressive_training=True,
        adaptive_training=False,  # Disable for speed
        use_advanced_aug=True,
        batch_size=16,
        gamma=8.2,
        learning_rate=0.002,
        total_kimg=10000,
        img_resolution=256,
        channel_base=16384,  # Reduced for speed
        channel_max=256,
    ),
    
    # Memory-efficient training
    'stylegan4-memory': StyleGAN4Config(
        name='StyleGAN4-Memory',
        description='Memory-efficient training for limited GPU resources',
        use_stylegan4=True,
        use_attention=False,
        use_residual=True,
        use_multi_scale_d=False,
        use_enhanced_loss=True,
        contrastive_weight=0.05,
        perceptual_weight=0.05,
        feature_matching_weight=0.5,
        progressive_training=True,
        adaptive_training=False,
        use_advanced_aug=False,  # Disable for memory
        batch_size=8,
        gamma=8.2,
        learning_rate=0.002,
        total_kimg=15000,
        img_resolution=512,
        channel_base=16384,
        channel_max=256,
    ),
    
    # Transfer learning
    'stylegan4-transfer': StyleGAN4Config(
        name='StyleGAN4-Transfer',
        description='Optimized for transfer learning from pre-trained models',
        use_stylegan4=True,
        use_attention=True,
        use_residual=True,
        use_multi_scale_d=True,
        use_enhanced_loss=True,
        contrastive_weight=0.05,  # Reduced for transfer
        perceptual_weight=0.2,    # Increased for transfer
        feature_matching_weight=1.0,
        progressive_training=False,  # Disable for transfer
        adaptive_training=True,
        use_advanced_aug=True,
        batch_size=16,
        gamma=6.6,
        learning_rate=0.001,  # Reduced for transfer
        total_kimg=5000,
        img_resolution=1024,
        channel_base=32768,
        channel_max=512,
    ),
    
    # Research/experimentation
    'stylegan4-research': StyleGAN4Config(
        name='StyleGAN4-Research',
        description='Full-featured configuration for research and experimentation',
        use_stylegan4=True,
        use_attention=True,
        use_residual=True,
        use_multi_scale_d=True,
        use_enhanced_loss=True,
        contrastive_weight=0.1,
        perceptual_weight=0.1,
        feature_matching_weight=1.0,
        progressive_training=True,
        adaptive_training=True,
        use_advanced_aug=True,
        batch_size=24,
        gamma=16.4,
        learning_rate=0.002,
        total_kimg=20000,
        img_resolution=512,
        channel_base=32768,
        channel_max=512,
    ),
    
    # Minimal StyleGAN4 (backward compatibility)
    'stylegan4-minimal': StyleGAN4Config(
        name='StyleGAN4-Minimal',
        description='Minimal StyleGAN4 with only essential improvements',
        use_stylegan4=True,
        use_attention=False,
        use_residual=True,
        use_multi_scale_d=False,
        use_enhanced_loss=True,
        contrastive_weight=0.0,  # Disabled
        perceptual_weight=0.0,   # Disabled
        feature_matching_weight=0.5,
        progressive_training=False,
        adaptive_training=False,
        use_advanced_aug=False,
        batch_size=32,
        gamma=8.2,
        learning_rate=0.002,
        total_kimg=25000,
        img_resolution=512,
        channel_base=32768,
        channel_max=512,
    ),
}

def get_config(config_name: str) -> StyleGAN4Config:
    """Get configuration preset by name."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]

def list_configs() -> None:
    """List all available configuration presets."""
    print("Available StyleGAN4 Configuration Presets:")
    print("=" * 60)
    
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config.description}")
        print(f"  Resolution: {config.img_resolution}x{config.img_resolution}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Total kimg: {config.total_kimg}")
        print(f"  Features:")
        print(f"    - Attention: {'✅' if config.use_attention else '❌'}")
        print(f"    - Residual: {'✅' if config.use_residual else '❌'}")
        print(f"    - Multi-scale D: {'✅' if config.use_multi_scale_d else '❌'}")
        print(f"    - Enhanced loss: {'✅' if config.use_enhanced_loss else '❌'}")
        print(f"    - Progressive: {'✅' if config.progressive_training else '❌'}")
        print(f"    - Adaptive: {'✅' if config.adaptive_training else '❌'}")
        print(f"    - Advanced aug: {'✅' if config.use_advanced_aug else '❌'}")

def generate_command(config_name: str, data_path: str, outdir: str, gpus: int = 8) -> str:
    """Generate training command for a specific configuration."""
    config = get_config(config_name)
    args = config.to_dict()
    
    cmd = f"python train_stylegan4.py --outdir={outdir} --cfg=stylegan4 --data={data_path} --gpus={gpus}"
    
    # Add all configuration arguments
    for key, value in args.items():
        if isinstance(value, bool):
            cmd += f" --{key}={str(value).lower()}"
        else:
            cmd += f" --{key}={value}"
    
    return cmd

def get_recommended_config(dataset_size: int, gpu_memory: int, resolution: int) -> str:
    """Get recommended configuration based on hardware and dataset."""
    
    if gpu_memory < 8:
        return 'stylegan4-memory'
    elif dataset_size < 10000:
        return 'stylegan4-transfer'
    elif resolution <= 256:
        return 'stylegan4-fast'
    elif resolution >= 1024:
        return 'stylegan4-hq'
    else:
        return 'stylegan4-research'

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='StyleGAN4 Configuration Management')
    parser.add_argument('--list', action='store_true', help='List all available configurations')
    parser.add_argument('--config', type=str, help='Configuration name')
    parser.add_argument('--data', type=str, help='Dataset path')
    parser.add_argument('--outdir', type=str, help='Output directory')
    parser.add_argument('--gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--recommend', action='store_true', help='Get recommended configuration')
    parser.add_argument('--dataset-size', type=int, help='Dataset size for recommendation')
    parser.add_argument('--gpu-memory', type=int, help='GPU memory in GB for recommendation')
    parser.add_argument('--resolution', type=int, help='Image resolution for recommendation')
    
    args = parser.parse_args()
    
    if args.list:
        list_configs()
    elif args.recommend:
        if not all([args.dataset_size, args.gpu_memory, args.resolution]):
            print("Error: --dataset-size, --gpu-memory, and --resolution are required for recommendation")
            exit(1)
        
        recommended = get_recommended_config(args.dataset_size, args.gpu_memory, args.resolution)
        print(f"Recommended configuration: {recommended}")
        config = get_config(recommended)
        print(f"Description: {config.description}")
    elif args.config and args.data and args.outdir:
        cmd = generate_command(args.config, args.data, args.outdir, args.gpus)
        print("Generated training command:")
        print(cmd)
    else:
        print("Usage examples:")
        print("  python stylegan4_configs.py --list")
        print("  python stylegan4_configs.py --recommend --dataset-size 50000 --gpu-memory 24 --resolution 512")
        print("  python stylegan4_configs.py --config stylegan4-hq --data /path/to/dataset --outdir /path/to/output") 