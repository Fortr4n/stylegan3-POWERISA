#!/usr/bin/env python3

"""Demonstration script for StyleGAN4 Enhanced features."""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import click
from PIL import Image
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_high_resolution_support():
    """Demonstrate high-resolution generation capabilities."""
    print("=== High-Resolution Support Demonstration ===")
    
    try:
        from training.networks_stylegan4_enhanced import HighResolutionGenerator, MemoryEfficientDiscriminator
        from training.networks_stylegan4_enhanced import EfficientSelfAttention, MemoryEfficientResidualBlock
        
        # Test different resolutions
        resolutions = [1024, 2048, 4096, 8192]
        
        for res in resolutions:
            print(f"\nTesting resolution: {res}x{res}")
            
            # Create generator
            generator = HighResolutionGenerator(
                z_dim=512,
                c_dim=0,
                w_dim=512,
                img_resolution=res,
                img_channels=3,
                use_multi_scale=True,
                use_checkpointing=True,
                target_scales=[1, 2, 4, 8]
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in generator.parameters())
            trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
            
            print(f"  Generator parameters: {total_params:,} (trainable: {trainable_params:,})")
            
            # Test memory usage
            if torch.cuda.is_available():
                device = torch.device('cuda')
                generator = generator.to(device)
                
                # Test with small batch to avoid OOM
                batch_size = 1 if res >= 4096 else 2
                z = torch.randn(batch_size, 512).to(device)
                c = torch.zeros(batch_size, 0).to(device)
                
                try:
                    with torch.no_grad():
                        outputs = generator(z, c)
                    
                    if isinstance(outputs, dict):
                        print(f"  Multi-scale outputs generated:")
                        for scale_name, output in outputs.items():
                            print(f"    {scale_name}: {output.shape}")
                    else:
                        print(f"  Single output: {outputs.shape}")
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  Memory limit reached for {res}x{res}")
                    else:
                        print(f"  Error: {e}")
                        
                torch.cuda.empty_cache()
            else:
                print("  CUDA not available, skipping GPU test")
                
    except ImportError as e:
        print(f"Error importing enhanced modules: {e}")
        print("Make sure you have created the enhanced StyleGAN4 modules.")

def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency improvements."""
    print("\n=== Memory Efficiency Demonstration ===")
    
    try:
        from training.networks_stylegan4_enhanced import EfficientSelfAttention, MemoryEfficientResidualBlock
        
        # Test attention mechanisms
        print("\nTesting Efficient Self-Attention:")
        
        # Test with different window sizes
        window_sizes = [32, 64, 128]
        input_size = 512
        
        for window_size in window_sizes:
            attention = EfficientSelfAttention(
                in_channels=512,
                reduction=8,
                window_size=window_size
            )
            
            if torch.cuda.is_available():
                device = torch.device('cuda')
                attention = attention.to(device)
                
                # Test memory usage
                x = torch.randn(1, 512, input_size, input_size).to(device)
                
                try:
                    with torch.no_grad():
                        output = attention(x)
                    
                    print(f"  Window size {window_size}: Success")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  Window size {window_size}: Memory limit reached")
                    else:
                        print(f"  Window size {window_size}: Error - {e}")
                        
                torch.cuda.empty_cache()
            else:
                print(f"  Window size {window_size}: CUDA not available")
        
        # Test residual blocks with checkpointing
        print("\nTesting Memory-Efficient Residual Blocks:")
        
        residual_block = MemoryEfficientResidualBlock(
            in_channels=512,
            out_channels=512,
            w_dim=512,
            use_attention=True,
            use_checkpointing=True
        )
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            residual_block = residual_block.to(device)
            
            x = torch.randn(1, 512, 256, 256).to(device)
            w = torch.randn(1, 512).to(device)
            
            try:
                with torch.no_grad():
                    output = residual_block(x, w)
                
                print(f"  Residual block: Success - Output shape: {output.shape}")
                
            except RuntimeError as e:
                print(f"  Residual block: Error - {e}")
                
            torch.cuda.empty_cache()
        else:
            print("  Residual block: CUDA not available")
            
    except ImportError as e:
        print(f"Error importing enhanced modules: {e}")

def demonstrate_multi_scale_generation():
    """Demonstrate multi-scale generation capabilities."""
    print("\n=== Multi-Scale Generation Demonstration ===")
    
    try:
        from training.networks_stylegan4_enhanced import MultiScaleSynthesisNetwork
        
        # Create multi-scale synthesis network
        synthesis = MultiScaleSynthesisNetwork(
            w_dim=512,
            img_resolution=1024,
            img_channels=3,
            use_checkpointing=True,
            target_scales=[1, 2, 4, 8]
        )
        
        print(f"Target scales: {synthesis.target_scales}")
        print(f"Multi-scale outputs: {list(synthesis.multi_scale_outputs.keys())}")
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            synthesis = synthesis.to(device)
            
            # Create dummy W vectors
            batch_size = 1
            num_layers = 15  # Standard StyleGAN3
            ws = torch.randn(batch_size, num_layers, 512).to(device)
            
            try:
                with torch.no_grad():
                    outputs = synthesis(ws)
                
                print("Multi-scale outputs generated:")
                for scale_name, output in outputs.items():
                    print(f"  {scale_name}: {output.shape}")
                    
            except RuntimeError as e:
                print(f"Error generating multi-scale outputs: {e}")
                
            torch.cuda.empty_cache()
        else:
            print("CUDA not available, skipping GPU test")
            
    except ImportError as e:
        print(f"Error importing enhanced modules: {e}")

def demonstrate_performance_comparison():
    """Compare performance between standard and enhanced StyleGAN4."""
    print("\n=== Performance Comparison ===")
    
    try:
        # Import both standard and enhanced versions
        from training.networks_stylegan4 import Generator as StandardGenerator
        from training.networks_stylegan4_enhanced import HighResolutionGenerator as EnhancedGenerator
        
        resolutions = [1024, 2048]
        
        for res in resolutions:
            print(f"\nResolution: {res}x{res}")
            
            # Standard StyleGAN4
            try:
                standard_gen = StandardGenerator(
                    z_dim=512,
                    c_dim=0,
                    w_dim=512,
                    img_resolution=res,
                    img_channels=3
                )
                
                standard_params = sum(p.numel() for p in standard_gen.parameters())
                print(f"  Standard StyleGAN4 parameters: {standard_params:,}")
                
            except Exception as e:
                print(f"  Standard StyleGAN4: Error - {e}")
            
            # Enhanced StyleGAN4
            try:
                enhanced_gen = EnhancedGenerator(
                    z_dim=512,
                    c_dim=0,
                    w_dim=512,
                    img_resolution=res,
                    img_channels=3,
                    use_multi_scale=True,
                    use_checkpointing=True
                )
                
                enhanced_params = sum(p.numel() for p in enhanced_gen.parameters())
                print(f"  Enhanced StyleGAN4 parameters: {enhanced_params:,}")
                
                # Calculate parameter increase
                if 'standard_params' in locals():
                    increase = ((enhanced_params - standard_params) / standard_params) * 100
                    print(f"  Parameter increase: {increase:.1f}%")
                    
            except Exception as e:
                print(f"  Enhanced StyleGAN4: Error - {e}")
                
    except ImportError as e:
        print(f"Error importing modules: {e}")

def demonstrate_memory_usage():
    """Demonstrate memory usage optimizations."""
    print("\n=== Memory Usage Demonstration ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory tests")
        return
    
    try:
        from training.networks_stylegan4_enhanced import HighResolutionGenerator
        
        device = torch.device('cuda')
        
        # Test different configurations
        configs = [
            {"res": 1024, "checkpointing": False, "multi_scale": False},
            {"res": 1024, "checkpointing": True, "multi_scale": False},
            {"res": 1024, "checkpointing": True, "multi_scale": True},
            {"res": 2048, "checkpointing": True, "multi_scale": True},
        ]
        
        for config in configs:
            print(f"\nTesting config: {config}")
            
            try:
                generator = HighResolutionGenerator(
                    z_dim=512,
                    c_dim=0,
                    w_dim=512,
                    img_resolution=config["res"],
                    img_channels=3,
                    use_multi_scale=config["multi_scale"],
                    use_checkpointing=config["checkpointing"]
                ).to(device)
                
                # Measure memory before
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
                
                # Generate sample
                z = torch.randn(1, 512).to(device)
                c = torch.zeros(1, 0).to(device)
                
                with torch.no_grad():
                    outputs = generator(z, c)
                
                # Measure memory after
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024**3  # Convert to GB
                
                print(f"  Memory used: {memory_used:.2f} GB")
                
                if isinstance(outputs, dict):
                    print(f"  Outputs: {list(outputs.keys())}")
                else:
                    print(f"  Output shape: {outputs.shape}")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Out of memory")
                else:
                    print(f"  Error: {e}")
                    
            torch.cuda.empty_cache()
            
    except ImportError as e:
        print(f"Error importing enhanced modules: {e}")

def create_sample_images():
    """Create sample images using the enhanced StyleGAN4."""
    print("\n=== Sample Image Generation ===")
    
    try:
        from training.networks_stylegan4_enhanced import HighResolutionGenerator
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping image generation")
            return
        
        device = torch.device('cuda')
        
        # Create generator
        generator = HighResolutionGenerator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=1024,
            img_channels=3,
            use_multi_scale=True,
            use_checkpointing=True
        ).to(device)
        
        # Generate sample images
        num_images = 4
        z = torch.randn(num_images, 512).to(device)
        c = torch.zeros(num_images, 0).to(device)
        
        print(f"Generating {num_images} sample images...")
        
        with torch.no_grad():
            outputs = generator(z, c)
        
        # Save images
        os.makedirs('samples', exist_ok=True)
        
        if isinstance(outputs, dict):
            for scale_name, images in outputs.items():
                for i in range(num_images):
                    # Convert to PIL image
                    img = images[i].cpu().numpy()
                    img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
                    img = np.clip(img, 0, 1)
                    img = (img * 255).astype(np.uint8)
                    img = np.transpose(img, (1, 2, 0))
                    
                    # Save image
                    pil_img = Image.fromarray(img)
                    filename = f'samples/sample_{scale_name}_{i:02d}.png'
                    pil_img.save(filename)
                    print(f"  Saved: {filename}")
        else:
            for i in range(num_images):
                img = outputs[i].cpu().numpy()
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))
                
                pil_img = Image.fromarray(img)
                filename = f'samples/sample_{i:02d}.png'
                pil_img.save(filename)
                print(f"  Saved: {filename}")
                
    except ImportError as e:
        print(f"Error importing enhanced modules: {e}")
    except Exception as e:
        print(f"Error generating sample images: {e}")

@click.command()
@click.option('--demo', type=click.Choice(['all', 'high-res', 'memory', 'multi-scale', 'performance', 'samples']), 
              default='all', help='Which demonstration to run')
def main(demo):
    """Demonstrate StyleGAN4 Enhanced features."""
    
    print("StyleGAN4 Enhanced Feature Demonstrations")
    print("=" * 50)
    
    if demo == 'all' or demo == 'high-res':
        demonstrate_high_resolution_support()
    
    if demo == 'all' or demo == 'memory':
        demonstrate_memory_efficiency()
    
    if demo == 'all' or demo == 'multi-scale':
        demonstrate_multi_scale_generation()
    
    if demo == 'all' or demo == 'performance':
        demonstrate_performance_comparison()
    
    if demo == 'all' or demo == 'samples':
        create_sample_images()
    
    print("\nDemonstration completed!")

if __name__ == "__main__":
    main() 