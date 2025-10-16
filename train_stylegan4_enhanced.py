#!/usr/bin/env python3

"""Enhanced StyleGAN4 training script with high-resolution support and memory optimizations."""

import os
import click
import re
import json
import tempfile
import warnings

import torch
import dnnlib
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)
        torch.distributed.barrier()

    # Init torch_utils.
    sync_file = os.path.abspath(os.path.join(temp_dir, '.torch_sync_file'))
    custom_ops.init_all()

    # Print network summary.
    if rank == 0:
        with dnnlib.util.open_url(c.training_set_kwargs.url) as f:
            data = json.load(f)
        def dataset_summary(data):
            size = data['resolution']
            channels = data['channels']
            duration = '%.1f' % (data['duration_frame'] / data['fps'] / 60)
            return f'{size}x{size}x{channels}, {duration}min'
        click.echo('Dataset:')
        click.echo(f'  {data["dataset_name"]}: {dataset_summary(data)}')
        click.echo(f'  Path: {c.training_set_kwargs.path}')
        click.echo(f'  Mirror: {c.training_set_kwargs.use_labels}')
        click.echo()

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(c.run_dir):
        prev_run_dirs = [x for x in os.listdir(c.run_dir) if os.path.isdir(os.path.join(c.run_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(c.run_dir, f'{cur_run_id:05d}-{c.desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    if rank == 0:
        print('Training options:')
        print(f'  Output directory:    {c.run_dir}')
        print(f'  Training data:       {c.training_set_kwargs.path}')
        print(f'  Training duration:   {c.total_kimg} kimg')
        print(f'  Number of GPUs:      {c.num_gpus}')
        print(f'  Number of images:    {c.minibatch_size} per GPU')
        print(f'  Image resolution:    {c.resolution}x{c.resolution}')
        print(f'  Conditioning labels: {c.training_set_kwargs.use_labels}')
        print()

    # Create output directory.
    if rank == 0:
        os.makedirs(c.run_dir)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    # Launch processes.
    if rank == 0:
        print('Launching processes...')
    torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        dataset_kwargs.resolution = dataset_obj.resolution
        dataset_kwargs.use_labels = dataset_obj.has_labels
        dataset_kwargs.max_size = len(dataset_obj)
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--cfg', help='Base configuration', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), default='stylegan3-t', show_default=True)
@click.option('--data', help='Training dataset (required)', required=True, metavar='ZIP|DIR')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--batch', help='Total batch size', type=int, default=32, metavar='INT', show_default=True)
@click.option('--batch-gpu', help='Number of samples processed at a time by one GPU', type=int, default=4, metavar='INT', show_default=True)
@click.option('--epochs', help='Training duration in epochs', type=int, default=200, metavar='INT', show_default=True)
@click.option('--res', help='Image resolution for training', type=int, default=1024, metavar='INT', show_default=True)

# StyleGAN4 Enhanced options.
@click.option('--use-stylegan4-enhanced', help='Use enhanced StyleGAN4 with high-resolution support', is_flag=True, default=False)
@click.option('--use-multi-scale', help='Enable multi-scale generation', is_flag=True, default=False)
@click.option('--use-checkpointing', help='Enable gradient checkpointing for memory efficiency', is_flag=True, default=True)
@click.option('--target-scales', help='Target scales for multi-scale generation (comma-separated)', type=str, default='1,2,4,8', metavar='LIST')
@click.option('--window-size', help='Window size for efficient attention', type=int, default=64, metavar='INT', show_default=True)
@click.option('--high-res', help='Enable high-resolution optimizations (8K/16K)', is_flag=True, default=False)

# Architecture options.
@click.option('--use-attention', help='Use self-attention mechanisms', is_flag=True, default=True)
@click.option('--use-residual', help='Use residual connections', is_flag=True, default=True)
@click.option('--use-multi-scale-d', help='Use multi-scale discriminator', is_flag=True, default=False)

# Loss options.
@click.option('--use-enhanced-loss', help='Use enhanced loss functions', is_flag=True, default=False)
@click.option('--contrastive-weight', help='Weight for contrastive loss', type=float, default=0.1, metavar='FLOAT', show_default=True)
@click.option('--perceptual-weight', help='Weight for perceptual loss', type=float, default=0.1, metavar='FLOAT', show_default=True)
@click.option('--feature-matching-weight', help='Weight for feature matching loss', type=float, default=0.1, metavar='FLOAT', show_default=True)

# Augmentation options.
@click.option('--use-advanced-aug', help='Use advanced data augmentation', is_flag=True, default=False)
@click.option('--use-adaptive-aug', help='Use adaptive augmentation', is_flag=True, default=False)
@click.option('--use-progressive-aug', help='Use progressive augmentation', is_flag=True, default=False)

# Training options.
@click.option('--progressive-training', help='Use progressive training strategy', is_flag=True, default=False)
@click.option('--adaptive-training', help='Use adaptive training strategies', is_flag=True, default=False)

# Misc options.
@click.option('--desc', help='String to include in result dir name', metavar='STR', default='')
@click.option('--metrics', help='Quality metrics', type=parse_comma_separated_list, default=['fid50k_full'], show_default=True)
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--tick', help='How often to print progress', type=int, default=4, metavar='INT', show_default=True)
@click.option('--snap', help='How often to save snapshots', type=int, default=50, metavar='INT', show_default=True)
@click.option('--seed', help='Random seed', type=int, default=0, metavar='INT', show_default=True)
@click.option('--fp16', help='Enable mixed-precision training', is_flag=True, default=False)
@click.option('--nobench', help='Disable cuDNN benchmarking', is_flag=True, default=False)
@click.option('--workers', help='DataLoader worker processes', type=int, default=3, metavar='INT', show_default=True)

def main(**kwargs):
    """Train StyleGAN4 Enhanced using the given dataset.

    Examples:

    \b
    # Train StyleGAN4 Enhanced on custom dataset using 8 GPUs.
    python train_stylegan4_enhanced.py --outdir=~/training-runs --data=~/datasets/custom --gpus=8 --res=1024 --use-stylegan4-enhanced

    \b
    # Train with multi-scale generation and high-resolution support.
    python train_stylegan4_enhanced.py --outdir=~/training-runs --data=~/datasets/custom --gpus=8 --res=2048 --use-stylegan4-enhanced --use-multi-scale --high-res

    \b
    # Train with memory optimizations for large models.
    python train_stylegan4_enhanced.py --outdir=~/training-runs --data=~/datasets/custom --gpus=4 --res=4096 --use-stylegan4-enhanced --use-checkpointing --batch-gpu=2
    """
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict(class_name='training.training_loop.TrainingLoop') # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name=None, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.metrics = opts.metrics
    c.total_kimg = opts.epochs * 1000
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu
    c.emacopy = None
    c.G_reg_interval = None
    c.G_reg_interval_frac = None
    c.D_reg_interval = None
    c.D_reg_interval_frac = None
    c.mbstd_group_size = None
    c.cudnn_benchmark = not opts.nobench
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)

    # Validate and set up dataset.
    dataset_kwargs, dataset_name = init_dataset_kwargs(opts.data)
    if opts.cfg == 'stylegan2' and dataset_kwargs.resolution != 512:
        raise click.ClickException(f'--cfg=stylegan2 requires --res=512, got {opts.res}')
    if opts.cfg == 'stylegan3-t' and dataset_kwargs.resolution != 512:
        raise click.ClickException(f'--cfg=stylegan3-t requires --res=512, got {opts.res}')

    # Set up network architectures.
    if opts.use_stylegan4_enhanced:
        # Import enhanced StyleGAN4 modules
        try:
            from training.networks_stylegan4_enhanced import HighResolutionGenerator, MemoryEfficientDiscriminator
            from training.loss_stylegan4 import StyleGAN4Loss, AdaptiveLoss, ProgressiveLoss
            from training.augment_stylegan4 import StyleGAN4AugmentPipe, AdaptiveAugmentPipe, ProgressiveAugmentPipe
        except ImportError:
            click.echo("Warning: Enhanced StyleGAN4 modules not found, falling back to standard StyleGAN4")
            from training.networks_stylegan4 import Generator, MultiScaleDiscriminator
            from training.loss_stylegan4 import StyleGAN4Loss
            from training.augment_stylegan4 import StyleGAN4AugmentPipe
            
        # Set up generator
        if opts.high_res:
            c.G_kwargs.class_name = 'training.networks_stylegan4_enhanced.HighResolutionGenerator'
            c.G_kwargs.use_multi_scale = opts.use_multi_scale
            c.G_kwargs.use_checkpointing = opts.use_checkpointing
            c.G_kwargs.target_scales = [int(s) for s in opts.target_scales.split(',')]
        else:
            c.G_kwargs.class_name = 'training.networks_stylegan4.Generator'
            
        # Set up discriminator
        if opts.use_multi_scale_d:
            c.D_kwargs.class_name = 'training.networks_stylegan4_enhanced.MemoryEfficientDiscriminator'
            c.D_kwargs.use_checkpointing = opts.use_checkpointing
        else:
            c.D_kwargs.class_name = 'training.networks_stylegan4.MultiScaleDiscriminator'
            
        # Set up loss
        if opts.use_enhanced_loss:
            if opts.adaptive_training:
                c.loss_kwargs.class_name = 'training.loss_stylegan4.AdaptiveLoss'
            elif opts.progressive_training:
                c.loss_kwargs.class_name = 'training.loss_stylegan4.ProgressiveLoss'
            else:
                c.loss_kwargs.class_name = 'training.loss_stylegan4.StyleGAN4Loss'
            c.loss_kwargs.contrastive_weight = opts.contrastive_weight
            c.loss_kwargs.perceptual_weight = opts.perceptual_weight
            c.loss_kwargs.feature_matching_weight = opts.feature_matching_weight
        else:
            c.loss_kwargs.class_name = 'training.loss.StyleGAN2Loss'
            
        # Set up augmentation
        if opts.use_advanced_aug:
            if opts.adaptive_aug:
                c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment_stylegan4.AdaptiveAugmentPipe')
            elif opts.progressive_aug:
                c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment_stylegan4.ProgressiveAugmentPipe')
            else:
                c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment_stylegan4.StyleGAN4AugmentPipe')
        else:
            c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
    else:
        # Standard StyleGAN3/2 setup
        if opts.cfg == 'stylegan3-t':
            c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
            c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
            if opts.cfg == 'stylegan3-t':
                c.G_kwargs.conv_kernel = 1
                c.G_kwargs.channel_base = 32768
                c.G_kwargs.channel_max = 512
        elif opts.cfg == 'stylegan3-r':
            c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
            c.G_kwargs.conv_kernel = 3
            c.G_kwargs.channel_base = 32768
            c.G_kwargs.channel_max = 512
        else:
            c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
            c.G_kwargs.fused_modconv_default = 'inference_only'
            c.loss_kwargs.gamma = 10
        c.D_kwargs.class_name = 'training.networks_stylegan2.Discriminator'
        c.G_reg_interval = 4
        c.G_reg_interval_frac = 0.002
        c.D_reg_interval = 16
        c.D_reg_interval_frac = 0.002
        c.mbstd_group_size = 4
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)

    # Set up network options.
    c.G_kwargs.channel_base = c.G_kwargs.get('channel_base', 32768)
    c.G_kwargs.channel_max = c.G_kwargs.get('channel_max', 512)
    c.G_kwargs.magnitude_ema_beta = c.G_kwargs.get('magnitude_ema_beta', 0.5 ** (c.batch_size / (20 * 1e3)))
    c.G_kwargs.use_attention = opts.use_attention
    c.G_kwargs.use_residual = opts.use_residual
    c.D_kwargs.channel_base = c.D_kwargs.get('channel_base', 32768)
    c.D_kwargs.channel_max = c.D_kwargs.get('channel_max', 512)
    c.D_kwargs.use_attention = opts.use_attention
    c.D_kwargs.use_residual = opts.use_residual

    # Set up training options.
    c.total_kimg = opts.kimg or c.total_kimg
    c.tick = opts.tick
    c.snap = opts.snap
    c.seed = opts.seed
    c.fp16 = opts.fp16
    c.num_gpus = opts.gpus
    c.minibatch_size = opts.batch_gpu
    c.image_snapshot_ticks = c.network_snapshot_ticks = 200
    c.random_seed = c.seed
    c.training_set_kwargs = dataset_kwargs
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)

    # Set up logging.
    c.run_dir = opts.outdir
    c.desc = opts.desc

    # Print training configuration.
    if opts.use_stylegan4_enhanced:
        print('StyleGAN4 Enhanced Configuration:')
        print(f'  High-resolution support: {opts.high_res}')
        print(f'  Multi-scale generation: {opts.use_multi_scale}')
        print(f'  Gradient checkpointing: {opts.use_checkpointing}')
        print(f'  Target scales: {opts.target_scales}')
        print(f'  Window size: {opts.window_size}')
        print(f'  Enhanced loss: {opts.use_enhanced_loss}')
        print(f'  Advanced augmentation: {opts.use_advanced_aug}')
        print()

    # Launch training.
    print('Launching training...')
    launch_training(c=c, desc=c.desc, outdir=c.run_dir, dry_run=False)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print('Training options:')
    print(f'  Output directory:    {c.run_dir}')
    print(f'  Training data:       {c.training_set_kwargs.path}')
    print(f'  Training duration:   {c.total_kimg} kimg')
    print(f'  Number of GPUs:      {c.num_gpus}')
    print(f'  Number of images:    {c.minibatch_size} per GPU')
    print(f'  Image resolution:    {c.training_set_kwargs.resolution}x{c.training_set_kwargs.resolution}')
    print(f'  Conditioning labels: {c.training_set_kwargs.use_labels}')
    print()

    # Create output directory.
    if not dry_run:
        os.makedirs(c.run_dir)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    # Launch processes.
    if dry_run:
        print('Dry run; not actually launching training.')
    else:
        print('Launching processes...')
        torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, tempfile.mkdtemp()), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter 