#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""StyleGAN4 training script with major improvements."""

import os
import click
import re
import json
import tempfile
import warnings

import torch
import dnnlib
import legacy
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)
        torch.distributed.barrier()

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    dnnlib.util.Logger(should_flush=True)
    _training_loop(rank=rank, **args)

    # Done.
    if rank == 0 and args.num_gpus > 1:
        torch.distributed.destroy_process_group()

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, use_labels, max_size=None, xflip=False, random_seed=0):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=use_labels, max_size=max_size, xflip=xflip, random_seed=random_seed)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        dataset_kwargs.resolution = dataset_obj.resolution # be explicit about resolution
        dataset_kwargs.use_labels = dataset_obj.has_labels # be explicit about labels
        dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
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

def _training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Arguments for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Arguments for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_kimg            = 0,        # How long to use augmentations. 0 = disable.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from. None = no resume.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    use_stylegan4           = True,     # Use StyleGAN4 architecture?
    use_enhanced_loss       = True,     # Use enhanced loss functions?
    use_advanced_aug        = True,     # Use advanced augmentation?
    use_attention           = True,     # Use attention mechanisms?
    use_residual            = True,     # Use residual connections?
    use_multi_scale_d       = True,     # Use multi-scale discriminator?
    contrastive_weight      = 0.1,      # Weight for contrastive loss.
    perceptual_weight       = 0.1,      # Weight for perceptual loss.
    feature_matching_weight = 1.0,      # Weight for feature matching loss.
    progressive_training    = True,     # Use progressive training?
    adaptive_training       = True,     # Use adaptive training?
):
    # Initialize.
    start_epoch = 0
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                        # Improves training speed.
    grid_sample_gradfix.enabled = True                   # Avoids errors with the second derivative.

    # Load training set.
    if rank == 0:
        print(f'Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print(f'Num images: {len(training_set)}')
        print(f'Image shape: {training_set.image_shape}')
        print(f'Label shape: {training_set.label_shape}')
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    
    # Choose network architecture based on StyleGAN4 flag
    if use_stylegan4:
        from training.networks_stylegan4 import Generator, MultiScaleDiscriminator
        G_kwargs.update(use_attention=use_attention, use_residual=use_residual)
        D_kwargs.update(use_attention=use_attention, use_residual=use_residual, num_scales=3 if use_multi_scale_d else 1)
    else:
        from training.networks_stylegan3 import Generator
        from training.networks_stylegan2 import Discriminator
    
    common_kwargs = dict(class_name='training.networks_stylegan4.Generator' if use_stylegan4 else 'training.networks_stylegan3.Generator', 
                        w_dim=512, mapping_kwargs=dnnlib.EasyDict(), **G_kwargs)
    G = dnnlib.util.construct_class_by_name(**common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    
    if use_stylegan4 and use_multi_scale_d:
        D = MultiScaleDiscriminator(c_dim=training_set.label_dim, img_resolution=training_set.resolution, 
                                   img_channels=training_set.num_channels, **D_kwargs).train().requires_grad_(False).to(device)
    else:
        D_kwargs.update(class_name='training.networks_stylegan2.Discriminator')
        D = dnnlib.util.construct_class_by_name(**D_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    
    if use_advanced_aug:
        from training.augment_stylegan4 import StyleGAN4AugmentPipe, AdaptiveAugmentPipe, ProgressiveAugmentPipe
        if adaptive_training:
            augment_pipe = AdaptiveAugmentPipe(use_advanced_aug=True, use_style_mixing=True)
        elif progressive_training:
            augment_pipe = ProgressiveAugmentPipe(use_advanced_aug=True, use_style_mixing=True)
        else:
            augment_pipe = StyleGAN4AugmentPipe(use_advanced_aug=True, use_style_mixing=True)
    else:
        augment_pipe = None
        if augment_kwargs is not None:
            augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    
    # Choose loss function based on StyleGAN4 flag
    if use_enhanced_loss:
        from training.loss_stylegan4 import StyleGAN4Loss, AdaptiveLoss, ProgressiveLoss
        if adaptive_training:
            loss = AdaptiveLoss(device=device, G=G, D=D, augment_pipe=augment_pipe, 
                              contrastive_weight=contrastive_weight, perceptual_weight=perceptual_weight,
                              feature_matching_weight=feature_matching_weight, **loss_kwargs)
        elif progressive_training:
            loss = ProgressiveLoss(device=device, G=G, D=D, augment_pipe=augment_pipe,
                                 contrastive_weight=contrastive_weight, perceptual_weight=perceptual_weight,
                                 feature_matching_weight=feature_matching_weight, **loss_kwargs)
        else:
            loss = StyleGAN4Loss(device=device, G=G, D=D, augment_pipe=augment_pipe,
                                contrastive_weight=contrastive_weight, perceptual_weight=perceptual_weight,
                                feature_matching_weight=feature_matching_weight, **loss_kwargs)
    else:
        from training.loss import StyleGAN2Loss
        loss = StyleGAN2Loss(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs)

    # Setup optimizer.
    if rank == 0:
        print('Setting up optimizer...')
    args = dict(params=G.parameters(), lr=G_opt_kwargs.lr, betas=(0, 0.99), eps=1e-8)
    first_optimizer = dnnlib.util.construct_class_by_name(**args) # subclass of torch.optim.Optimizer
    args = dict(params=D.parameters(), lr=D_opt_kwargs.lr, betas=(0, 0.99), eps=1e-8)
    second_optimizer = dnnlib.util.construct_class_by_name(**args) # subclass of torch.optim.Optimizer

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard
            stats_tfevents = torch.utils.tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print(f'Skipping tfevents export on {err}: tensorboard not available')

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu) if phase_real_c is not None else None
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            with torch.autograd.profiler.record_function(phase.name):
                if phase.start_event is not None:
                    phase.start_event.record(torch.cuda.current_stream(device))

                # Accumulate gradients.
                phase.requires_grad_(True)
                for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
                    phase.interval -= 1

                # Update weights.
                if phase.interval == 0:
                    with torch.autograd.profiler.record_function(phase.name + '_opt'):
                        for param in phase.module.parameters():
                            if param.grad is not None:
                                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                        phase.optimizer.step()
                    phase.requires_grad_(False)
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line.
        tick_end_time = time.time()
        if rank == 0:
            print(f'tick {cur_nimg//1000:3d}k/ {total_kimg:3d}k  elapsed {dnnlib.util.format_time(tick_end_time - start_time)}  '
                  f'tick {dnnlib.util.format_time(tick_end_time - tick_start_time)}  '
                  f'maintenance {dnnlib.util.format_time(maintenance_time)}  '
                  f'cpumem {dnnlib.util.format_bytes(psutil.cpu_percent())}  '
                  f'gpumem {dnnlib.util.format_bytes(torch.cuda.max_memory_allocated(device))}  '
                  f'augment {augment_pipe.p.item():.3f}' if augment_pipe is not None else 'augment disabled')
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (not done) and (image_snapshot_ticks is not None) and (cur_tick % image_snapshot_ticks == 0):
            if rank == 0:
                images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        if (not done) and (network_snapshot_ticks is not None) and (cur_tick % network_snapshot_ticks == 0):
            if rank == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(dict(G=G, D=D, G_ema=G_ema, training_set_kwargs=dict(training_set_kwargs)), f)

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fieldnames = list(stats_dict.keys()) + ['timestamp']
            stats_jsonl.write(json.dumps(dict(stats_dict, timestamp=timestamp)) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
                stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Parse either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--cfg', help='Base configuration [default: look up]', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r', 'stylegan4']))
@click.option('--data', help='Training data (directory or zip)', required=True, metavar='PATH')
@click.option('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, default=False, metavar='BOOL')
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, default=False, metavar='BOOL')
@click.option('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--resume', help='Resume from given network pickle', metavar='[PATH|URL]')
@click.option('--freezed', help='Freeze first layers of D', type=int, default=0, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, default=50)
@click.option('--seed', help='Random seed [default: 0]', type=int, default=0, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
@click.option('--use-stylegan4', help='Use StyleGAN4 architecture', type=bool, default=True, metavar='BOOL')
@click.option('--use-enhanced-loss', help='Use enhanced loss functions', type=bool, default=True, metavar='BOOL')
@click.option('--use-advanced-aug', help='Use advanced augmentation', type=bool, default=True, metavar='BOOL')
@click.option('--use-attention', help='Use attention mechanisms', type=bool, default=True, metavar='BOOL')
@click.option('--use-residual', help='Use residual connections', type=bool, default=True, metavar='BOOL')
@click.option('--use-multi-scale-d', help='Use multi-scale discriminator', type=bool, default=True, metavar='BOOL')
@click.option('--contrastive-weight', help='Weight for contrastive loss', type=float, default=0.1, metavar='FLOAT')
@click.option('--perceptual-weight', help='Weight for perceptual loss', type=float, default=0.1, metavar='FLOAT')
@click.option('--feature-matching-weight', help='Weight for feature matching loss', type=float, default=1.0, metavar='FLOAT')
@click.option('--progressive-training', help='Use progressive training', type=bool, default=True, metavar='BOOL')
@click.option('--adaptive-training', help='Use adaptive training', type=bool, default=True, metavar='BOOL')
def main(ctx, outdir, cfg, data, cond, mirror, aug, resume, freezed, snap, seed, dry_run,
         use_stylegan4, use_enhanced_loss, use_advanced_aug, use_attention, use_residual,
         use_multi_scale_d, contrastive_weight, perceptual_weight, feature_matching_weight,
         progressive_training, adaptive_training):
    """Train StyleGAN4 using the techniques described in the paper.

    Examples:

    \b
    # Train StyleGAN4-T for AFHQv2 using 8 GPUs.
    python train_stylegan4.py --outdir=~/training-runs --cfg=stylegan4 --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1 --use-stylegan4=True --use-enhanced-loss=True

    \b
    # Fine-tune StyleGAN4-R for MetFaces using 1 GPU, starting from the pre-trained FFHQ pickle.
    python train_stylegan4.py --outdir=~/training-runs --cfg=stylegan4 --data=~/datasets/metfaces-1024x1024.zip \\
        --gpus=1 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl

    \b
    # Train StyleGAN4 with all advanced features enabled.
    python train_stylegan4.py --outdir=~/training-runs --cfg=stylegan4 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=32 --mirror=1 --use-stylegan4=True --use-enhanced-loss=True \\
        --use-advanced-aug=True --use-attention=True --use-residual=True --use-multi-scale-d=True \\
        --contrastive-weight=0.1 --perceptual-weight=0.1 --feature-matching-weight=1.0 \\
        --progressive-training=True --adaptive-training=True
    """

    # Setup training options.
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(group.group()) for group in prev_run_ids if group is not None]
    cur_run_id = max([0] + prev_run_ids) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-stylegan4')
    assert not os.path.exists(run_dir)

    # Print options.
    print()
    print('Training options:')
    print(f'  Output directory:    {run_dir}')
    print(f'  Training data:       {data}')
    print(f'  Class labels:        {cond}')
    print(f'  Dataset x-flips:     {mirror}')
    print(f'  Augmentation:        {aug}')
    print(f'  Resume from:         {resume}')
    print(f'  Number of GPUs:      {num_gpus}')
    print(f'  Batch size:          {batch_size}')
    print(f'  Batch size per GPU:  {batch_gpu}')
    print(f'  Gradient accumulation: {batch_size // (batch_gpu * num_gpus)}')
    print(f'  Total training time: {total_kimg} kimg')
    print(f'  Image snapshots:     {image_snapshot_ticks}')
    print(f'  Network snapshots:   {network_snapshot_ticks}')
    print(f'  Random seed:         {seed}')
    print(f'  StyleGAN4:           {use_stylegan4}')
    print(f'  Enhanced loss:       {use_enhanced_loss}')
    print(f'  Advanced aug:        {use_advanced_aug}')
    print(f'  Attention:           {use_attention}')
    print(f'  Residual:            {use_residual}')
    print(f'  Multi-scale D:       {use_multi_scale_d}')
    print(f'  Progressive:         {progressive_training}')
    print(f'  Adaptive:            {adaptive_training}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print(f'Creating output directory...')
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, 'training_options.txt'), 'wt') as f:
        f.write('\n'.join([f'{k}={v}' for k, v in locals().items() if k not in ['ctx', 'outdir']]))
    with open(os.path.join(run_dir, 'README.txt'), 'wt') as f:
        f.write('StyleGAN4 training run.\n')
        f.write('See training_options.txt for details.\n')

    # Launch processes.
    print(f'Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter 