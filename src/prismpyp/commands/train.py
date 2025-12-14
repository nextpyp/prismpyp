#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import yaml
import toml
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
from torch.amp import autocast, GradScaler

from prismpyp.simsiam import loader as simsiam_loader
from prismpyp.simsiam import builder as simsiam_builder
from prismpyp.utils.mrc_dataset import MRCDataset
from prismpyp.utils import parse_args
from prismpyp.core import crystalline_transforms as cryst_xforms
from prismpyp.core import fft_transforms as fft_xforms

# For Tensorboard
from torch.utils.tensorboard import SummaryWriter

from functools import wraps  # new import

def print_training_time(elapsed, epoch=None):
    elapsed_hours = int(elapsed // 3600)
    elapsed_minutes = int((elapsed % 3600) // 60)
    elapsed_seconds = elapsed % 60
    if epoch is not None:
        epoch_string = f"[epoch {epoch}]"
    else:
        epoch_string = ""
    if elapsed_hours > 0:
        print(f"[train]{epoch_string} finished in {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds:.1f}s")
    elif elapsed_minutes > 0:
        print(f"[train]{epoch_string} finished in {elapsed_minutes}m {elapsed_seconds:.1f}s")
    else:
        print(f"[train]{epoch_string} finished in {elapsed:.1f}s")

def main(args):

    if not os.path.exists(args.output_path):
        print("Output directory does not exist.")
        os.makedirs(args.output_path)
        # return
    if not os.path.exists(os.path.join(args.output_path, 'checkpoints')):
        print("Checkpoints directory does not exist.")
        os.makedirs(os.path.join(args.output_path, 'checkpoints'))
        # return
    
    tensorboard_dir = os.path.join(args.output_path, 'runs')
    print("TENSORBOARD DIR IS: ", tensorboard_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    # Create a unique subdirectory for this run
    run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    print("THIS TENSORBOARD DIR IS: " , run_name)
            
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, tensorboard_dir, run_name))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, tensorboard_dir, run_name)


def main_worker(gpu, ngpus_per_node, args, tensorboard_dir, run_name):
    import os, math, yaml, builtins
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.backends.cudnn as cudnn
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import transforms

    # identify ranks early
    args.gpu = gpu
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ.get("RANK", 0))
    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + gpu
    is_dist = bool(args.distributed)
    is_main_process = (not args.multiprocessing_distributed) or (args.rank % ngpus_per_node == 0)

    writer = None
    model = None
    train_loader = None
    train_sampler = None

    try:
        # only main process writes the config and logs to TB
        if is_main_process:
            os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
            with open(os.path.join(args.output_path, "training_config.yaml"), "w") as f:
                yaml.dump(vars(args), f)

        if is_main_process:
            writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, run_name))

        # silence non-master prints to keep logs readable
        if args.multiprocessing_distributed and args.gpu != 0:
            def _print_noop(*_a, **_k): pass
            builtins.print = _print_noop

        if args.gpu is not None:
            print(f"Use GPU: {args.gpu} for training")
        
        if is_dist:
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
                timeout=torch.timedelta(minutes=10),
            )
            print(f"Initialized distributed process group: rank {args.rank}/{args.world_size}.")
            # ensure CUDA device is set before any collective work
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
            print("Done 2")
            
            try:
                dist.barrier()
            except Exception:
                print("A rank lagged during initial barrier.")
                pass  # do not fail if a rank lagged during warmup
        
        # build model
        assert 'resnet' in args.arch, "Only ResNet models are supported."
        print(f"=> creating model '{args.arch}'")
        is_pretrained = bool(args.resume is not None or args.pretrained)

        model = simsiam_builder.SimSiam(
            args.arch,
            args.dim, args.pred_dim,
            use_checkpoint=True,
            pretrained=is_pretrained,
        )

        init_lr = args.lr * args.batch_size / 256.0

        if is_dist:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # shard batch and workers per process
                args.batch_size = max(1, int(args.batch_size // ngpus_per_node))
                args.workers = int((args.workers + ngpus_per_node - 1) // ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            raise NotImplementedError("Only DistributedDataParallel is supported.")

        print(model)  # after SyncBN

        criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu if args.gpu is not None else 0)

        if args.fix_pred_lr:
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                optim_params = [
                    {'params': model.module.backbone.parameters(),  'fix_lr': False},
                    {'params': model.module.reducer.parameters(),   'fix_lr': False},
                    {'params': model.module.projector.parameters(), 'fix_lr': False},
                    {'params': model.module.predictor.parameters(), 'fix_lr': True},
                ]
            else:
                optim_params = [
                    {'params': model.backbone.parameters(),  'fix_lr': False},
                    {'params': model.reducer.parameters(),   'fix_lr': False},
                    {'params': model.projector.parameters(), 'fix_lr': False},
                    {'params': model.predictor.parameters(), 'fix_lr': True},
                ]
        else:
            optim_params = model.parameters()

        optimizer = torch.optim.SGD(
            optim_params, init_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        cudnn.benchmark = True

        # -------- dataset and loader --------
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        trainfiles = os.path.join(args.metadata_path, 'all_micrographs_list.micrographs')
        assert os.path.exists(trainfiles), f".micrographs list does not exist at {args.metadata_path}."

        with open(os.path.join(args.metadata_path, 'pixel_size.txt'), 'r') as f:
            pixel_size = float(f.readline().strip())

        if args.use_fft:
            augmentation = [
                transforms.Resize((224, 224)),
                cryst_xforms.HPF(pixel_size=pixel_size * (224/512), cutoff=20 * (224/512), prob=0.5),
                cryst_xforms.RandomCLAHEOrSharpen(prob_clahe=0.3, prob_sharpen=0.3),
                transforms.RandomResizedCrop(224, (0.9, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                cryst_xforms.RandomRadialStretch(prob=0.5),
            ]
        else:
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(60),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                simsiam_loader.GaussianBlur([.1, .5]),
                transforms.ToTensor(),
            ]

        train_dataset = MRCDataset(
            mrc_dir=trainfiles,
            transform=simsiam_loader.TwoCropsTransform(transforms.Compose(augmentation)),
            webp_dir=os.path.join(args.metadata_path, 'webp'),
            is_fft=args.use_fft,
            metadata_path=os.path.join(args.metadata_path, 'micrograph_metadata.csv'),
            pixel_size=pixel_size,
        )

        if is_dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        else:
            train_sampler = None

        # keep workers=0 or set persistent_workers=False if you raise workers
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=0,
            pin_memory=False,
            sampler=train_sampler,
            drop_last=True,
            persistent_workers=False if args.workers else False,
        )

        # -------- training loop --------
        epoch_losses = []
        collapse_level = []
        best_loss = float('inf')
        best_collapse_level = float('inf')
        is_best = False
        is_lowest_collapse = False
        is_last = False

        patience = args.epochs
        epochs_no_improve = 0
        avg_output_std = 0.0
        w = 0.9

        start_time = time.perf_counter()
        for epoch in range(args.start_epoch, args.epochs):
            if is_dist and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            adjust_learning_rate(optimizer, init_lr, epoch, args)

            epoch_loss, this_avg_output_std = train(train_loader, model, criterion, optimizer, epoch, w, args, writer if is_main_process else None)
            
            avg_output_std = w * avg_output_std + (1 - w) * this_avg_output_std
            epoch_losses.append(float(np.mean(np.array(epoch_loss))))
            this_collapse_level = max(0.0, 1 - math.sqrt(args.dim) * avg_output_std)
            if is_main_process:
                print(f"[Epoch: {epoch}] Collapse Level: {this_collapse_level}/1.00")
            collapse_level.append(this_collapse_level)

            if writer is not None and is_main_process:
                writer.add_scalar("Collapse/Level", this_collapse_level, epoch)

            this_loss = float(np.mean(np.array(epoch_loss)))
            if this_loss < best_loss:
                best_loss = this_loss
                is_best = True
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch == args.epochs - 1:
                is_last = True
            if this_collapse_level < best_collapse_level:
                best_collapse_level = this_collapse_level
                is_lowest_collapse = True

            if is_main_process:
                filename = os.path.join(args.output_path, 'checkpoints', f'checkpoint_{epoch:04d}.pth.tar')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, is_last=is_last, is_lowest_collapse=is_lowest_collapse, filename=filename)

                if is_best:
                    with open(os.path.join(args.output_path, 'checkpoints', 'best_epoch.txt'), 'w') as f:
                        f.write(f"Best epoch: {epoch}, loss: {best_loss}, collapse level: {this_collapse_level}")
                if is_lowest_collapse:
                    with open(os.path.join(args.output_path, 'checkpoints', 'best_collapse_epoch.txt'), 'w') as f:
                        f.write(f"Best collapse epoch: {epoch}, loss: {this_loss}, collapse level: {best_collapse_level}")

            # reset epoch flags
            is_best = False
            is_lowest_collapse = False

            if epochs_no_improve >= patience:
                if is_dist:
                    try: dist.barrier()
                    except Exception: pass
                if is_main_process:
                    print(f"Early stopping at epoch {epoch}.")
                    filename = os.path.join(args.output_path, 'checkpoints', f'checkpoint_{epoch:04d}.pth.tar')
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best=False, is_last=True, is_lowest_collapse=False, filename=filename)
                    early_stopping_time = time.perf_counter() - start_time
                    print_training_time(early_stopping_time, epoch=None)
                if is_dist:
                    try: dist.barrier()
                    except Exception: pass
                break
        
        total_elapsed = time.perf_counter() - start_time
        if is_main_process:
            print_training_time(total_elapsed, epoch=None)
            
        if is_dist:
            try: dist.barrier()
            except Exception: pass

        if writer is not None and is_main_process:
            writer.flush()
            writer.close()
            writer = None

        if is_main_process:
            plot(epoch_losses, args, 'Total Loss', 'Total Loss', 'total_loss.webp')
            plot(collapse_level, args, 'Collapse Level', 'Collapse Level', 'collapse_level.webp')

        if is_dist:
            try: dist.barrier()
            except Exception: pass

    except KeyboardInterrupt:
        if is_main_process:
            print("Caught KeyboardInterrupt. Cleaning up...")
        raise  # let torchrun register nonzero exit
    finally:
        # Best-effort quiet down CUDA and DDP before destroying the group
        try:
            if model is not None:
                try:
                    model.cpu()  # release NCCL buckets
                except Exception:
                    pass
                del model
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception:
            pass

        # close writer on any leftover path
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass

        # final barrier then destroy process group
        if is_dist and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass

        # return explicitly to avoid accidental sys.exit in launchers
        return
    

def plot(arr, args, label, title, save_title):
    
    plt.figure()
    plt.plot(range(args.start_epoch, args.epochs), arr, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(label.capitalize())
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_path, save_title))
    plt.close()


def train(train_loader, model, simsiam_loss, optimizer, epoch, w, args, writer=None):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Total Loss', ':.4f')
    
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, losses, 
        ],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    all_losses = []
    
    batch_size = args.batch_size
    end = time.time()
    avg_output_std = 0
    
    # Check that cuda is available before using GradScaler
    assert torch.cuda.is_available(), "CUDA must be available in order to use GradScaler"
    scaler = GradScaler(device='cuda')
    
    for i, images in enumerate(train_loader):
        optimizer.zero_grad()
        
        batch_loss = []
        
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        
        # Used for gradient checkpointing
        images[0].requires_grad_()
        images[1].requires_grad_()

        # compute output and loss
        with autocast(device_type='cuda'):
            p1, p2, z1, z2, f1, f2 = model(
                x1=images[0], 
                x2=images[1], 
            )
            total_loss = -(simsiam_loss(p1, z2).mean() + simsiam_loss(p2, z1).mean()) * 0.5
        scaler.scale(total_loss).backward()
        batch_loss.append(total_loss.item())
        
        losses.update(total_loss.item(), images[0].size(0))

        output = p1.detach()
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()
        
        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
        all_losses.append(np.mean(np.array(batch_loss), axis=None))
        
        if writer is not None:
            # Log loss
            writer.add_scalar("Loss/Total", np.mean(np.array(all_losses), axis=None), epoch)

    return all_losses, avg_output_std


def save_checkpoint(state, is_best, is_last, is_lowest_collapse, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    is_best_filename = os.path.dirname(filename) + '/model_best.pth.tar'
    is_last_filename = os.path.dirname(filename) + '/model_last.pth.tar'
    is_lowest_collapse_filename = os.path.dirname(filename) + '/model_lowest_collapse.pth.tar'
    if is_best:
        torch.save(state, is_best_filename)
    if is_last:
        torch.save(state, is_last_filename)
    if is_lowest_collapse:
        torch.save(state, is_lowest_collapse_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main(add_args().parse_args())