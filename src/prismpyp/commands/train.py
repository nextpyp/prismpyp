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
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, run_name))
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier(device_ids=[args.gpu])
    
    # create model
    assert 'resnet' in args.arch, "Only ResNet models are supported."
    print("=> creating model '{}'".format(args.arch))
    if args.resume is not None or args.pretrained:
        is_pretrained = True
    else:
        is_pretrained = False
    model = simsiam_builder.SimSiam(
        args.arch,
        args.dim, args.pred_dim,
        use_checkpoint=True,
        pretrained=is_pretrained)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        if isinstance(model, torch.nn.DataParallel) or \
            isinstance(model, torch.nn.parallel.DistributedDataParallel):
            optim_params = [{'params': model.module.backbone.parameters(), 'fix_lr': False},
                            {'params': model.module.reducer.parameters(), 'fix_lr': False},
                            {'params': model.module.projector.parameters(), 'fix_lr': False},
                            {'params': model.module.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = [{'params': model.backbone.parameters(), 'fix_lr': False},
                            {'params': model.reducer.parameters(), 'fix_lr': False},
                            {'params': model.projector.parameters(), 'fix_lr': False},
                            {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, 1e-4,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    trainfiles = args.micrographs_list
    
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    # Load parameters from a .toml file
    toml_file_path = os.path.join(args.nextpyp_preproc, '.pyp_config.toml')
    with open(toml_file_path, 'r') as toml_file:
        pyp_params = toml.load(toml_file)

    if args.use_fft:
        augmentation = [
            transforms.Resize((224, 224)),  # Resize image to 224x224
            # cryst_xforms.HPF(
            #     pixel_size=pyp_params.get('scope_pixel', 1) * (224/512), 
            #     cutoff=20 * (224/512), 
            #     prob=0.5
            # ),
            # cryst_xforms.RandomCLAHEOrSharpen(prob_clahe=0.3, prob_sharpen=0.3),
            transforms.RandomResizedCrop(224, (0.9, 1)),  # Keep image the same dimension
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            # cryst_xforms.RandomRadialMask(prob=0.5),
            cryst_xforms.RandomRadialStretch(prob=0.5)
            # normalize,
        ]
    else:
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1)),
            # cryst_xforms.HistogramEqualization(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(60),
            transforms.ColorJitter(brightness=0.2, 
                                    contrast=0.2, 
                                    saturation=0.2, 
                                    hue=0.2),  # not strengthened
            simsiam_loader.GaussianBlur([.1, .5]),
            transforms.ToTensor(),
        ]
    
    train_dataset = MRCDataset(
        mrc_dir=trainfiles, 
        transform=simsiam_loader.TwoCropsTransform(transforms.Compose(augmentation)),
        webp_dir=os.path.join(args.nextpyp_preproc, 'webp'),
        is_fft=args.use_fft,
        metadata_path=args.metadata_path,
        pixel_size=pyp_params.get('scope_pixel', 1)
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=False, sampler=train_sampler, drop_last=True)
    
    epoch_losses = []
    collapse_level = []
    
    best_loss = float('inf')
    best_collapse_level = float('inf')
    is_best = False
    is_lowest_collapse = False
    is_last = False
    
    # For early stopping
    patience = args.epochs
    epochs_no_improve = 0
    avg_output_std = 0
    w = 0.9
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        epoch_loss, this_avg_output_std = train(train_loader, model, criterion, optimizer, epoch, w, args, writer)
        avg_output_std = w * avg_output_std + (1 - w) * this_avg_output_std
        epoch_losses.append(np.mean(np.array(epoch_loss), axis=None))
        
        # calculate collapse
        this_collapse_level = max(0.0, 1 - math.sqrt(args.dim) * avg_output_std)
        print("[Epoch: {}] Collapse Level: {}/1.00".format(epoch, this_collapse_level))
        collapse_level.append(this_collapse_level)
        
        if writer is not None:
            # Log total loss and collapse level to TensorBoard
            writer.add_scalar("Collapse/Level", this_collapse_level, epoch)
            
        this_loss = np.mean(np.array(epoch_loss), axis=None)
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
        
        
        # Save best epoch
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            filename = os.path.join(args.output_path, 'checkpoints', 'checkpoint_{:04d}.pth.tar'.format(epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, is_last=is_last, is_lowest_collapse=is_lowest_collapse, filename=filename)
            
            # Record which epoch was the best and which has the lowest collapse
            if is_best:
                best_epoch = epoch
                with open(os.path.join(args.output_path, 'checkpoints', 'best_epoch.txt'), 'w') as f:
                    f.write(f"Best epoch: {best_epoch}, loss: {best_loss}, collapse level: {this_collapse_level}")
            if is_lowest_collapse:
                best_collapse_epoch = epoch
                with open(os.path.join(args.output_path, 'checkpoints', 'best_collapse_epoch.txt'), 'w') as f:
                    f.write(f"Best collapse epoch: {best_collapse_epoch}, loss: {this_loss}, collapse level: {best_collapse_level}")
        
        if epochs_no_improve >= patience:
            # Wait for all processes to reach this point
            if args.distributed:
                torch.distributed.barrier(device_ids=[args.gpu])
        
            print(f'Early stopping at epoch {epoch}.')
            filename = os.path.join(args.output_path, 'checkpoints', 'checkpoint_{:04d}.pth.tar'.format(epoch))
            # Save checkpoint before exiting
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, is_last=True, is_lowest_collapse=False, filename=filename)
            
            # Wait for all processes to reach this point
            if args.distributed:
                torch.distributed.barrier(device_ids=[args.gpu])
            break
    
    # Ensure all processes reach this point
    if args.distributed:
        torch.distributed.barrier(device_ids=[args.gpu])
    
    # Close tensorboard
    writer.close()

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed 
                                                and args.rank % ngpus_per_node == 0):
        plot(epoch_losses, args, 'Total Loss', 'Total Loss', 'total_loss.webp')
        plot(collapse_level, args, 'Collapse Level', 'Collapse Level', 'collapse_level.webp')
        
    # Write all args to a .yaml file
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed 
                                                and args.rank % ngpus_per_node == 0):
        with open(os.path.join(args.output_path, 'training_config.yaml'), 'w') as file:
            yaml.dump(vars(args), file)
            

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