import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import mrcfile
import pandas as pd
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.builder
from mrc_dataset import MRCDataset, WeakSupMRCDataset
from feature_clustering import WeaklySupevisedLabels
import parse_args
from bootstrap_loss import BootstrapLoss

# For dimensionality reduction
import umap
import faiss
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# for plotting
import matplotlib.offsetbox as osb

# for resizing images to thumbnails
import torchvision.transforms.functional as functional
from matplotlib import rcParams as rcp

# for parameter sweeps
import wandb

def main():
    args = parse_args.get_args()
    
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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for inference".format(args.gpu))
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    
    output_path = os.path.join(args.output_path, 'inference')
    if args.output_path and (not args.distributed or args.rank % ngpus_per_node == 0):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if args.distributed:
            dist.barrier(device_ids=[args.gpu])
    
    # Get pseudolabels
    weak_labels = WeaklySupevisedLabels(
                    metadata_path=args.metadata_path,
                    output_path=output_path,
                    min_dist_umap=args.min_dist_umap
                )
    weak_labels.run()
                
    # Load pre-trained SimSiam model for feature extraction
    print("=> loading feature extractor '{}'".format(args.arch))
    feature_extractor = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)
    
    # Instantiate MLP classifier
    print("=> building classifier")
    classifier = simsiam.builder.MLPClassifier(
        input_dim=args.dim,
        num_classes=weak_labels.num_clusters
    )
    
    criterion = BootstrapLoss().cuda(args.gpu)
    init_lr = args.lr * args.batch_size / 256
    optimizer = torch.optim.Adam(classifier.parameters(), lr=init_lr)
    
    # Load pre-trained feature extractor
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, weights_only=True)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc, weights_only=True)

            args.start_epoch = checkpoint['epoch']
            feature_extractor.load_state_dict(checkpoint['state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            feature_extractor.cuda(args.gpu)
            classifier.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[args.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu])
        else:
            feature_extractor.cuda()
            classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        feature_extractor = feature_extractor.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            feature_extractor.features = torch.nn.DataParallel(feature_extractor.features)
            feature_extractor.cuda()
            classifier.features = torch.nn.DataParallel(classifier.features)
            classifier.cuda()
        else:
            feature_extractor = torch.nn.DataParallel(feature_extractor).cuda()
            classifier = torch.nn.DataParallel(classifier).cuda()
    cudnn.benchmark = True
            
    # Load weakly-supervised dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Split valid_indices into train and validation sets
    weakly_labeled_train_dataset, weakly_labeled_val_dataset = get_train_val_datasets(
        args, 
        weak_labels.valid_indices, 
        weak_labels.filtered_labels, 
        train_transform, 
        val_transform
    )
    
    # print(len(weakly_labeled_train_dataset), len(weakly_labeled_val_dataset))
        
    train_loader, val_loader = get_train_val_dataloaders(
        args, 
        weakly_labeled_train_dataset, 
        weakly_labeled_val_dataset
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
    )
    
    if args.distributed:
        weak_sup_sampler = torch.utils.data.distributed.DistributedSampler(weakly_labeled_train_dataset)
    else:
        weak_sup_sampler = None
        
    train_losses = []
    train_top1_acc = []
    val_top1_acc = []
    best_acc1 = 0
    is_best = False
    
    for round in range(0, args.rounds):
        # print("Round: {}".format(round))
        for epoch in range(args.start_epoch, args.epochs, args.rounds):
            # print("Epoch: {}".format(epoch//args.rounds))
            
            if args.distributed:
                weak_sup_sampler.set_epoch(epoch)
            
            # adjust_learning_rate(optimizer, init_lr, epoch, args)
            
            # train for one epoch
            loss, top1 = train_classifier(
                feature_extractor, 
                classifier, 
                train_loader, 
                criterion, 
                optimizer, 
                epoch, 
                round, 
                lr_scheduler, 
                args
            )
            train_losses.append(loss)
            train_top1_acc.append(top1)

            # evaluate on validation set
            acc1, new_labels = validate_classifier(val_loader, feature_extractor, classifier, criterion, args)
            val_top1_acc.append(acc1)
            
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': classifier.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, filename=os.path.join(args.output_path, 'checkpoint.pth.tar'))

        # Update labels using most confident predictions from the classifier
        # valid_indices = new_labels >= 0
        # print("[DEBUGGING] Number of valid indices: ", len(valid_indices), "previously: ", len(weak_labels.valid_indices))
        # filtered_labels = new_labels
        # # print("[DEBUGGING] Number of filtered labels: ", len(filtered_labels), "previously: ", len(weak_labels.filtered_labels))
        # new_train_dataset, new_val_dataset = get_train_val_datasets(
        #     args, 
        #     valid_indices, 
        #     filtered_labels, 
        #     train_transform, 
        #     val_transform
        # )
        # # print("[DEBUGGING] Number of new train dataset: ", len(new_train_dataset), "previously: ", len(weakly_labeled_train_dataset))
        # # print("[DEBUGGING] Number of new val dataset: ", len(new_val_dataset), "previously: ", len(weakly_labeled_val_dataset))
        # train_loader, val_loader = get_train_val_dataloaders(
        #     args, 
        #     new_train_dataset, 
        #     new_val_dataset
        # )
        
    # Plot training and validation losses and accuracies
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        for metric, title, ylabel, save_title in zip(
            [train_losses, train_top1_acc, val_top1_acc],
            ['Training Loss', 'Training Top-1 Accuracy', 'Validation Top-1 Accuracy'],
            ['Loss', 'Accuracy', 'Accuracy'],
            ['train_loss', 'train_top1_acc', 'val_top1_acc']
        ):
            plot_metrics(args, metric, title, ylabel, save_title)
        
        
def get_train_val_datasets(args, valid_indices, filtered_labels, train_transform, val_transform):
    train_idx = int(0.8 * len(valid_indices))
    
    # Create masks to split indices for train and val sets
    train_indices_mask = np.array([True] * train_idx + [False] * (len(valid_indices) - train_idx))
    val_indices_mask = np.array([False] * train_idx + [True] * (len(valid_indices) - train_idx))
    
    # Slice valid_indices according to the masks
    train_valid_indices = valid_indices * train_indices_mask
    val_valid_indices = valid_indices * val_indices_mask
        
    # Slice filtered_labels according to the split
    # train_labels = filtered_labels[:train_idx]
    # val_labels = filtered_labels[train_idx:]

    # Pass sliced valid indices and labels into datasets
    weakly_labeled_train_dataset = WeakSupMRCDataset(
        args.data, 
        transform=train_transform,
        is_fft=args.use_fft,
        webp_dir=os.path.join(args.nextpyp_preproc, 'webp'),
        metadata_path=args.metadata_path,
        valid_indices=train_valid_indices,
        labels=filtered_labels
    )
    
    weakly_labeled_val_dataset = WeakSupMRCDataset(
        args.data,
        transform=val_transform,
        is_fft=args.use_fft,
        webp_dir=os.path.join(args.nextpyp_preproc, 'webp'),
        metadata_path=args.metadata_path,
        valid_indices=val_valid_indices,
        labels=filtered_labels
    )
    
    return weakly_labeled_train_dataset, weakly_labeled_val_dataset

    
def get_train_val_dataloaders(args, train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    return train_loader, val_loader
    
        
def train_classifier(feature_extractor, classifier, loader, criterion, optimizer, epoch, round, lr_scheduler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    lr = AverageMeter('LR', ':.4e')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, lr, losses, top1],
        prefix="Round: [{}]  Epoch: [{}]".format(round, epoch))
    
    feature_extractor.eval()
    classifier.train()
    
    end = time.time()
    
    batch_loss = []
    batch_top1_acc = []
    
    for i, (images, target) in enumerate(loader):
        # print("Batch: {}".format(i))
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        if torch.cuda.is_available() or args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        with torch.no_grad():
            if isinstance(feature_extractor, torch.nn.DataParallel) or \
                isinstance(feature_extractor, torch.nn.parallel.DistributedDataParallel):
                encoder = feature_extractor.module.encoder
            else:
                encoder = feature_extractor.encoder
            features = encoder(images)
        output = classifier(features)
        
        # log learning rate
        lr.update(lr_scheduler.get_last_lr()[0])
        
        # compute loss
        loss = criterion(output, target)
        losses.update(loss.item(), images.size(0))
        
        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0].item(), images.size(0))
        
        # print("Loss: {}, Acc@1: {}".format(losses.avg, top1.avg))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch + i / len(loader))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)
        
        batch_loss.append(loss.item())
        batch_top1_acc.append(acc1[0])
    
    return np.mean(batch_loss), top1.avg


def validate_classifier(val_loader, feature_extractor, classifier, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, 
         losses, 
         top1, 
        #  top5
        ],
        prefix='Test: ')

    # switch to evaluate mode
    feature_extractor.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            if isinstance(feature_extractor, torch.nn.DataParallel) or \
                isinstance(feature_extractor, torch.nn.parallel.DistributedDataParallel):
                encoder = feature_extractor.module.encoder
            else:
                encoder = feature_extractor.encoder
            features = encoder(images)
            
            # compute output
            output = classifier(features)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            # top5.update(acc5[0], images.size(0))
            
            # re-compute pseudolabels
            probabilities = F.softmax(output, dim=1)
            confidence_scores, new_labels = probabilities.max(dim=1)
            conf_thresh = args.conf_thresh
            # Only keep labels where confidence is high, otherwise mark as -1 (unknown)
            final_labels = torch.where(confidence_scores >= conf_thresh, new_labels, -1)

            # Convert to NumPy for indexing
            final_labels_np = final_labels.cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f}'
        #       .format(top1=top1))

    return top1.avg, final_labels_np


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def plot_metrics(args, arr, title, ylabel, save_title):
    plt.figure(figsize=(6, 4))
    plt.plot(range(args.start_epoch, args.epochs), arr)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(args.output_path, save_title))
       
 
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
        param_group['lr'] = cur_lr
        

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

if __name__ == '__main__':
    main()