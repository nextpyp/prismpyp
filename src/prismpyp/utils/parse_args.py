import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

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
import torchvision.datasets as datasets
import torchvision.models as models


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    else:
        # this script is called from prismpyp.__main__ entry point, in which case a parser is already created
        pass
    
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    # Positional arguments
    # parser.add_argument('--micrographs-list', metavar='DIR', help='list of micrographs')
    parser.add_argument('--metadata-path', metavar='METADATA_PATH', help='path to metadata file')
    parser.add_argument('--embedding-path', metavar='EMBEDDING_PATH', nargs='?', default=None,
                        help='optional path to precomputed embeddings')
    
    # General configurations
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                        help='mini-batch size (default: 512), total for all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', dest='lr',
                        help='initial (base) learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', dest='weight_decay',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--feature-extractor-weights', default='', type=str, metavar='PATH',
                        help='path to pre-trained feature extractor weights (default: none)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='use multi-processing distributed training')

    # SimSiam-specific configurations
    parser.add_argument('--dim', default=2048, type=int,
                        help='feature dimension (default: 2048)')
    parser.add_argument('--pred-dim', default=512, type=int,
                        help='hidden dimension of the predictor (default: 512)')
    parser.add_argument('--fix-pred-lr', action='store_true',
                        help='fix learning rate for the predictor')

    # Dataset-specific configurations
    parser.add_argument('--use-fft', action='store_true', 
                        help='use FFT of the image as input')
    parser.add_argument("--downsample", type=int, default=1, 
                        help="downsample the image")
    parser.add_argument("--pixel-size", default=1, type=float, 
                        help="pixel size of the image")
    parser.add_argument("--size", type=int, 
                        help="size of the image in pixels, before downsampling")
    
    # Additional configurations
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')

    # KMeans configurations
    parser.add_argument('--n-clusters', default=100, type=int,
                        help='number of clusters for KMeans')

    # UMap configurations
    parser.add_argument('--num-neighbors', default=30, type=int,
                        help='number of neighbors for UMap')
    parser.add_argument('--min-dist-umap', default=0.3, type=float,
                        help='minimum distance for UMap')
    parser.add_argument('--n-components', default=2, type=int,
                        help='number of components for UMap')
    
    # Image thumbnail configurations
    parser.add_argument('--zip-images', action='store_true', default=False,
                        help='save zipped image thumbnails')
    return parser

def get_args():
    
    args = add_args().parse_args()
    return args