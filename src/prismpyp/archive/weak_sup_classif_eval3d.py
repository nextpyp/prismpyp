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
    
    # Load pre-trained feature extractor
    if args.feature_extractor_weights:
        if os.path.isfile(args.feature_extractor_weights):
            print("=> loading checkpoint '{}'".format(args.feature_extractor_weights))
            checkpoint = torch.load(args.feature_extractor_weights)

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            msg = feature_extractor.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained feature extractor '{}' (epoch {})"
                  .format(args.feature_extractor_weights, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.feature_extractor_weights))
    
    # Load pre-trained classifier
    if args.classifier_weights:
        if os.path.isfile(args.classifier_weights):
            print("=> loading checkpoint '{}'".format(args.classifier_weights))
            checkpoint = torch.load(args.classifier_weights)

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            args.start_epoch = 0
            msg = classifier.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained classifier '{}' (epoch {})"
                  .format(args.classifier_weights, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.classifier_weights))

        
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
            
    # Load dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_dataset = MRCDataset(
        args.data,
        transform=val_transform,
        is_fft=args.use_fft,
        webp_dir=os.path.join(args.nextpyp_preproc, 'webp'),
        metadata_path=args.metadata_path
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    data_for_export = {}
    
    if args.evaluate:
        if args.label_path and args.embedding_path:
            # Use existing labels
            new_labels = torch.load(args.label_path)
            embeddings = torch.load(args.embedding_path)
        else:
            embeddings, new_labels = validate_classifier(test_loader, feature_extractor, classifier, args)
            # Save the new labels
            labels_save_path = os.path.join(output_path, 'labels.pth')
            torch.save(new_labels, labels_save_path)
            print(f"Labels saved to {labels_save_path}")
            
            # Save the embeddings
            embeddings_save_path = os.path.join(output_path, 'embeddings.pth')
            torch.save(embeddings, embeddings_save_path)
            print(f"Embeddings saved to {embeddings_save_path}")
        
        # Convert embeddings to numpy array (new_labels is already a numpy array)
        embeddings = embeddings.cpu().numpy()
        new_labels = new_labels.cpu().numpy()
                    
        # Use labeled clusters to plot
        actual_assignments = new_labels
        n_clusters = len(np.unique(actual_assignments))
        print("Number of clusters: ", n_clusters)
        args.n_clusters = n_clusters
        data_for_export['cluster_id'] = actual_assignments
        
        pca_reducer = PCA(n_components=3)
        pca_fit = pca_reducer.fit_transform(embeddings)
        data_for_export["pca_fit_x"] = pca_fit[:, 0]
        data_for_export["pca_fit_y"] = pca_fit[:, 1]
        data_for_export["pca_fit_z"] = pca_fit[:, 2]        
        
        umap_reducer = umap.UMAP(n_neighbors=args.num_neighbors, min_dist=args.min_dist_umap, random_state=args.seed)
        umap_fit = umap_reducer.fit_transform(embeddings)
        data_for_export["umap_fit_x"] = umap_fit[:, 0]
        data_for_export["umap_fit_y"] = umap_fit[:, 1]
        data_for_export["umap_fit_z"] = umap_fit[:, 2]

        tsne_reducer = TSNE(n_components=2, perplexity=args.num_neighbors, verbose=1, random_state=args.seed, n_iter=1000)
        tsne_fit = tsne_reducer.fit_transform(embeddings)
        data_for_export["tsne_fit_x"] = tsne_fit[:, 0]
        data_for_export["tsne_fit_y"] = tsne_fit[:, 1]
        data_for_export["tsne_fit_z"] = tsne_fit[:, 2]

        # Save images to zip file
        if os.path.exists(os.path.join(output_path, 'thumbnail_images')):
            shutil.rmtree(os.path.join(output_path, 'thumbnail_images'))
        os.makedirs(os.path.join(output_path, 'thumbnail_images'))
            
        if args.nextpyp_preproc is not None:
            webp_dir = os.path.join(args.nextpyp_preproc, 'webp')

            for img_path in test_dataset.file_paths:
                basename = os.path.basename(img_path)
                ctf_file = os.path.join(webp_dir, basename[:-4] + '_ctffit.webp') # Trim .mrc ending
                mg_file = os.path.join(webp_dir, basename[:-4] + '.webp') # Trim .mrc ending
                
                if os.path.exists(ctf_file) and os.path.exists(mg_file):
                    ctf_img = Image.open(ctf_file)
                    mg_img = Image.open(mg_file)
                    crop_and_stitch_imgs(ctf_img, mg_img, output_path, basename)

        else:
            for img_path in test_dataset.file_paths:
                img_arr = mrcfile.read(img_path)
                basename = os.path.basename(img_path)
                img = transforms.ToPILImage()(soft_normalize(img_arr))
                img = img.resize((256, 256))
                img.save(os.path.join(output_path, 'thumbnail_images', basename + '.png'))
        
        # Zip the folder that we just made
        if args.output_path and (not args.distributed or args.rank % ngpus_per_node == 0):
            dest = os.path.join(output_path, 'zipped_thumbnail_images')
            src = os.path.join(output_path, 'thumbnail_images')
            shutil.make_archive(dest, 'gztar', src)
            print(f"Thumbnail images saved to {dest}")
            
        # Save metadata in prep for exporting
        data_for_export['image_thumbnails'] = []
        data_for_export['micrograph_name'] = []
        for idx in test_dataset.file_paths:
            data_for_export['image_thumbnails'].append(idx)
            data_for_export['micrograph_name'].append(os.path.basename(idx)[:-4])
        
        data_for_export_df = pd.DataFrame(data_for_export)
        data_for_export_df = data_for_export_df.merge(test_dataset.metadata, on='micrograph_name', how='inner')
        data_for_export_df['embeddings'] = embeddings.tolist()
        print(data_for_export_df.columns)
        
        # Save data_for_export as zip file
        if args.output_path and (not args.distributed or args.rank % ngpus_per_node == 0):
            path_to_save = os.path.join(output_path, 'data_for_export.parquet.zip')
            data_for_export_df.to_parquet(path_to_save, compression='gzip')
            print(f"Data for export saved to {path_to_save}")


def validate_classifier(val_loader, feature_extractor, classifier, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, 
        #  losses, 
        #  top1, 
        #  top5
        ],
        prefix='Test: ')

    # switch to evaluate mode
    feature_extractor.eval()
    classifier.eval()

    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        end = time.time()
        for i, images in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            if isinstance(feature_extractor, torch.nn.DataParallel) or \
                isinstance(feature_extractor, torch.nn.parallel.DistributedDataParallel):
                encoder = feature_extractor.module.encoder
            else:
                encoder = feature_extractor.encoder
                
            features = encoder(images)
            
            # compute output
            output = classifier(features)
            
            # re-compute pseudolabels
            probabilities = F.softmax(output, dim=1)
            confidence_scores, new_labels = probabilities.max(dim=1)
            conf_thresh = 0 #args.conf_thresh # Assign labels to all points
            final_labels = torch.where(confidence_scores >= conf_thresh, new_labels, -1)

            # Append embeddings and final labels
            all_embeddings.append(features)
            all_labels.append(final_labels)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_embeddings, all_labels


def crop_and_stitch_imgs(image1, image2, output_path, basename):
    """
    image1: ctf image
    image2: micrograph
    """
    # Crop image1 to half, heightwise, to only keep the power spectrum
    width1, height1 = image1.size
    width2, height2 = image2.size
    image1_rotated = image1.rotate(90) # Rotate in degrees counterclockwise
    image1_cropped = image1_rotated.crop((0, 0, image1.width // 2, image1.height))
    
    # Resize image2 to be the same y dimension as image1_rotated
    new_image1_width, new_image1_height = image1_cropped.size
    image2_resized = image2.resize((new_image1_height, new_image1_height))
    
    image1_to_use = image1_cropped
    image2_to_use = image2_resized
    width1, height1 = image1_to_use.size
    width2, height2 = image2_to_use.size
    
    # Create a new image with combined width and the maximum height
    new_width = width1 + width2
    new_height = max(height1, height2)

    # Create a new blank image (white background)
    new_image = Image.new("RGB", (new_width, new_height), (255, 255, 255))

    # Paste the first image at the left side of the new image
    new_image.paste(image1_to_use, (0, 0))

    # Paste the second image at the right side of the new image
    new_image.paste(image2_to_use, (width1, 0))

    # Save the new image
    path_to_save = os.path.join(output_path, 'thumbnail_images', basename + '.combined.jpg')
    new_image.save(path_to_save, "JPEG", quality=100, optimize=True, progressive=True)

def soft_normalize(img):
    p1, p99 = np.percentile(img, (1, 99))
    normalized = np.clip((img - p1) / (p99 - p1), 0, 1)
    normalized *= 255
    normalized = normalized.astype(np.uint8)
    return normalized

                 
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