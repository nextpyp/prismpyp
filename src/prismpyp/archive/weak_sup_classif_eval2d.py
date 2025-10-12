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
            checkpoint = torch.load(args.feature_extractor_weights, weights_only=False)

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
            checkpoint = torch.load(args.classifier_weights, weights_only=False)

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
    
    if args.evaluate:
        if args.label_path and args.embedding_path:
            # Use existing labels
            new_labels = torch.load(args.label_path, weights_only=False)
            embeddings = torch.load(args.embedding_path, weights_only=False)
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
        cmap = plt.get_cmap('tab10', n_clusters)
        
        pca_reducer = PCA(n_components=2)
        pca_fit = pca_reducer.fit_transform(embeddings)
        plot_projections(pca_fit, actual_assignments, "PCA Projection", output_path, cmap, args, ngpus_per_node)
        for modality in ["real", "fft"]:
            get_scatter_plot_with_thumbnails(args, pca_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, 
                                            method="pca", K=args.n_clusters, ngpus_per_node=ngpus_per_node, modality=modality)
        
        
        umap_reducer = umap.UMAP(n_neighbors=args.num_neighbors, min_dist=args.min_dist_umap, random_state=args.seed)
        umap_fit = umap_reducer.fit_transform(embeddings)
        plot_projections(umap_fit, actual_assignments, "UMAP Projection", output_path, cmap, args, ngpus_per_node)
        for modality in ["real", "fft"]:
            get_scatter_plot_with_thumbnails(args, umap_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, 
                                            method="umap", K=args.n_clusters, ngpus_per_node=ngpus_per_node, modality=modality)


        tsne_reducer = TSNE(n_components=2, perplexity=args.num_neighbors, verbose=1, random_state=args.seed, n_iter=1000)
        tsne_fit = tsne_reducer.fit_transform(embeddings)
        plot_projections(tsne_fit, actual_assignments, "t-SNE Projection", output_path, cmap, args, ngpus_per_node)
        for modality in ["real", "fft"]:
            get_scatter_plot_with_thumbnails(args, tsne_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, 
                                                method="tsne", K=args.n_clusters, ngpus_per_node=ngpus_per_node, modality=modality)


        for i in range(5):
            random_idx = random.randint(0, len(test_dataset) - 1)
            plot_nearest_neighbors_3x3(args, embeddings, test_dataset, example_idx=random_idx, i=i, 
                                        path_to_save=output_path, ngpus_per_node=ngpus_per_node)


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
            print("[DEBUGGING] new labels: ", new_labels)
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


def plot_projections(projection, actual_assignments, title, path_to_save, cmap, args, ngpus_per_node, num_neighbors=None, min_dist_umap=None):
    plt.scatter(projection[:, 0], projection[:, 1], c=actual_assignments, cmap=cmap, s=10, alpha=0.8)
    plt.title(title)
    num_neighbors = args.num_neighbors if num_neighbors is None else num_neighbors
    min_dist_umap = args.min_dist_umap if min_dist_umap is None else min_dist_umap
    title = title.split(" ")[0]
    save_title = f"scatter_plot_{title}.png"
    # if args.min_dist_umap:
    #     save_title += f"_{min_dist_umap}"
    # save_title += ".png"
    
    if not args.distributed or args.rank % ngpus_per_node == 0:
        plt.savefig(os.path.join(path_to_save, save_title))
    plt.clf()
       

def get_scatter_plot_with_thumbnails(
    args, embeddings_2d, cmap, labels, test_dataset, path_to_save, method, K, is_mrc=True, ngpus_per_node=1, is_wandb=False, modality="real"
):
    # Use nextpyp-postprocessed .webp micrograph thumbnails for real images
    if args.nextpyp_preproc is not None:
        webp_dir =  os.path.join(args.nextpyp_preproc, "webp")
    
    # Normalize embeddings
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    fig.suptitle(f"Scatter Plot Using {method} Projection, K = {K}")
    
    # shuffle images and find out which images to show
    shown_images_idx = []
    # shown_images = np.array([[1.0, 1.0]])
    shown_images = np.expand_dims(embeddings_2d[0], axis=0)
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 5e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap, s=10, alpha=0.8)

    for idx in shown_images_idx:
        if modality == "real":
            if args.nextpyp_preproc is not None:
                # img = test_dataset.get_real_image(idx)
                img_path = test_dataset.file_paths[idx]
                # print("img_path: ", img_path)
                basename = os.path.basename(img_path)
                mg_file = os.path.join(webp_dir, basename.replace(".mrc", ".webp"))
                
                if os.path.exists(mg_file):
                    img = Image.open(mg_file)
            else:
                img = test_dataset.get_real_image(idx)
                
        elif modality == "fft":
            img = test_dataset.get_fft_image(idx)
        
        thumbnail_size = int(rcp["figure.figsize"][0] * 7.0)
        # thumbnail_zoom = 0.5 if dataset_name.lower() == "mnist" else 0.7  # Adjust zoom
        if isinstance(img, Image.Image):
            sharpened = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            img = sharpened.resize((thumbnail_size, thumbnail_size), Image.LANCZOS)
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=1))
            # img = ImageEnhance.Contrast(img)
        elif isinstance(img, np.ndarray) or isinstance(img, torch.Tensor):
            img = functional.to_pil_image(img)
            img = functional.resize(img, thumbnail_size)
            img = np.array(img, dtype=np.uint8)
            
        
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap="gray"),
            embeddings_2d[idx],
            pad=0.2,
        )
        img_box.patch.set_edgecolor(cmap(labels[idx]))
        ax.add_artist(img_box)
        
    # set aspect ratio
    ratio = 1.0 / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable="box")
    # Adjust axes limits to fit the data tightly
    ax.set_xlim(embeddings_2d[:, 0].min() - 0.1, embeddings_2d[:, 0].max() + 0.1)
    ax.set_ylim(embeddings_2d[:, 1].min() - 0.1, embeddings_2d[:, 1].max() + 0.1)
    
    # Set aspect ratio and remove unnecessary whitespace
    # ax.set_aspect("equal", adjustable="datalim")
    # plt.tight_layout()
    if is_wandb:
        wandb.log({f"scatter_plot_{method}": wandb.Image(plt)})
    elif not args.distributed or args.rank % ngpus_per_node == 0:
        # print("[DEBUGGING][regular] save name: ", f"{path_to_save}/thumbnail_plot_{method}.png")
        plt.savefig(f"{path_to_save}/thumbnail_plot_{method}_{modality}.png")
    plt.clf()
    

def plot_nearest_neighbors_3x3(args, embeddings, test_dataset, example_idx, i, path_to_save=None, ngpus_per_node=1):
    """Plots the example image and its eight nearest neighbors."""
    n_subplots = 9
    fig = plt.figure()
    fig.suptitle(f"Nearest Neighbor Plot {i + 1}")
    
    # Calculate distances to the center
    distances = embeddings - embeddings[example_idx]
    distances = np.power(distances, 2).sum(-1).squeeze()
    nearest_neighbors = np.argsort(distances)[:n_subplots]
    
    # Show images
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)
        
        if args.nextpyp_preproc is not None:
            webp_dir = os.path.join(args.nextpyp_preproc, "webp")
            img_path = test_dataset.file_paths[plot_idx]
            basename = os.path.basename(img_path)
            mg_file = os.path.join(webp_dir, basename.replace(".mrc", ".webp"))
            if os.path.exists(mg_file):
                img = Image.open(mg_file)
        else:
            tmp = test_dataset[plot_idx]
            if len(tmp) == 2:
                img, _ = tmp
            else:
                img = tmp
            # img, _ = test_dataset[plot_idx]
            img = img.permute(1, 2, 0).numpy() if img.ndimension() == 3 else img.numpy()
            img = soft_normalize(img)
            img = (img * 255).astype('uint8') if img.max() <= 1 else img        
        
        if plot_offset == 0:
            ax.set_title(f"Example Image")
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img, cmap="gray")
        plt.axis("off")
    
    if path_to_save:
        if not args.distributed or args.rank % ngpus_per_node == 0:
            plt.savefig(f"{path_to_save}/nearest_neighbors_{i + 1}.png")
    plt.clf()      

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