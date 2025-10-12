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

import simsiam.loader
import simsiam.builder
from mrc_dataset import MRCDataset
import parse_args

# For dimensionality reduction
import umap
import faiss
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# for plotting
import matplotlib.offsetbox as osb

# for resizing images to thumbnails
import torchvision.transforms.functional as functional
from matplotlib import rcParams as rcp

# for retrieving MNIST and CIFAR-10 images
from torchvision import datasets, transforms
from tqdm import tqdm

# for parameter sweeps
import wandb

import fft

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
    
    # Create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)
    
    # Load pre-trained
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            # checkpoint = torch.load(args.pretrained, map_location="cpu")
            checkpoint = torch.load(args.pretrained)

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    
    if args.distributed:
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
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
        
   
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    
    test_dataset = MRCDataset(
                    args.data, 
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]),
                    is_fft=args.use_fft,
                    downsample=args.downsample,
                    pixel_size=args.pixel_size
                    )
        
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    if args.output_path and (not args.distributed or args.rank % ngpus_per_node == 0):
        output_path = os.path.join(args.output_path, 'inference')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if args.distributed:
            dist.barrier(device_ids=[args.gpu])
    
    DO_ABLATIONS = args.do_ablations
    
    if args.evaluate:
        all_embeddings = validate(test_loader, model, args)
        filepath = os.path.join(output_path, 'embeddings.pth')
        torch.save(all_embeddings, filepath)
        print(f"Embeddings saved to {filepath}")
        
        # Convert embeddings to numpy array
        all_embeddings = all_embeddings.cpu().numpy()
        
        if DO_ABLATIONS:
            # Launch the sweep agent
            start_sweep(test_dataset, all_embeddings, args)
        else:
            # Do KMeans clustering on the embeddings
            actual_assignments = kmeans_clustering(all_embeddings, args.n_clusters)
            
            # Do UMAP/TSNE/PCA projection on embeddings
            cmap = plt.get_cmap('tab10', args.n_clusters)
            
            pca_reducer = PCA(n_components=2)
            pca_fit = pca_reducer.fit_transform(all_embeddings)
            plot_projections(pca_fit, actual_assignments, "PCA Projection", output_path, cmap, args, ngpus_per_node)
            get_scatter_plot_with_thumbnails(args, pca_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, 
                                             method="pca", K=args.n_clusters, ngpus_per_node=ngpus_per_node)
            get_scatter_plot_with_thumbnails_other_domain(args, pca_fit, cmap, actual_assignments, test_dataset, 
                                                          path_to_save=output_path, method="pca", K=args.n_clusters, 
                                                          ngpus_per_node=ngpus_per_node)
            
            umap_reducer = umap.UMAP(n_neighbors=args.num_neighbors, min_dist=args.min_dist_umap, random_state=args.seed)
            umap_fit = umap_reducer.fit_transform(all_embeddings)
            plot_projections(umap_fit, actual_assignments, "UMAP Projection", output_path, cmap, args, ngpus_per_node)
            get_scatter_plot_with_thumbnails(args, umap_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, 
                                             method="umap", K=args.n_clusters, ngpus_per_node=ngpus_per_node)
            get_scatter_plot_with_thumbnails_other_domain(args, umap_fit, cmap, actual_assignments, test_dataset, 
                                                          path_to_save=output_path, method="umap", K=args.n_clusters, 
                                                          ngpus_per_node=ngpus_per_node)
            
            tsne_reducer = TSNE(n_components=2, perplexity=args.num_neighbors, verbose=1, random_state=args.seed, n_iter=1000)
            tsne_fit = tsne_reducer.fit_transform(all_embeddings)
            plot_projections(tsne_fit, actual_assignments, "t-SNE Projection", output_path, cmap, args, ngpus_per_node)
            get_scatter_plot_with_thumbnails(args, tsne_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, 
                                             method="tsne", K=args.n_clusters, ngpus_per_node=ngpus_per_node)
            get_scatter_plot_with_thumbnails_other_domain(args, tsne_fit, cmap, actual_assignments, test_dataset, 
                                                          path_to_save=output_path, method="tsne", K=args.n_clusters, 
                                                          ngpus_per_node=ngpus_per_node)
            
            for i in range(5):
                plot_nearest_neighbors_3x3(args, all_embeddings, test_dataset, example_idx=i, i=i, 
                                           path_to_save=output_path, ngpus_per_node=ngpus_per_node)

def plot_projections(projection, actual_assignments, title, path_to_save, cmap, args, ngpus_per_node, num_neighbors=None, min_dist_umap=None):
    plt.scatter(projection[:, 0], projection[:, 1], c=actual_assignments, cmap=cmap)
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
    
def normalize_img(img, lower=0, upper=1):
    min = img.min()
    if min < 0:
        img = img + np.abs(min)
    max = img.max()
    img = (upper - lower) * (img - img.min()) / (img.max() - img.min()) + lower
    return img

def soft_normalize(img):
    p1, p99 = np.percentile(img, (1, 99))
    normalized = np.clip((img - p1) / (p99 - p1), 0, 1)
    normalized *= 255
    normalized = normalized.astype(np.uint8)
    return normalized

def spectral_clustering(embeddings, n_clusters):
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0)
    spectral_clustering.fit(embeddings)
    y_pred = spectral_clustering.labels_
    return y_pred
                                             
def kmeans_clustering(embeddings, n_clusters):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    d = embeddings.shape[1]
    niter = 300
    verbose = False 
    kmeans = faiss.Kmeans(d, n_clusters, niter=niter, verbose=verbose)
    kmeans.train(embeddings)
    D, I = kmeans.index.search(embeddings, 1)
    return I.flatten()


def get_scatter_plot_with_thumbnails(
    args, embeddings_2d, cmap, labels, test_dataset, path_to_save, method, K, is_mrc=True, ngpus_per_node=1, is_wandb=False
):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    fig.suptitle(f"Scatter Plot Using {method} Projection, K = {K}")
    
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e-1:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap, s=10, alpha=0.8)

    for idx in shown_images_idx:
        tmp = test_dataset[idx]
        if len(tmp) == 2:
            img, _ = tmp
        else:
            img = tmp
        
        # img, _ = test_dataset[idx]
        if isinstance(img, torch.Tensor):
            if img.ndimension() == 3:
                img = img.permute(1, 2, 0).numpy()
            else:
                img = img.numpy()
        # img = img.numpy()
        # print(img.shape)
        img = soft_normalize(img)
        img = (img * 255).astype('uint8') if img.max() <= 1 else img
        
        thumbnail_size = int(rcp["figure.figsize"][0] * 5.0)
        # thumbnail_zoom = 0.5 if dataset_name.lower() == "mnist" else 0.7  # Adjust zoom
        img = functional.to_pil_image(img)
        img = functional.resize(img, thumbnail_size)
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
        plt.savefig(f"{path_to_save}/thumbnail_plot_{method}.png")
    plt.clf()

def get_scatter_plot_with_thumbnails_other_domain(
    args, embeddings_2d, cmap, labels, test_dataset, path_to_save, method, K, is_mrc=True, ngpus_per_node=1, is_wandb=False
):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    fig.suptitle(f"Scatter Plot Using {method} Projection, K = {K}")
    
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e-1:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap, s=10, alpha=0.8)

    for idx in shown_images_idx:
        # Get the opposite modality image
        if args.use_fft:
            img = test_dataset.get_real_image(idx)
        else:
            img = test_dataset.get_fft_image(idx)
        # img, _ = test_dataset[idx]
        if isinstance(img, torch.Tensor):
            if img.ndimension() == 3:
                img = img.permute(1, 2, 0).numpy()
            else:
                img = img.numpy()
                
        img = soft_normalize(img)
        img = (img * 255).astype('uint8') if img.max() <= 1 else img
        
        thumbnail_size = int(rcp["figure.figsize"][0] * 5.0)
        # thumbnail_zoom = 0.5 if dataset_name.lower() == "mnist" else 0.7  # Adjust zoom
        img = functional.to_pil_image(img)
        img = functional.resize(img, thumbnail_size)
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
    
    is_fft = "real" if args.use_fft else "fft"
    if is_wandb:
        wandb.log({f"scatter_plot_{method}_{is_fft}": wandb.Image(plt)})
    elif not args.distributed or args.rank % ngpus_per_node == 0:
        plt.savefig(f"{path_to_save}/thumbnail_plot_{method}_{is_fft}.png")
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
        
def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        end = time.time()
        
        for i, images in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute embeddings
            if isinstance(model, torch.nn.DataParallel):
                encoder = model.module.encoder
            else:
                encoder = model.encoder
            embeddings = encoder(images)
            all_embeddings.append(embeddings)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings

# Function to train and evaluate based on configurations
def start_sweep(test_dataset, all_embeddings, args):
    num_elements = len(test_dataset)
    num_neighbors = [0, int(0.2 * num_elements), int(0.4 * num_elements), int(0.6 * num_elements), int(0.8 * num_elements)]
    min_dist_umap = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
    num_clusters = [2, 5, 10]
    
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'num_neighbors': {'values': num_neighbors},
            'min_dist_umap': {'values': min_dist_umap},
            'num_clusters': {'values': num_clusters}
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="simsiam_clICEification")
    
    def param_sweep(config=None):
        with wandb.init(config=config):
            config = wandb.config
            
            temp_args = args
            # temp_args['num_neighbors'] = config.num_neighbors
            # temp_args['min_dist_umap'] = config.min_dist_umap
            # temp_args['n_clusters'] = config.num_clusters

            actual_assignments = kmeans_clustering(temp_args, all_embeddings, n_clusters=config.num_clusters)
            # UMAP dimensionality reduction
            umap_reducer = umap.UMAP(
                n_neighbors=config.num_neighbors,
                min_dist=config.min_dist_umap,
                random_state=temp_args.seed
            )
            umap_fit = umap_reducer.fit_transform(all_embeddings)

            # t-SNE dimensionality reduction
            tsne_reducer = TSNE(
                n_components=2,
                perplexity=config.num_neighbors,
                random_state=temp_args.seed
            )
            tsne_fit = tsne_reducer.fit_transform(all_embeddings)

            # Visualize UMAP results
            plt.scatter(umap_fit[:, 0], umap_fit[:, 1], c=actual_assignments, cmap='tab10')
            plt.title("UMAP Projection")
            wandb.log({"UMAP Projection": wandb.Image(plt)})
            plt.clf()
            
            get_scatter_plot_with_thumbnails(temp_args, umap_fit, plt.get_cmap('tab10', config.num_clusters), actual_assignments, test_dataset, path_to_save=None, method="umap", K=config.num_clusters, ngpus_per_node=0, is_wandb=True)
            plt.clf()

            # Visualize t-SNE results
            plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1], c=actual_assignments, cmap='tab10')
            plt.title("t-SNE Projection")
            wandb.log({"t-SNE Projection": wandb.Image(plt)})
            plt.clf()
            
            get_scatter_plot_with_thumbnails(temp_args, tsne_fit, plt.get_cmap('tab10', config.num_clusters), actual_assignments, test_dataset, path_to_save=None, method="tsne", K=config.num_clusters, ngpus_per_node=0, is_wandb=True)

            # Log configuration and clustering results
            wandb.log({
                "num_neighbors": config.num_neighbors,
                "min_dist_umap": config.min_dist_umap,
                "num_clusters": config.num_clusters
            })
    wandb.agent(sweep_id, function=param_sweep)
    return
    
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
    
if __name__ == '__main__':
    main()