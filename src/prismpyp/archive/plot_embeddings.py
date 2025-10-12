import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np

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

import umap
import faiss
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from mrc_dataset import MRCDataset
import parse_args
import fft

# for plotting
import matplotlib.offsetbox as osb

# for resizing images to thumbnails
import torchvision.transforms.functional as functional
from matplotlib import rcParams as rcp

import wandb
        
def main():
    args = parse_args.get_args()
    # cmap = plt.cm.get_cmap('jet')
    
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if args.output_path and (not args.distributed or args.rank % ngpus_per_node == 0):
        output_path = os.path.join(args.output_path, 'inference')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if args.distributed:
            dist.barrier(device_ids=[args.gpu])
    
    # Load dataset
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    test_dataset = MRCDataset(
                        args.data, 
                        transform=transforms.Compose([
                            transforms.Grayscale(num_output_channels=3),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]),
                        is_fft=args.use_fft,
                        downsample=args.downsample,
                        pixel_size=args.pixel_size
                        )
    
    # Load embeddings
    if args.embedding_path is not None:
        if args.embedding_path:
            if not os.path.isfile(args.embedding_path):
                raise ValueError(f"Embeddings file {args.embedding_path} does not exist")
        embeddings = torch.load(args.embedding_path)
        embeddings = embeddings.cpu().numpy()
    else:
        raise ValueError("Embeddings file not provided")
    
    DO_ABLATIONS = args.do_ablations
    if DO_ABLATIONS:
        # Launch the sweep agent
        start_sweep(test_dataset, embeddings, args)
    else:
        # Do KMeans clustering on the embeddings
        actual_assignments = kmeans_clustering(args, embeddings)
        
        # Do UMAP/TSNE/PCA projection on embeddings
        cmap = plt.get_cmap('tab10', args.n_clusters)
        
        pca_reducer = PCA(n_components=2)
        pca_fit = pca_reducer.fit_transform(embeddings)
        plot_projections(pca_fit, actual_assignments, "PCA Projection", output_path, cmap, args, ngpus_per_node)
        get_scatter_plot_with_thumbnails(args, pca_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, method="pca", K=args.n_clusters, ngpus_per_node=ngpus_per_node)
        get_scatter_plot_with_thumbnails_other_domain(args, pca_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, method="pca", K=args.n_clusters, ngpus_per_node=ngpus_per_node)
        
        umap_reducer = umap.UMAP(n_neighbors=args.num_neighbors, min_dist=args.min_dist_umap, random_state=args.seed)
        umap_fit = umap_reducer.fit_transform(embeddings)
        plot_projections(umap_fit, actual_assignments, "UMAP Projection", output_path, cmap, args, ngpus_per_node)
        get_scatter_plot_with_thumbnails(args, umap_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, method="umap", K=args.n_clusters, ngpus_per_node=ngpus_per_node)
        get_scatter_plot_with_thumbnails_other_domain(args, umap_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, method="umap", K=args.n_clusters, ngpus_per_node=ngpus_per_node)
        
        tsne_reducer = TSNE(n_components=2, perplexity=args.num_neighbors, verbose=1, random_state=args.seed, n_iter=1000)
        tsne_fit = tsne_reducer.fit_transform(embeddings)
        plot_projections(tsne_fit, actual_assignments, "t-SNE Projection", output_path, cmap, args, ngpus_per_node)
        get_scatter_plot_with_thumbnails(args, tsne_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, method="tsne", K=args.n_clusters, ngpus_per_node=ngpus_per_node)
        get_scatter_plot_with_thumbnails_other_domain(args, tsne_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, method="tsne", K=args.n_clusters, ngpus_per_node=ngpus_per_node)
        
        for i in range(5):
            plot_nearest_neighbors_3x3(args, embeddings, test_dataset, example_idx=i, i=i, path_to_save=output_path, ngpus_per_node=ngpus_per_node)

def plot_projections(projection, actual_assignments, title, path_to_save, cmap, args, ngpus_per_node, num_neighbors=None, min_dist_umap=None):
    plt.scatter(projection[:, 0], projection[:, 1], c=actual_assignments, cmap=cmap)
    plt.title(title)
    num_neighbors = args.num_neighbors if num_neighbors is None else num_neighbors
    min_dist_umap = args.min_dist_umap if min_dist_umap is None else min_dist_umap
    title = title.split(" ")[0]
    save_title = f"scatter_plot_{title}.png"
    
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

def kmeans_clustering(args, embeddings, n_clusters=None):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
        
    num_clusters = n_clusters if n_clusters is not None else args.n_clusters
    d = embeddings.shape[1]
    # ncentroids = 256
    niter = 300
    verbose = False 
    kmeans = faiss.Kmeans(d, num_clusters, niter=niter, verbose=verbose)
    kmeans.train(embeddings)
    D, I = kmeans.index.search(embeddings, 1)

    kmeans_centroids = kmeans.centroids
    spectral_clustering = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=0)
    spectral_clustering.fit(kmeans_centroids)
    y_pred = spectral_clustering.labels_
    num_clusters_ = len(set(y_pred))
    print("Predicted number of clusters: {}\nActual number of clusters: {}".format(num_clusters, num_clusters_))

    # Assign points to actual clusters
    actual_assignments = []
    for i in I:
        final_pred = y_pred[i[0]]
        actual_assignments.append(final_pred)
    actual_assignments = np.array(actual_assignments)
    return actual_assignments

# def get_scatter_plot_with_thumbnails(embeddings_2d, cmap, actual_assignments, dataset_name="mnist", path_to_save=None, method="umap", K=10):
def get_scatter_plot_with_thumbnails(
    args, embeddings_2d, cmap, labels, test_dataset, path_to_save, method, K, is_mrc=True, ngpus_per_node=1, is_wandb=False
):
    """
    Create a scatter plot with thumbnail images.

    Args:
        embeddings: 2D array of shape (num_samples, 2), the reduced dimensionality data.
        cmap: Colormap for assigning colors to labels.
        labels: Array of class labels for the data points.
        test_dataset: Dataset object containing the test images (MNIST or CIFAR).
        path_to_save: Path to save the generated plot.
        method: Dimensionality reduction method used (e.g., 'umap', 'tsne', 'pca').
        K: Number of clusters.
    """
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
        img = img.permute(1, 2, 0).numpy() if img.ndimension() == 3 else img.numpy()
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
    is_fft = "fft" if args.use_fft else "real"
    if is_wandb:
        wandb.log({f"scatter_plot_{method}_{is_fft}": wandb.Image(plt)})
    elif not args.distributed or args.rank % ngpus_per_node == 0:
        plt.savefig(f"{path_to_save}/thumbnail_plot_{method}_{is_fft}.png")
    plt.clf()

def get_scatter_plot_with_thumbnails_other_domain(
    args, embeddings_2d, cmap, labels, test_dataset, path_to_save, method, K, is_mrc=True, ngpus_per_node=1, is_wandb=False
):
    """
    Create a scatter plot with thumbnail images.

    Args:
        embeddings: 2D array of shape (num_samples, 2), the reduced dimensionality data.
        cmap: Colormap for assigning colors to labels.
        labels: Array of class labels for the data points.
        test_dataset: Dataset object containing the test images (MNIST or CIFAR).
        path_to_save: Path to save the generated plot.
        method: Dimensionality reduction method used (e.g., 'umap', 'tsne', 'pca').
        K: Number of clusters.
    """
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
        img = img.permute(1, 2, 0).numpy() if img.ndimension() == 3 else img.numpy()
        if args.use_fft:
            img = fft.iht2_center(img)
        else:
            img = fft.ht2_center(img)
            img = np.abs(img)
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
    is_fft = "fft" if args.use_fft else "real"
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

# Function to train and evaluate based on configurations
def start_sweep(test_dataset, all_embeddings, args):
    num_elements = len(test_dataset)
    num_neighbors = [int(0.2 * num_elements), int(0.4 * num_elements), int(0.6 * num_elements), int(0.8 * num_elements)]
    min_dist_umap = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
    num_clusters = [2, 4, 6, 8, 10]
    
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'num_neighbors': {'values': num_neighbors},
            'min_dist_umap': {'values': min_dist_umap},
            'num_clusters': {'values': num_clusters},
            'is_fft' : {'value': args.use_fft},
        },
    }
    sweep_id_name = "new_simsiam_clICEification_fft" if args.use_fft else "new_simsiam_clICEification_real"
    print(f"Starting sweep with name: {sweep_id_name}")
    sweep_id = wandb.sweep(sweep_config, project=sweep_id_name)
    
    def param_sweep(config=None):
        with wandb.init(config=config):
            config = wandb.config
            
            temp_args = args
            
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
            get_scatter_plot_with_thumbnails_other_domain(temp_args, umap_fit, plt.get_cmap('tab10', config.num_clusters), actual_assignments, test_dataset, path_to_save=None, method="umap", K=config.num_clusters, ngpus_per_node=0, is_wandb=True)
            plt.clf()

            # Visualize t-SNE results
            plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1], c=actual_assignments, cmap='tab10')
            plt.title("t-SNE Projection")
            wandb.log({"t-SNE Projection": wandb.Image(plt)})
            plt.clf()
            
            get_scatter_plot_with_thumbnails(temp_args, tsne_fit, plt.get_cmap('tab10', config.num_clusters), actual_assignments, test_dataset, path_to_save=None, method="tsne", K=config.num_clusters, ngpus_per_node=0, is_wandb=True)
            plt.clf()
            get_scatter_plot_with_thumbnails_other_domain(temp_args, tsne_fit, plt.get_cmap('tab10', config.num_clusters), actual_assignments, test_dataset, path_to_save=None, method="tsne", K=config.num_clusters, ngpus_per_node=0, is_wandb=True)
            plt.clf()
            
            # Log configuration and clustering results
            wandb.log({
                "num_neighbors": config.num_neighbors,
                "min_dist_umap": config.min_dist_umap,
                "num_clusters": config.num_clusters
            })
    wandb.agent(sweep_id, function=param_sweep)
    return
  
if __name__ == "__main__":
    main()