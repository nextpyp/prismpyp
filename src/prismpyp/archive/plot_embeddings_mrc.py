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

# for plotting
import matplotlib.offsetbox as osb

# for resizing images to thumbnails
import torchvision.transforms.functional as functional
from matplotlib import rcParams as rcp

# for retrieving MNIST and CIFAR-10 images
from torchvision import datasets, transforms

import mrc_dataset
import mrcfile
import fft
from glob import glob

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('embeddings', metavar='DIR',
                        help='path to embeddings file')
    parser.add_argument('output_path', metavar='DIR',
                        help='path to output directory')
    parser.add_argument('--add_datetime', action='store_true',
                        help='Add datetime to output directory')
    parser.add_argument('--num_clusters', default=100, type=int,
                        help='Number of clusters to create')
    parser.add_argument('--num_neighbors', default=5, type=int,
                        help='Number of neighbors to find')
    parser.add_argument('--mode', choices=['tsne', 'umap'], default='umap',
                        help='choice of dimensionality reduction technique(default: umap)')
    parser.add_argument('--seed', type=int, default=42,
                        help='file to output max stats')
    parser.add_argument('--min_dist_umap', type=float, default=0.5,
                        help='min distance for umap parameters')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    return parser.parse_args()

def get_scatter_plot_with_thumbnails_mrcs(embeddings_2d, dataset_name="mrc", path_to_data=None, method="umap", K=10, plot_dir=None):
    # Load MNIST or CIFAR-10 data
    dataset = []
    fft_dataset = []
    
    if dataset_name.lower() == "mnist":
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root=".", train=False, download=True, transform=transform)
    elif dataset_name.lower() == "cifar10" or dataset_name.lower() == 'cifar':
        transform = transforms.ToTensor()
        dataset = datasets.CIFAR10(root=".", train=False, download=True, transform=transform)
    else:
        # Load MRC data
        if path_to_data is not None:
            files = glob(path_to_data + "/*.mrc")
            for file in files:
                img = mrcfile.read(file)
                
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=0)
                    
                dataset.append(img)
                fft_img = np.abs(fft.fft2_center(img))
                fft_dataset.append(fft_img)
    
    """Creates a scatter plot with image overlays."""
    # initialize empty figure and add subplot
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
        if np.min(dist) < 2e-2:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    # plot image overlays
    for idx in shown_images_idx:
        img = np.array(fft_dataset[idx].permute(1, 2, 0))
        
        if dataset_name.lower() == "mnist":
            img = img.squeeze(-1) # MNIST is grayscale
        # thumbnail_size = 32 if dataset_name.lower() == "mnist" else 64
        thumbnail_size = int(rcp["figure.figsize"][0] * 2.0)
        # thumbnail_zoom = 0.5 if dataset_name.lower() == "mnist" else 0.7  # Adjust zoom
        img = functional.to_pil_image(img)
        img = functional.resize(img, thumbnail_size)
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),#, zoom=thumbnail_zoom),
            embeddings_2d[idx],
            pad=0.2,
        )
        img_box.patch.set_edgecolor(cmap(actual_assignments[idx]))
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
    if plot_dir is not None:
        plt.savefig(os.path.join(plot_dir, f"scatter_{method}_K-{K}.png"))
        
def main():
    args = get_args()
    cmap = plt.cm.get_cmap('jet')
    
    # Load embeddings
    if args.embeddings:
        if not os.path.isfile(args.embeddings):
            raise ValueError(f"Embeddings file {args.embeddings} does not exist")
    embeddings = torch.load(args.embeddings)
    embeddings = embeddings.cpu().numpy()
    
    is_gpu = False
    if args.gpu == 0:
        is_gpu = True
        
    # Set up for KMeans
    num_clusters = args.num_clusters
    d = embeddings.shape[1]
    # ncentroids = 256
    niter = 300
    verbose = True 
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
    
    # Do dimensionality reduction
    # if args.mode == 'umap':
    #     reducer = umap.UMAP(n_neighbors=args.num_neighbors, min_dist=args.min_dist_umap, random_state=args.seed)
    #     # reduced_embeddings = reducer.fit_transform(embeddings)
    # elif args.mode == 'tsne':
    #     reducer = TSNE(n_components=2, perplexity=args.num_neighbors, verbose=1, random_state=args.seed, n_iter=1000)
    #     # reduced_embeddings = reducer.fit_transform(embeddings)
    # else:
    #     raise ValueError("Invalid mode")
    
    # embeddings_2d = reducer.fit_transform(embeddings)
    # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=actual_assignments, cmap=cmap)
    # plt.colorbar()
    # plt.title("Cluster assignments, K = {}".format(num_clusters))
    # plt.savefig(os.path.join(args.output_path, 'cluster_assignments.png'))
    
    cmap = plt.get_cmap('tab20')
    umap_reducer = umap.UMAP(n_neighbors=args.num_neighbors, min_dist=args.min_dist_umap, random_state=args.seed)
    tsne_reducer = TSNE(n_components=2, perplexity=args.num_neighbors, verbose=1, random_state=args.seed, n_iter=1000)
    dbscan_reducer = DBSCAN(eps=0.5, min_samples=5)
    pca_reducer = PCA(n_components=2)

    umap_fit = umap_reducer.fit_transform(embeddings)
    tsne_fit = tsne_reducer.fit_transform(embeddings)
    dbscan_fit = dbscan_reducer.fit_predict(embeddings)
    pca_fit = pca_reducer.fit_transform(embeddings)

    cmap = plt.get_cmap('tab20')
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    axs[0].scatter(umap_fit[:, 0], umap_fit[:, 1], c=actual_assignments, cmap=cmap)
    axs[0].set_title('UMAP Projection, K = {}'.format(num_clusters))
    axs[1].scatter(tsne_fit[:, 0], tsne_fit[:, 1], c=actual_assignments, cmap=cmap)
    axs[1].set_title('t-SNE Projection, K = {}'.format(num_clusters))
    axs[2].scatter(pca_fit[:, 0], pca_fit[:, 1], c=actual_assignments, cmap=cmap)
    axs[2].set_title('PCA Projection, K = {}'.format(num_clusters))
    
    # plt.title("Cluster assignments, K = {}".format(num_clusters))
    plot_dir = os.path.join(args.output_path, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, 'cluster_assignments.png'))
    
    # get a scatter plot with thumbnail overlays
    path_to_data = "/nfs/bartesaghilab2/sh696/simsiam/small_data_balanced/val" # HARDCODED
    get_scatter_plot_with_thumbnails_mrcs(umap_fit, dataset_name="mrc", path_to_data=path_to_data, method="UMAP", K=10, plot_dir=plot_dir)
    get_scatter_plot_with_thumbnails_mrcs(tsne_fit, dataset_name="mrc", path_to_data=path_to_data, method="t-SNE", K=10, plot_dir=plot_dir)
    
if __name__ == "__main__":
    main()