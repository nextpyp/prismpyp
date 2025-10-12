import argparse
import builtins
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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from prismpyp.simsiam import builder as simsiam_builder
from prismpyp.utils.mrc_dataset import MRCDataset
from prismpyp.utils import parse_args
from prismpyp.core import crystalline_transforms as cryst_xforms

# For dimensionality reduction
import umap
import faiss
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize

# for plotting
import matplotlib.offsetbox as osb

# for resizing images to thumbnails
import torchvision.transforms.functional as functional
from matplotlib import rcParams as rcp

# for parameter sweeps
import wandb

def main(args):
    
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
    model = simsiam_builder.SimSiam(
        args.arch,
        args.dim, args.pred_dim,
        use_checkpoint=False,
        pretrained=False) # Load full pre-trained model instead
    
    print(model)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    
    # Load pre-trained
    if args.feature_extractor_weights:
        if os.path.isfile(args.feature_extractor_weights):
            print("=> loading checkpoint '{}'".format(args.feature_extractor_weights))
            # checkpoint = torch.load(args.pretrained, map_location="cpu")
            checkpoint = torch.load(args.feature_extractor_weights)

            # load pre-trained model
            model.load_state_dict(checkpoint['state_dict'], strict=False)

            args.start_epoch = 0
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.feature_extractor_weights))
        else:
            print("=> no checkpoint found at '{}'".format(args.feature_extractor_weights))
    
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
        
    # Load inference dataset
    if args.use_fft:
        augmentations = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    else:
        augmentations = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    
    test_dataset = MRCDataset(
        args.micrographs_list, 
        transform=transforms.Compose(augmentations),
        is_fft=args.use_fft,
        webp_dir=os.path.join(args.nextpyp_preproc, 'webp'),
        metadata_path=args.metadata_path
    )
        
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    if args.output_path and (not args.distributed or args.rank % ngpus_per_node == 0):
        output_path = os.path.join(args.output_path, 'inference')
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        if args.distributed:
            dist.barrier(device_ids=[args.gpu])
            
    if args.evaluate:
        if not args.embedding_path:
            all_embeddings = validate(test_loader, model, args)
            filepath = os.path.join(output_path, 'embeddings.pth')
            torch.save(all_embeddings, filepath)
            print(f"Embeddings saved to {filepath}")
        else:
            # Use precomputed embeddings
            all_embeddings = torch.load(args.embedding_path)
        
        # Convert embeddings to numpy array
        all_embeddings = all_embeddings.cpu().numpy()
        
        # normalize embeddings
        normed_embeddings = normalize(all_embeddings, axis=1)
        
        # Do KMeans clustering on the embeddings
        actual_assignments = kmeans_clustering(args, normed_embeddings)
        n_clusters = args.n_clusters
                
        # Do UMAP/TSNE/PCA projection on embeddings
        cmap = plt.get_cmap('tab10', n_clusters)
        
        pca_reducer = PCA(n_components=2)
        pca_fit = pca_reducer.fit_transform(normed_embeddings)
        plot_projections(pca_fit, actual_assignments, "PCA Projection", output_path, cmap, args, ngpus_per_node)
        get_scatter_plot_with_thumbnails_real_fft(args, pca_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, 
                                        method="pca", K=n_clusters, ngpus_per_node=ngpus_per_node)

        
        umap_reducer = umap.UMAP(n_components=2, n_neighbors=args.num_neighbors, min_dist=args.min_dist_umap, random_state=args.seed)
        umap_fit = umap_reducer.fit_transform(normed_embeddings)
        plot_projections(umap_fit, actual_assignments, "UMAP Projection", output_path, cmap, args, ngpus_per_node)
        get_scatter_plot_with_thumbnails_real_fft(args, umap_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, 
                                        method="umap", K=n_clusters, ngpus_per_node=ngpus_per_node)

        
        tsne_reducer = TSNE(n_components=2, perplexity=args.num_neighbors, verbose=1, random_state=args.seed, n_iter_without_progress=1000)
        tsne_fit = tsne_reducer.fit_transform(normed_embeddings)
        plot_projections(tsne_fit, actual_assignments, "t-SNE Projection", output_path, cmap, args, ngpus_per_node)
        get_scatter_plot_with_thumbnails_real_fft(args, tsne_fit, cmap, actual_assignments, test_dataset, path_to_save=output_path, 
                                        method="tsne", K=n_clusters, ngpus_per_node=ngpus_per_node)
        
        for i in range(5):
            random_idx = random.randint(0, len(test_dataset) - 1)
            plot_nearest_neighbors_3x3(args, normed_embeddings, test_dataset, example_idx=random_idx, i=i, 
                                        path_to_save=output_path, ngpus_per_node=ngpus_per_node)
        
        
def plot_projections(projection, actual_assignments, title, path_to_save, cmap, args, ngpus_per_node, num_neighbors=None, min_dist_umap=None):
    plt.scatter(projection[:, 0], projection[:, 1], c=actual_assignments, cmap=cmap, s=10, alpha=0.8)
    plt.title(title)
    num_neighbors = args.num_neighbors if num_neighbors is None else num_neighbors
    min_dist_umap = args.min_dist_umap if min_dist_umap is None else min_dist_umap
    title = title.split(" ")[0]
    save_title = f"scatter_plot_{title}.webp"
    
    if not args.distributed or args.rank % ngpus_per_node == 0:
        plt.savefig(os.path.join(path_to_save, save_title))
        
        plt.gca().set_title("")  # Remove title
        plt.axis('off')  # Turn off axes
        plt.gca().legend().remove()  # Remove legend
        new_save_title = f"scatter_plot_{title}_no_labels.webp"
        plt.savefig(os.path.join(path_to_save, new_save_title), bbox_inches='tight', pad_inches=0)
    plt.clf()
    

def soft_normalize(img):
    p1, p99 = np.percentile(img, (1, 99))
    normalized = np.clip((img - p1) / (p99 - p1), 0, 1)
    normalized *= 255
    normalized = normalized.astype(np.uint8)
    return normalized


def hdbscan_clustering(args, embeddings):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
        
    clusterer = AgglomerativeClustering(n_clusters=4, linkage='ward')
    cluster_labels = clusterer.fit(embeddings).labels_
    num_clusters = len(np.unique(cluster_labels))
    
    print(f"Number of clusters: {len(np.unique(cluster_labels))}")
    print(f"Number of clusters (excluding noise): {num_clusters}")
    return cluster_labels, num_clusters

def kmeans_clustering(args, embeddings, n_clusters=None):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
        
    num_clusters = n_clusters if n_clusters is not None else args.n_clusters
    d = embeddings.shape[1]
    # ncentroids = 256
    niter = 300
    verbose = False 
    kmeans = faiss.Kmeans(d, num_clusters, niter=niter, verbose=verbose, spherical=True)
    kmeans.train(embeddings)
    D, I = kmeans.index.search(embeddings, 1)
    
    # The cluster assignments are simply the indices of the nearest centroids
    cluster_assignments = I.flatten()  # Flatten to get a 1D array of cluster assignments

    # Return the cluster assignments
    return cluster_assignments


def get_scatter_plot_with_thumbnails_real_fft(
    args, embeddings_2d, cmap, labels, test_dataset, path_to_save, method, K, is_mrc=True, ngpus_per_node=1, is_wandb=False
):
    from PIL import ImageFilter
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib import offsetbox as osb
    
    def load_img_pair(img_path, webp_dir):
        basename = img_path #os.path.basename(img_path)
        ctf_file = os.path.join(webp_dir, basename + '_ctffit.webp')
        mg_file = os.path.join(webp_dir, basename + '.webp')
        if os.path.exists(ctf_file) and os.path.exists(mg_file):
            mg = Image.open(mg_file)
            ctf = Image.open(ctf_file)
            ctf_to_plot = crop_and_mirror_image(ctf)
            right_half = ctf_to_plot.crop((ctf_to_plot.width // 2, 0, ctf_to_plot.width, ctf_to_plot.height))
            
            return mg, right_half
        else:
            raise FileNotFoundError(f"Missing webp or ctffit file for {basename}")

    if args.nextpyp_preproc is None:
        raise ValueError("No webp directory found. Please provide a valid path.")

    webp_dir = os.path.join(args.nextpyp_preproc, "webp")
    
    # Normalize embeddings
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)
    
    # Find image indices to display
    shown_images_idx = []
    shown_images = np.expand_dims(embeddings_2d[0], axis=0)
    idx_pool = np.random.permutation(len(embeddings_2d))
    
    for i in idx_pool:
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 3e-3:
            continue
        shown_images = np.vstack([shown_images, embeddings_2d[i]])
        shown_images_idx.append(i)
    
    def create_scatter_with_images(img_type="micrograph", suffix="mg"):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        # ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap, s=10, alpha=0.8)
        fig.suptitle(f"Scatter Plot ({img_type.capitalize()}s) using {method}, K = {K}")

        for idx in shown_images_idx:
            img_path = test_dataset.file_paths[idx]
            mg_img, ctf_img = load_img_pair(img_path, webp_dir)
            img = mg_img if img_type == "micrograph" else ctf_img
            # print("img size before resizing: ", img.size)

            if img_type == "micrograph":
                thumbnail_x = int(rcp["figure.figsize"][0] * 5.0)
                thumbnail_y = int(rcp["figure.figsize"][1] * 5.0)
            else:
                thumbnail_x = int(rcp["figure.figsize"][0] * 9.0)
                thumbnail_y = int(rcp["figure.figsize"][1] * 9.0)
                
            if isinstance(img, Image.Image):
                img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                if img_type == "micrograph":
                    img.thumbnail((thumbnail_x, thumbnail_x), Image.LANCZOS)
                else:
                    img.thumbnail((thumbnail_x, thumbnail_y), Image.LANCZOS)
                    # print("img size after resizing: ", img.size)
            img_array = np.array(img)
            img_box = osb.AnnotationBbox(
                osb.OffsetImage(img_array, cmap="gray"),
                embeddings_2d[idx],
                pad=0.2
            )
            # img_box.patch.set_edgecolor(cmap(labels[idx]))
            ax.add_artist(img_box)

        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim(embeddings_2d[:, 0].min() - 0.1, embeddings_2d[:, 0].max() + 0.1)
        ax.set_ylim(embeddings_2d[:, 1].min() - 0.1, embeddings_2d[:, 1].max() + 0.1)

        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f'Class {i}') for i in range(K)]
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.1, 1.05), title="Classes")

        if is_wandb:
            wandb.log({f"scatter_plot_{method}_{suffix}": wandb.Image(fig)})
        elif not args.distributed or args.rank % ngpus_per_node == 0:
            fig.savefig(f"{path_to_save}/thumbnail_plot_{method}_{suffix}.webp")
            ax.axis('off')
            ax.set_title("")
            ax.legend().remove()
            fig.savefig(f"{path_to_save}/thumbnail_plot_{method}_{suffix}_no_labels.webp", bbox_inches='tight', pad_inches=0, transparent=True)

        plt.close(fig)
    
    # Create both plots
    create_scatter_with_images("micrograph", "mg")
    create_scatter_with_images("ctf", "ps")


def get_scatter_plot_with_thumbnails(
    args, embeddings_2d, cmap, labels, test_dataset, path_to_save, method, K, is_mrc=True, ngpus_per_node=1, is_wandb=False
):
    # Use nextpyp-postprocessed .webp micrograph thumbnails for real images
    if args.nextpyp_preproc is not None:
        webp_dir =  os.path.join(args.nextpyp_preproc, "webp")
    
    # Normalize embeddings
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)
    
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111)
    fig.suptitle(f"Scatter Plot Using {method} Projection, K = {K}")
    
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.expand_dims(embeddings_2d[0], axis=0)
    
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 1e-2:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap, s=10, alpha=0.8)

    for idx in shown_images_idx:
        if args.nextpyp_preproc is not None:
            webp_dir = os.path.join(args.nextpyp_preproc, 'webp')
            img_path = test_dataset.file_paths[idx]
            basename = os.path.basename(img_path)
            ctf_file = os.path.join(webp_dir, basename[:-4] + '_ctffit.webp')
            mg_file = os.path.join(webp_dir, basename[:-4] + '.webp')
            
            if os.path.exists(ctf_file) and os.path.exists(mg_file):
                ctf_img = Image.open(ctf_file)
                mg_img = Image.open(mg_file)
                img = crop_and_stitch_imgs(ctf_img, mg_img)
            else:
                raise ValueError("No webp directory found. Please provide a valid path.")
        else:
            raise ValueError("No webp directory found. Please provide a valid path.")
        
        
        thumbnail_x = int(rcp["figure.figsize"][0] * 9.0)
        thumbnail_y = int(rcp["figure.figsize"][1] * 9.0)
        
        # thumbnail_zoom = 0.5 if dataset_name.lower() == "mnist" else 0.7  # Adjust zoom
        if isinstance(img, Image.Image):
            sharpened = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            img = sharpened.resize((thumbnail_x, thumbnail_y), Image.LANCZOS)
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=1))
            # img = ImageEnhance.Contrast(img)
        elif isinstance(img, np.ndarray) or isinstance(img, torch.Tensor):
            img = functional.to_pil_image(img)
            img = functional.resize(img, (thumbnail_x, thumbnail_y), interpolation=transforms.InterpolationMode.LANCZOS)
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
    
    # Create a legend mapping each class to a color
    handles = []
    for i in range(K):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f'Class {i}'))
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.1, 1.05), title="Classes")
    
    # Set aspect ratio and remove unnecessary whitespace
    if is_wandb:
        wandb.log({f"scatter_plot_{method}": wandb.Image(plt)})
    elif not args.distributed or args.rank % ngpus_per_node == 0:
        plt.savefig(f"{path_to_save}/thumbnail_plot_{method}.webp")
        
        ax.axis('off')  # Turn off axes
        ax.set_title("")  # Remove title
        ax.legend().remove()  # Remove legend
        plt.savefig(f"{path_to_save}/thumbnail_plot_{method}_no_labels.webp", bbox_inches='tight', pad_inches=0)
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
            webp_dir = os.path.join(args.nextpyp_preproc, 'webp')
            img_path = test_dataset.file_paths[plot_idx]
            basename = img_path #os.path.basename(img_path)
            ctf_file = os.path.join(webp_dir, basename + '_ctffit.webp')
            mg_file = os.path.join(webp_dir, basename + '.webp')
            
            if os.path.exists(ctf_file) and os.path.exists(mg_file):
                ctf_img = Image.open(ctf_file)
                mg_img = Image.open(mg_file)
                img = crop_and_stitch_imgs(ctf_img, mg_img)
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
            plt.savefig(f"{path_to_save}/nearest_neighbors_{i + 1}.webp")
        
        
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
            if isinstance(model, torch.nn.DataParallel) or \
                isinstance(model, torch.nn.parallel.DistributedDataParallel):
                backbone = model.module.backbone
                reducer = model.module.reducer
            else:
                backbone = model.backbone
                reducer = model.reducer
            embeddings = backbone(images)
            embeddings = embeddings.view(embeddings.size(0), -1)
            embeddings = reducer(embeddings)
            
            all_embeddings.append(embeddings)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings

def crop_and_stitch_imgs(image1, image2):
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

    return new_image


def crop_and_mirror_image(image1):
    image1_cropped = image1.crop((0, 0, image1.width, image1.height // 2))
    image1_flipped_lr = image1_cropped.transpose(Image.FLIP_LEFT_RIGHT)
    image1_flipped_tb = image1_flipped_lr.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Create a new image with combined width and the maximum height
    new_width = max(image1_cropped.size[0], image1_flipped_tb.size[0])
    new_height = image1_cropped.size[1] + image1_flipped_tb.size[1]

    # Create a new blank image (white background)
    new_image = Image.new("RGB", (new_width, new_height), (255, 255, 255))

    # Paste the first image at the left side of the new image
    new_image.paste(image1_cropped, (0, 0))

    # Paste the second image at the bottom half of the new image
    new_image.paste(image1_flipped_tb, (0, image1_cropped.size[1]))
    
    return new_image

    
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
    main(add_args().parse_args())