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

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.builder
import mrc_dataset as dataset_zoo
import parse_args

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
        torch.distributed.barrier(device_ids=[args.gpu])
    
    # Create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)
    
    # Load pre-trained
    if args.feature_extractor_weights:
        if os.path.isfile(args.feature_extractor_weights):
            print("=> loading checkpoint '{}'".format(args.feature_extractor_weights))
            # checkpoint = torch.load(args.pretrained, map_location="cpu")
            checkpoint = torch.load(args.feature_extractor_weights)

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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    
    test_dataset = dataset_zoo.MRCDataset(
                    args.data, 
                    transform=transforms.Compose([
                        # transforms.Grayscale(num_output_channels=3),
                        transforms.Resize(224),
                        # transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]),
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
            os.makedirs(output_path)
        if args.distributed:
            dist.barrier(device_ids=[args.gpu])
    
    DO_ABLATIONS = args.do_ablations
    
    data_for_export = {}
    
    if args.evaluate:
        # Do KMeans clustering on the embeddings
        metadata_without_name = test_dataset.metadata.drop(columns=['micrograph_name', 'avg_motion']) # Drop the name column
        metadata_array = metadata_without_name.to_numpy()
        print(metadata_array.shape)
        actual_assignments = kmeans_clustering(metadata_array, args.n_clusters)
        data_for_export['cluster_id'] = actual_assignments
        
        # pca_reducer = PCA(n_components=3)
        # pca_fit = pca_reducer.fit_transform(metadata_array)
        # data_for_export["pca_fit_x"] = pca_fit[:, 0]
        # data_for_export["pca_fit_y"] = pca_fit[:, 1]
        # data_for_export["pca_fit_z"] = pca_fit[:, 2]
        
        # umap_reducer = umap.UMAP(n_components=3, n_neighbors=args.num_neighbors, min_dist=args.min_dist_umap, random_state=args.seed)
        # umap_fit = umap_reducer.fit_transform(metadata_array)
        # data_for_export["umap_fit_x"] = umap_fit[:, 0]
        # data_for_export["umap_fit_y"] = umap_fit[:, 1]
        # data_for_export["umap_fit_z"] = umap_fit[:, 2]
        
        # tsne_reducer = TSNE(n_components=3, perplexity=args.num_neighbors, verbose=1, random_state=args.seed, n_iter=1000)
        # tsne_fit = tsne_reducer.fit_transform(metadata_array)
        # data_for_export["tsne_fit_x"] = tsne_fit[:, 0]
        # data_for_export["tsne_fit_y"] = tsne_fit[:, 1]
        # data_for_export["tsne_fit_z"] = tsne_fit[:, 2]
            
        # Save images to zip file
        if os.path.exists(os.path.join(output_path, 'thumbnail_images')):
            shutil.rmtree(os.path.join(output_path, 'thumbnail_images'))
        os.makedirs(os.path.join(output_path, 'thumbnail_images'))
        
        # Concatenate *.ctffit.webp and *.webp images into one image
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
                img.save(os.path.join(output_path, 'thumbnail_images', basename + '.jpg'), 
                         "JPEG", quality=100, optimize=True, progressive=True)
        
        # Zip the folder that we just made
        if args.output_path and (not args.distributed or args.rank % ngpus_per_node == 0):
            dest = os.path.join(output_path, 'zipped_thumbnail_images')
            src = os.path.join(output_path, 'thumbnail_images')
            shutil.make_archive(dest, 'gztar', src)
            shutil.rmtree(src)
            print(f"Thumbnail images saved to {dest}")
            
        # Save metadata in prep for exporting
        data_for_export['image_thumbnails'] = []
        data_for_export['micrograph_name'] = []
        for idx in test_dataset.file_paths:
            data_for_export['image_thumbnails'].append(idx)
            data_for_export['micrograph_name'].append(os.path.basename(idx)[:-4])
        
        # Join metadata columns
        data_for_export_df = pd.DataFrame(data_for_export)
        data_for_export_df = data_for_export_df.merge(test_dataset.metadata, on='micrograph_name', how='inner')
        data_for_export_df['embeddings'] = metadata_array.tolist()
        print(data_for_export_df.columns)
        
        # Save data_for_export as zip file
        if args.output_path and (not args.distributed or args.rank % ngpus_per_node == 0):
            path_to_save = os.path.join(output_path, 'data_for_export.parquet.zip')
            data_for_export_df.to_parquet(path_to_save, compression='gzip')
            print(f"Data for export saved to {path_to_save}")

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