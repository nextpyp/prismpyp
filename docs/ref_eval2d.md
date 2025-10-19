# `prismpyp eval2d`

## Purpose
Evaluate and visualize **2D embeddings** or model features from trained SimSiam networks.

This command can:

- Run 2D embedding evaluation on real-space or Fourier-space inputs  
- Optionally resume from precomputed embeddings  
- Perform clustering and UMAP dimensionality reduction  
- Save visualizations, plots, and `.webp` image thumbnails  

## Usage
```bash
usage: prismpyp eval2d [-h] [--output-path DIR] [--metadata-path METADATA_PATH] [--embedding-path [EMBEDDING_PATH]] [-a ARCH] [-j N] [--epochs N]
                       [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [--feature-extractor-weights PATH]
                       [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
                       [--multiprocessing-distributed] [--dim DIM] [--pred-dim PRED_DIM] [--fix-pred-lr] [--use-fft] [--downsample DOWNSAMPLE]
                       [--pixel-size PIXEL_SIZE] [--size SIZE] [--evaluate] [--n-clusters N_CLUSTERS] [--num-neighbors NUM_NEIGHBORS]
                       [--min-dist-umap MIN_DIST_UMAP] [--n-components N_COMPONENTS] [--nextpyp-preproc NEXTPYP_PREPROC] [--zip-images]
```

## Named Arguments

### Commonly Changed Arguments
| Argument | Description | Default |
|-----------|--------------|----------|
| `--output-path DIR` | Path to output directory | — |
| `--metadata-path METADATA_PATH` | Path to metadata file | — |
| `--embedding-path EMBEDDING_PATH` | Optional path to precomputed embeddings | — |
| `--feature-extractor-weights PATH` | Path to pre-trained feature extractor weights | `none` |
| `--dim DIM` | Feature dimension | `2048` |
| `--pred-dim PRED_DIM` | Hidden dimension of the predictor | `512` |
| `--fix-pred-lr` | Fix learning rate for the predictor | — |
| `--use-fft` | Use FFT of the image as input | — |
| `--evaluate` | Evaluate model on validation set | — |
| `--n-clusters N_CLUSTERS` | Number of clusters for KMeans | — |
| `--num-neighbors NUM_NEIGHBORS` | Number of neighbors for UMAP | — |
| `--min-dist-umap MIN_DIST_UMAP` | Minimum distance for UMAP | — |
| `--n-components N_COMPONENTS` | Number of UMAP components | — |


**Available Architectures:**  
`resnet18`, `resnet34`, `resnet50`

