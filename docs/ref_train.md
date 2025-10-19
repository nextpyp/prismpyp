# `prismpyp train`

## Purpose
Run **SimSiam** training on either the **real-space** inputs or the **Fourier-space** inputs.

When executed, this command creates an output subdirectory named `checkpoints` that contains:

- **Best** and **last** model weights  
- **Total loss** function plot  
- **Collapse** plot  
- `training_config.yaml` — a record of all command-line arguments used during the run  

---

## Usage
```bash
usage: prismpyp train [-h]
                      [--metadata-path METADATA_PATH]
                      [--output-path OUTPUT_PATH]
                      [-a {resnet18,resnet34,resnet50}]
                      [--epochs N]
                      [-b N, --batch-size N]
                      [--lr LR]
                      [--resume PATH]
                      [--dist-url DIST_URL]
                      ...
```

---

## Named Arguments

### Commonly Changed Arguments
| Argument | Description | Default |
|-----------|--------------|----------|
| `--output-path` | Path to output directory | — |
| `--metadata-path` | Path to metadata file | — |
| `-a`, `--arch` | Model architecture.<br>Options: `resnet18`, `resnet34`, `resnet50` | `resnet50` |
| `--epochs N` | Number of total epochs to run | — |
| `-b N`, `--batch-size N` | Mini-batch size (total across GPUs) | `512` |
| `--lr LR`, `--learning-rate LR` | Initial (base) learning rate | — |
| `--resume PATH` | Path to latest checkpoint | `none` |
| `--dist-url DIST_URL` | URL used to set up distributed training | — |

---

### Less Commonly Changed Arguments
| Argument | Description | Default |
|-----------|--------------|----------|
| `-j`, `--workers` | Number of data loading workers | `32` |
| `--momentum` | Momentum of SGD solver | — |
| `--wd W`, `--weight-decay W` | Weight decay | `1e-4` |
| `-p N`, `--print-freq N` | Print frequency | `10` |
| `--feature-extractor-weights PATH` | Path to pre-trained feature extractor weights | `none` |
| `--classifier-weights PATH` | Path to pre-trained classifier weights | `none` |
| `--world-size WORLD_SIZE` | Number of nodes for distributed training | — |
| `--rank RANK` | Node rank for distributed training | — |
| `--dist-backend DIST_BACKEND` | Distributed backend | — |
| `--seed SEED` | Random seed | — |
| `--gpu GPU` | GPU ID to use | — |
| `--multiprocessing-distributed` | Use multi-processing distributed training | — |
| `--dim DIM` | Feature dimension | `2048` |
| `--pred-dim PRED_DIM` | Hidden dimension of the predictor | `512` |
| `--fix-pred-lr` | Fix learning rate for predictor | — |
| `--use-fft` | Use FFT of the image as input | — |
| `--downsample DOWNSAMPLE` | Downsample the image | — |
| `--pixel-size PIXEL_SIZE` | Pixel size of the image | — |
| `--size SIZE` | Size of the image in pixels (before downsampling) | — |
| `--conf-thresh CONF_THRESH` | Confidence threshold for filtering | — |
| `--add-datetime` | Append datetime to output directory name | — |
| `--evaluate` | Evaluate model on validation set | — |
| `--n-clusters N_CLUSTERS` | Number of clusters for KMeans | — |
| `--num-neighbors NUM_NEIGHBORS` | Number of neighbors for UMAP | — |
| `--min-dist-umap MIN_DIST_UMAP` | Minimum distance for UMAP | — |
| `--n-components N_COMPONENTS` | Number of UMAP components | — |
| `--nextpyp-preproc NEXTPYP_PREPROC` | Path to NextPYP project pre-processing directory | — |
| `--zip-images` | Save zipped image thumbnails | — |

