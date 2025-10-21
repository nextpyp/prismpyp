# `prismpyp intersect`

## Purpose
Compute the **intersection of micrograph embeddings** between images with high-quality features in the real domain and images with high-quality features in the Fourier domain.

This command outputs images with high-quality features in both domains, with the aim of reducing false positive rates.

## Usage
```bash
usage: prismpyp upload_nextpyp [-h] 
                               [--intersection-folder INTERSECTION_FOLDER]
                               [--link-type {hard,soft}]
                               [--mrc-path MRC_PATH]
```

## Named Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `--intersection-folder INTERSECTION_FOLDER` | Path to intersection folder earlier results were saved | — |
| `--link-type {hard,soft}` | Type of filesystem link to create for intersected micrographs (`hard` or `soft`) | — |
| `--mrc-path MRC_PATH` | Path to directory containing the original `.mrc` files | — |
