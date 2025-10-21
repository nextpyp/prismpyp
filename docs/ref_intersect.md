# `prismpyp intersect`

## Purpose
Compute the **intersection of micrograph embeddings** between images with high-quality features in the real domain and images with high-quality features in the Fourier domain.

This command outputs images with high-quality features in both domains, with the aim of reducing false positive rates.

## Usage
```bash
usage: prismpyp intersect [-h] 
                          [--parquet-files PARQUET_FILES [PARQUET_FILES ...]] 
                          [--output-folder OUTPUT_FOLDER] 
                          [--link-type {hard,soft}]
                          [--webp-path WEBP_PATH]
```

## Named Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `--parquet-files PARQUET_FILES [PARQUET_FILES ...]` | List of `.parquet` files containing embeddings to intersect | — |
| `--output-folder OUTPUT_FOLDER` | Path to output folder for saving intersection results | — |
| `--link-type {hard,soft}` | Type of filesystem link to create for intersected micrographs (`hard` or `soft`) | — |
| `--webp-path WEBP_PATH` | Path to directory containing the original `.webp` files | — |
