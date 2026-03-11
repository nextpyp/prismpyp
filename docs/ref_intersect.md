# `prismpyp intersect`

## Purpose
Compute the **intersection of micrograph embeddings** between images with high-quality features in the real domain and images with high-quality features in the Fourier domain.

This command outputs images with high-quality features in both domains, with the aim of reducing false positive rates.

## Usage
```bash
usage: prismpyp intersect [-h] 
                          [--real-parquet-file REAL_PARQUET_FILE] 
                          [--fft-parquet-file FFT_PARQUET_FILE]
                          [--good-real-classes GOOD_REAL_CLASSES [GOOD_REAL_CLASSES ...]]
                          [--good-fft-classes GOOD_FFT_CLASSES [GOOD_FFT_CLASSES ...]]
                          [--output-folder OUTPUT_FOLDER] 
                          [--link-type {hard,soft}]
                          [--webp-path WEBP_PATH]
```

## Named Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `--real-parquet-file REAL_PARQUET_FILE` | Path to real `.parquet` file | — |
| `--fft-parquet-file FFT_PARQUET_FILE` | Path to FFT `.parquet` file | — |
| `--good-real-classes GOOD_REAL_CLASSES [GOOD_REAL_CLASSES ...]` | 
                        Space-separated list of real domain cluster IDs to treat as good (e.g. --good-real-classes 2 5 7) | - |
| `--good-fft-classes GOOD_FFT_CLASSES [GOOD_FFT_CLASSES ...]` | 
                        Space-separated list of Fourier domain cluster IDs to treat as good (e.g. --good-fft-classes 1 8 9) | - |
| `--output-folder OUTPUT_FOLDER` | Path to output folder for saving intersection results | — |
| `--link-type {hard,soft}` | Type of filesystem link to create for intersected micrographs (`hard` or `soft`) | — |
| `--webp-path WEBP_PATH` | Path to directory containing the original `.webp` files | — |
