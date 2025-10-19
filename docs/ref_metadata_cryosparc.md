# `prismpyp metadata_cryosparc`

## Purpose
Convert micrograph `.mrc` files and power spectra `.mrc` files output by **CTFFIND** into `.webp` images, saving them to a new directory.

Additionally, this command parses the outputs of **Patch CTF Estimation** and **CTFFIND4** to retrieve heuristic metrics, including:
- CTF fit  
- Estimated resolution  
- Mean defocus  
- Relative ice thickness  

## Usage
```bash
usage: prismpyp metadata_cryosparc [-h] 
                                   [--patch-ctf-file PATCH_CTF_FILE] 
                                   [--ctffind-file CTFFIND_FILE] 
                                   [--ctffind-dir CTFFIND_DIR]
                                   [--imported-dir IMPORTED_DIR] 
                                   [--output-dir OUTPUT_DIR]
```

## Named Arguments

| Argument | Description |
|-----------|--------------|
| `--patch-ctf-file` | Path to CryoSPARC Patch CTF Estimation job exported `.cs` file |
| `--ctffind-file` | Path to CryoSPARC CTFFIND job exported `.cs` file |
| `--ctffind-dir` | Path to directory containing CTFFIND output `.mrc` files |
| `--imported-dir` | Path to directory containing imported micrographs |
| `--output-dir` | Output directory to save metadata and generated `.webp` images |
