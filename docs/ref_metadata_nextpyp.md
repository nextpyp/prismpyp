# `prismpyp metadata_nextpyp`

## Purpose
Copy `.webp` files of micrographs and power spectra (as estimated from **CTFFIND4**) from a **nextPYP project** into the metadata directory.

Additionally, this command parses `.pkl` files for pre-processing heuristics, including:

- CTF fit  
- Estimated resolution  
- Mean defocus  

## Usage
```bash
usage: prismpyp metadata_nextpyp [-h] 
                                 [--pkl-path PKL_PATH] 
                                 [--cryosparc-path CRYOSPARC_PATH] 
                                 [--output-dir OUTPUT_DIR]
                                 [--micrographs-list MICROGRAPHS_LIST]
```

## Named Arguments

| Argument | Description |
|-----------|--------------|
| `--pkl-path` | Path to the `.pkl` files |
| `--cryosparc-path` | Path to cryoSPARC Curate Exposure job exported `.cs` file *(optional)* |
| `--output-dir` | Output directory to save metadata and copied `.webp` files |
| `--micrographs-list` | Optional path to a text file containing a list of micrograph names (one per line) to filter the output |
