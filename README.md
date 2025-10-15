# Self-Supervised Learning for Automatic Cryo-EM Micrograph Exploration

This repository implements a SimSiam-based self-supervised pipeline for classifying cryo-EM micrographs based on both real-space and Fourier-space features. The goal is to uncover image quality categories such as vitreous ice, crystalline ice, contaminants, and support film without using labels. 

The code uses a [PyTorch implementation of SimSiam](https://github.com/facebookresearch/simsiam) distributed under the [Attribution-NonCommercial 4.0 International](https://github.com/facebookresearch/simsiam/blob/main/LICENSE) license.

## 🧪 Download the test data
We will use the ```example_data.tar.gz``` file containing micrograph and power spectra images, and metadata from EMPIAR-10379 as an example test case of self-supervised micrograph sorting. The file is available on [Zenodo](https://doi.org/10.5281/zenodo.17161604). 


Select an appropiate location on your file system, create a new folder named ```example_data```, download and unpack the files from zenodo into it:
```bash
mkdir example_data
tar -xvzf example_data.tar.gz -C example_data
```

This command extracts the data into a folder ```example_data``` with the following folders and files:
* ```pkl``` contains the metadata from pre-processing EMPIAR-10379 in nextPYP
* ```webp``` contains ```512 x 512``` images of the micrographs and their corresponding power spectra
* The ```.cs``` file contains metadata from the Manual Curation job in CryoSPARC
* The ```.toml``` file contains microscope data such as the pixel size of the micrographs
* The ```.micrographs``` file contains a list of micrographs that are present in the data, without any file extensions

The data should be structured like this:
```
example_data/
    |
    |- pkl/
    |   |- file1.pkl
    |   |- file2.pkl
    |   |- ...
    |- webp/
    |   |- file1.webp
    |   |- file1_ctffit.webp
    |   |- file2.webp
    |   |- file2_ctffit.webp
    |   |- ...
    |- J7_exposures_accepted_exported.cs
    |- sp-preprocessing-fhgRaEnEqUsEFrUj.micrographs
    |- .pyp_config.toml # This file will only be visible in the directory if you use 'ls -al' to show all files
```

The following instructions assume a directory structure similar to the one above. If your directory structure is different, please note that you may need to change the commands in order to run them properly.

## Zenodo Files

The Zenodo link for this project (https://doi.org/10.5281/zenodo.17161604) contains the following files:
* ```model_weights.tar.gz```: Trained model weights for the real domain input (```real_model_best.pth.tar```) and the Fourier domain input (```fft_model_best.pth.tar```)
* ```fft_good_export.parquet```: Data points that have high-quality features in the Fourier domain
* ```real_good_export.parquet```: Data points that have high-qualtiy features in the real domain

By taking the intersection of ```fft_good_export.parquet``` and ```real_good_export.parquet```, you can obtain the 862 micrographs that were used to obtain the 2.9A structure in the paper.