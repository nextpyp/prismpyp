# 🔮 prismPYP: Power-spectrum and image domain learning for self-supervised micrograph evaluation

prismPYP is a data-driven pipeline for classifying cryo-EM micrographs based on both real-space and Fourier-space features. The goal is to uncover image quality categories such as vitreous ice, crystalline ice, contaminants, and support film without using labels. During training, it learns embeddings for both real-space micrographs and Fourier-space power spectra images such that similar-looking images are grouped together and dissimilar-looking images are located far apart. Through 2D and 3D visualization, users can select images in both domains that exhibit high-quality features which are critical for high-resolution structure determination. The "good" images in both domains are used to filter the images in the dataset and keep only those with high-quality features in both domains.

For more information, read the documentation (insert link here when available).

The code uses a [PyTorch implementation of SimSiam](https://github.com/facebookresearch/simsiam) distributed under the [Attribution-NonCommercial 4.0 International](https://github.com/facebookresearch/simsiam/blob/main/LICENSE) license.

## Zenodo Files

The [Zenodo link](https://doi.org/10.5281/zenodo.17161604) for this project  contains the following files:
* ```model_weights.tar.gz```: Trained model weights for the real domain input (```real_model_best.pth.tar```) and the Fourier domain input (```fft_model_best.pth.tar```)
* ```fft_good_export.parquet```: Data points that have high-quality features in the Fourier domain
* ```real_good_export.parquet```: Data points that have high-qualtiy features in the real domain

By taking the intersection of ```fft_good_export.parquet``` and ```real_good_export.parquet```, you can obtain the 862 micrographs that were used to obtain the 2.9&nbsp;Å structure in the paper.