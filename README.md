# 🔮 prismPYP: Power-spectrum and image domain learning for self-supervised micrograph evaluation

prismPYP is a data-driven pipeline for classifying cryo-EM micrographs based on both real-space and Fourier-space features. The goal is to uncover image quality categories such as vitreous ice, crystalline ice, contaminants, and support film without using labels. During training, it learns embeddings for both real-space micrographs and Fourier-space power spectra images such that similar-looking images are grouped together and dissimilar-looking images are located far apart. Through 2D and 3D visualization, users can select images in both domains that exhibit high-quality features which are critical for high-resolution structure determination. The "good" images in both domains are used to filter the images in the dataset and keep only those with high-quality features in both domains.

For more information, read the [documentation](https://nextpyp.app/prismpyp/).

The code uses a [PyTorch implementation of SimSiam](https://github.com/facebookresearch/simsiam) distributed under the [Attribution-NonCommercial 4.0 International](https://github.com/facebookresearch/simsiam/blob/main/LICENSE) license.

The software is developed and maintained by the [Bartesaghi Lab](http://cryoem.cs.duke.edu) at [Duke University](http://www.duke.edu) and released under the [BSD 3-Clause](https://github.com/nextpyp/prismpyp/blob/main/LICENSE) license.