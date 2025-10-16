# 🔮 prismPYP: Power-spectrum and image domain learning for self-supervised micrograph evaluation

**prismPYP** implements a SimSiam-based self-supervised pipeline for classifying cryo-EM micrographs using both real-space and Fourier-space features.  
The goal is to automatically uncover image-quality categories such as vitreous ice, crystalline ice, contaminants, and support film entirely **without labels**.

The framework builds on a [PyTorch implementation of SimSiam](https://github.com/facebookresearch/simsiam)  
(distributed under the [Attribution-NonCommercial 4.0 International](https://github.com/facebookresearch/simsiam/blob/main/LICENSE) license).

---

## 📘 Documentation

Use the links below to explore each part of the workflow:

- [Installation](env_setup.md)
- [Gathering Input Data](metadata.md)
- [Label-Free Feature Learning](train.md)
- [2D Embedding Generation](eval2d.md)
- [3D Embedding Generation](eval3d.md)
- [Interactive Visualization (Phoenix)](phoenix.md)
- [Dual-Domain Filtering](intersect.md)

---

## 🧪 Download the Test Data

You can test prismPYP using the **`example_data.tar.gz`** archive from [Zenodo](https://doi.org/10.5281/zenodo.17161604),  
which contains micrograph and power-spectrum images plus metadata from EMPIAR-10379.

```bash
mkdir example_data
tar -xvzf example_data.tar.gz -C example_data
```

This command extracts the data into an example_data/ folder containing:

```
example_data/
    ├── pkl/                      # metadata from NextPYP preprocessing
    ├── webp/                     # 512×512 images (micrographs + power spectra)
    ├── J7_exposures_accepted_exported.cs
    ├── sp-preprocessing-*.micrographs
    └── .pyp_config.toml
```

## 📦 Zenodo Files

The Zenodo link (https://doi.org/10.5281/zenodo.17161604) contains the following files:

* ```model_weights.tar.gz```: Trained model weights for the real domain input (```real_model_best.pth.tar```) and the Fourier domain input (```fft_model_best.pth.tar```)
* ```fft_good_export.parquet```: Data points that have high-quality features in the Fourier domain
* ```real_good_export.parquet```: Data points that have high-qualtiy features in the real domain

By taking the intersection of ```fft_good_export.parquet``` and ```real_good_export.parquet```, you can obtain the 862 micrographs that were used to obtain the 2.9&nbsp;Å structure in the paper.