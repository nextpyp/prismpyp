hide:
    - navigation

# 🔮 prismPYP: Power-spectrum and image domain learning for self-supervised micrograph evaluation

**prismPYP** implements a SimSiam-based self-supervised pipeline for classifying cryo-EM micrographs using both real-space and Fourier-space features.  

The goal is to automatically uncover image-quality categories such as vitreous ice, crystalline ice, contaminants, and support film entirely **without labels**.

The framework builds on a [PyTorch implementation of SimSiam](https://github.com/facebookresearch/simsiam)  
(distributed under the [Attribution-NonCommercial 4.0 International](https://github.com/facebookresearch/simsiam/blob/main/LICENSE) license).

![Phoenix visualization](assets/phoenix_example.png)

