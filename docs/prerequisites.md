# Prerequisites

This section outlines the computational and data-processing requirements needed to run **prismPYP**.

## ⚙️ Computational Prerequisites

Because prismPYP depends on **CUDA** and **PyTorch-GPU**, all installation, training, and embedding-generation steps must be performed on a machine with a compatible **GPU**.

> 🧠 This setup has been tested with **PyTorch 2.4.0** and **CUDA 12.1** on an **NVIDIA A6000** GPU.

## 🧪 Data Processing Prerequisites

As input, prismPYP expects `.mrc` files containing both **micrographs** (i.e., frame averages) and their corresponding **power spectra** (i.e., output of CTF estimation).  

In addition, traditional metrics such as **CTF fit**, **estimated resolution**, **relative ice thickness**, and **mean defocus** can be used during interactive visualization to help identify high-quality subsets.

Before running prismPYP, micrographs must be pre-processed through **CTF estimation** to produce the necessary data.  

This pre-processing can be performed in either [nextPYP](https://nextpyp.app) or [cryoSPARC](https://cryosparc.com):

=== "nextPYP"

    The following steps from the [single-particle workflow](https://nextpyp.app/files/pyp/latest/docs/tutorials/spa_empiar_10025.html) must be executed:

    1. **Import Raw Data**  
    2. **Pre-Processing**

=== "cryoSPARC"

    The following steps must be executed:

    1. **Import Movies/Micrographs**  
    2. **Patch Motion Estimation** (if importing movies)  
    3. **Patch CTF Estimation** (optional)
    4. **CTFFIND4**