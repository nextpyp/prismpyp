# Setting Up the Environment

This section describes how to prepare your environment for running **prismPYP**.  

## 1. Install Conda

Download and install [**Miniconda**](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [**Anaconda**](https://www.anaconda.com/).

Verify your installation:
```bash
conda --version
```

## 2. Clone the prismPYP Repository

```bash
git clone https://github.com/nextpyp/prismpyp.git
cd prismpyp
```

## 3. Create and Activate the prismPYP Environment

Using **Conda** and **pip**:

```bash
conda create -n prismpyp -c conda-forge python=3.12 pip
conda activate prismpyp
```

## 4. Install prismPYP and Dependencies

Install the core package and dependencies with GPU support:

```bash
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
```

> This ensures PyTorch is installed with the correct CUDA 12.1 build.

Then install **FAISS-GPU** (via Conda, since pip wheels for Python 3.12 are unsupported):

```bash
conda install -c pytorch -c conda-forge faiss-gpu=1.9.0
```

## 5. Create and Install the Phoenix Environment
[Phoenix](https://phoenix.arize.com/) enables **interactive 3D visualization and manual selection** of micrographs directly within the embedding space.  

For the best performance, Phoenix should be installed and run **locally** (not on a remote cluster).

> 💡 Phoenix provides an intuitive interface to explore embeddings, filter high-quality micrographs, and export subsets for further refinement.

Install Phoenix using either the provided Conda environment file or `pip` (make sure Conda and pip are installed in your local environment).

=== "Conda YAML Installation"

    ```bash
    wget https://github.com/nextpyp/prismpyp/blob/main/phoenix.yml -O phoenix.yml
    conda env create -f phoenix.yml -n phoenix
    conda activate phoenix
    ```

=== "Pip Installation"

    ```bash
    conda create -n phoenix -c conda-forge python=3.8 pip
    conda activate phoenix
    wget https://github.com/nextpyp/prismpyp/blob/main/requirements-phoenix.txt -O requirements-phoenix.txt
    python -m pip install -r requirements-phoenix.txt
    ```


Your environments are now ready for use!

Continue to the next step to learn how to **prepare and organize input data** for prismPYP.