## ⚙️ Setting up the environment
Because of the dependency on CUDA and PyTorch-GPU, all of the following instructions should be run from a computer with a GPU. 

This setup process and code has been tested with ```PyTorch/2.4.0``` and  ```cuda/12.1```.

### Install conda
   Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [Anaconda](https://www.anaconda.com/).

### Clone the repository
   ```bash
   git clone git@github.com:nextpyp/prismpyp.git
   cd prismpyp
   ```
   
### Set up the environment
   
   Using pip:
   ```bash
   conda create -n prismpyp -c conda-forge python=3.12 pip
   conda activate prismpyp
   
   python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
   
   conda install -c pytorch -c conda-forge faiss-gpu=1.9.0 # The pip wheel for faiss-gpu does not support python/3.12
   ```

  Your Conda environment should now be active.