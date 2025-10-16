# ⚙️ Setting Up the Environment

This section describes how to prepare your environment for running **prismPYP**.  
Because of its dependency on **CUDA** and **PyTorch-GPU**, all installation steps must be performed on a machine with a compatible **GPU**.

> 💡 This setup has been tested with **PyTorch 2.4.0** and **CUDA 12.1**.  

---

## 🐍 1. Install Conda

Download and install [**Miniconda**](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [**Anaconda**](https://www.anaconda.com/).

Verify your installation:
```bash
conda --version
```

---

## 📦 2. Clone the Repository

```bash
git clone git@github.com:nextpyp/prismpyp.git
cd prismpyp
```

If you don’t have SSH access set up, you can also clone via HTTPS:
```bash
git clone https://github.com/nextpyp/prismpyp.git
cd prismpyp
```

---

## 🧩 3. Create and Activate the Environment

Using **Conda** and **pip**:

```bash
conda create -n prismpyp -c conda-forge python=3.12 pip
conda activate prismpyp
```

---

## ⚡ 4. Install prismPYP and Dependencies

Install the core package and dependencies with GPU support:

```bash
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
```

> 🧠 This ensures PyTorch is installed with the correct CUDA 12.1 build.

Then install **FAISS-GPU** (via Conda, since pip wheels for Python 3.12 are unsupported):

```bash
conda install -c pytorch -c conda-forge faiss-gpu=1.9.0
```

---

Your environment is now ready for use! 🎉  
Continue to the next step to learn how to **gather and organize input data** for prismPYP.

---

### Next Steps
➡️ [Gathering Input Data →](metadata.md)
