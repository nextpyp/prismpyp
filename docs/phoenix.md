# 🐦‍🔥 Visualizing 3D Results in Phoenix

Phoenix enables **interactive 3D visualization and manual selection** of micrographs directly within the embedding space.  
For the best performance, Phoenix should be installed and run **locally** (not on a remote cluster).

> 💡 Phoenix provides an intuitive interface to explore embeddings, filter high-quality micrographs, and export subsets for further refinement.

---

## ⚙️ 1. Install Phoenix

Install Phoenix using either the provided Conda environment file or `pip`.

### Option A — Conda YAML Installation

```bash
wget https://github.com/nextpyp/prismpyp/blob/main/phoenix.yml -O phoenix.yml
conda env create -f phoenix.yml -n phoenix
conda activate phoenix
```

### Option B — Pip Installation

```bash
conda create -n phoenix -c conda-forge python=3.8 pip
conda activate phoenix
wget https://github.com/nextpyp/prismpyp/blob/main/requirements-phoenix.txt -O requirements-phoenix.txt
python -m pip install -r requirements-phoenix.txt
```

---

## 🧩 2. Prepare Real-Domain Visualization

These instructions assume inference results from **real-domain inputs**.

1. Copy output files to your local machine:

   ```bash
   cp output_dir/real/inference/data_for_export.parquet.zip real/
   cp output_dir/real/inference/zipped_thumbnail_images.tar.gz real/
   ```

2. Create a folder for the thumbnail images:

   ```bash
   mkdir -p real/thumbnail_images
   ```

3. Extract thumbnails:

   ```bash
   tar -xvzf zipped_thumbnail_images.tar.gz -C real/thumbnail_images
   ```

4. Start a local HTTP server to host thumbnails:

   ```bash
   cd real
   python -m http.server 5004
   ```

5. In another terminal, download and launch the visualizer:

   ```bash
   wget https://github.com/nextpyp/prismpyp/blob/main/scripts/visualizer.py

   python visualizer.py \
      real/data_for_export.parquet.zip \
      --port 5004 \
      --which-embedding umap
   ```

> ⚠️ Make sure that the ports used in steps 4 and 5 are identical.

When launched successfully, you’ll see output like:

```bash
🌍 To view the Phoenix app in your browser, visit http://localhost:54116/
📺 To view the Phoenix app in a notebook, run `px.active_session().view()`
📖 For more information on how to use Phoenix, check out https://docs.arize.com/phoenix
```

You can now access the interactive visualization at [http://localhost:54116/](http://localhost:54116/).

---

## 🔄 3. Visualize Fourier-Domain Results

You can repeat the same steps for Fourier-domain inputs.

1. Copy the output files:

   ```bash
   cp output_dir/fft/inference/data_for_export.parquet.zip fft/
   cp output_dir/fft/inference/zipped_thumbnail_images.tar.gz fft/
   ```

2. Create a folder for the thumbnail images:

   ```bash
   mkdir -p fft/thumbnail_images_dir
   ```

3. Extract thumbnails:

   ```bash
   tar -xvzf zipped_thumbnail_images.tar.gz -C fft/thumbnail_images_dir
   ```

4. Start an HTTP server (you can use `screen` if desired):

   ```bash
   cd fft
   screen
   python -m http.server 5004
   ```

5. Launch the visualization:

   ```bash
   python visualizer.py \
      fft/data_for_export.parquet.zip \
      --port 5004 \
      --which-embedding umap
   ```

---

## 💾 4. Lasso Selections and Output

For both domains, interactive **lasso selections** will be saved as downloadable `.parquet` files:

| Domain | Output File |
|---------|--------------|
| Real | `real/real_good_export.parquet` |
| Fourier | `fft/fft_good_export.parquet` |

These outputs contain the selected subset of high-quality micrographs and can be used for further filtering or downstream processing.

---

### Next Steps
⬅️ [Back: 3D Embedding Generation](eval3d.md) | ➡️ [Next: Dual-Domain Filtering](intersect.md)
