# Visualizing 3D Results in Phoenix 🐦‍🔥 

## 1. Prepare Real-Domain Visualization

These instructions assume inference results from **real-domain inputs**.

### a. Copy output files to your local machine:

   ```bash
   cp output_dir/real/inference/data_for_export.parquet.zip real/
   cp output_dir/real/inference/zipped_thumbnail_images.tar.gz real/
   ```

### b. Create a folder for the thumbnail images:

   ```bash
   mkdir -p real/thumbnail_images
   ```

### c. Extract thumbnails:

   ```bash
   tar -xvzf zipped_thumbnail_images.tar.gz -C real/thumbnail_images
   ```

### d. Start a local HTTP server to host thumbnails:

   ```bash
   cd real
   screen
   python -m http.server 5004
   ```

### e. In another terminal, download and launch the visualizer:

   ```bash
   wget https://github.com/nextpyp/prismpyp/blob/main/scripts/visualizer.py

   python visualizer.py \
      real/data_for_export.parquet.zip \
      --port 5004 \
      --which-embedding umap
   ```

!!! warning

   Make sure that the ports used in steps 4 and 5 are identical.

When launched successfully, you’ll see output like:

```bash
🌍 To view the Phoenix app in your browser, visit http://localhost:54116/
📺 To view the Phoenix app in a notebook, run `px.active_session().view()`
📖 For more information on how to use Phoenix, check out https://docs.arize.com/phoenix
```

You can now access the interactive visualization at [http://localhost:54116/](http://localhost:54116/).

## 2. Visualize Fourier-Domain Results

You can repeat the same steps for Fourier-domain inputs.

### a. Copy the output files:

   ```bash
   cp output_dir/fft/inference/data_for_export.parquet.zip fft/
   cp output_dir/fft/inference/zipped_thumbnail_images.tar.gz fft/
   ```

### b. Create a folder for the thumbnail images:

   ```bash
   mkdir -p fft/thumbnail_images_dir
   ```

### c. Extract thumbnails:

   ```bash
   tar -xvzf zipped_thumbnail_images.tar.gz -C fft/thumbnail_images_dir
   ```

### d. Start an HTTP server (you can use `screen` if desired):

   ```bash
   cd fft
   screen
   python -m http.server 5004
   ```

### e. Launch the visualization:

   ```bash
   python visualizer.py \
      fft/data_for_export.parquet.zip \
      --port 5004 \
      --which-embedding umap
   ```

## 3. Lasso Selections and Output

For both domains, interactive **lasso selections** will be saved as downloadable `.parquet` files:

| Domain | Output File |
|---------|--------------|
| Real | `real/real_good_export.parquet` |
| Fourier | `fft/fft_good_export.parquet` |

These outputs contain the selected subset of high-quality micrographs and can be used for further filtering or downstream processing.