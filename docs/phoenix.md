## 🐦‍🔥 Visualizing 3D results in Phoenix

We use Phoenix to do interactive visualization and selection of good micrographs. For the best visualization experience, Phoenix should be installed locally (i.e.: not on a remote cluster). As such, the following steps are recommended to be carried out on your local machine.

Install Phoenix from either the ```.yml``` file:
```bash
wget https://github.com/nextpyp/prismpyp/blob/main/phoenix.yml -O phoenix.yml
conda env create -f phoenix.yml -n phoenix
conda activate phoenix
```

Or using pip:
```bash
conda create -n phoenix -c conda-forge python=3.8 pip
conda activate phoenix
wget https://github.com/nextpyp/prismpyp/blob/main/requirements-phoenix.txt -O requirements-phoenix.txt
python -m pip install -r requirements-phoenix.txt
```

The following instructions use the inference results generated from the real domain inputs.

1. Copy the output files (```output_dir/real/inference/data_for_export.parquet.zip```, ```output_dir/real/inference/zipped_thumbnail_images.tar.gz```) from 3D inference to your local machine (e.g., ```real```).
2. Make a folder for the thumbnail images:
```bash
mkdir -p real/thumbnail_images
```
3. Unzip the thumbnail images file:
```bash
tar -xvzf zipped_thumbnail_images.tar.gz -C real/thumbnail_images
```

4. Start an HTTP server to host the thumbnail images:
```bash
cd real
python -m http.server 5004
```

5. In another Bash window, download the visualization script and start the visualization code:
```bash
wget https://github.com/nextpyp/prismpyp/blob/main/scripts/visualizer.py

python visualizer.py \
    real/data_for_export.parquet.zip \
    --port 5004 \
    --which-embedding umap
```

Make sure that the port used in steps 4 and 5 are the same!

This step will output something like this to the terminal:
```bash
                                          embeddings  
0  [-0.07116878777742386, -0.02092127315700054, 0...  
1  [0.03098396770656109, -0.012207355350255966, 0...  
2  [-0.001951615558937192, -0.059190645813941956,...  
3  [0.04287125542759895, 0.055738892406225204, -0...  
4  [-0.03130991384387016, -0.024252604693174362, ...  
🌍 To view the Phoenix app in your browser, visit http://localhost:54116/
📺 To view the Phoenix app in a notebook, run `px.active_session().view()`
📖 For more information on how to use Phoenix, check out https://docs.arize.com/phoenix
```

You should now be able to access the interactive session by visiting ```http://localhost:54116/``` in a web browser.

The same steps can be followed to do interactive selection on the Fourier domain inputs:
1. Copy the output files (```output_dir/fft/inference/data_for_export.parquet.zip```, ```output_dir/fft/inference/zipped_thumbnail_images.tar.gz```) from 3D inference to your local machine (e.g., ```fft```).
2. Make a folder for the thumbnail images:
```bash
mkdir -p fft/thumbnail_images_dir
```
3. Unzip the thumbnail images file:
```bash
tar -xvzf zipped_thumbnail_images.tar.gz -C fft/thumbnail_images_dir
```

4. In another Bash window, start an HTTP server to host the thumbnail images:
```bash
cd fft
screen
python -m http.server 5004
```

5. In one Bash window, start the visualization code:
```bash
python visualizer.py \
    fft/data_for_export.parquet.zip \
    --port 5004 \
    --which-embedding umap
```

For both domains, the lasso selection output will be saved and downloaded to a ```.parquet``` file, e.g., ```real/real_good_export.parquet``` or ```fft/fft_good_export.parquet```